"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

"""
PASS 1 – Jasp→Quake lowering and QuantumState elimination.

This pass performs two tightly coupled transformations on the xDSL module
produced by :func:`~qrisp.jasp.mlir.mlir_emission.jaspr_to_mlir`:

1. **Jasp-op rewriting** – each ``jasp.*`` op is replaced by the equivalent
   Quake op(s).  The mapping is:

   ========================  ==========================================
   Jasp op                   Quake op(s)
   ========================  ==========================================
   ``jasp.create_qubits``    ``quake.alloca !quake.veq<?>``
   ``jasp.get_qubit``        ``quake.extract_ref``
   ``jasp.get_size``         ``quake.veq_size``
   ``jasp.slice``            ``quake.subveq``
   ``jasp.fuse``             ``quake.concat``
   ``jasp.delete_qubits``    ``quake.dealloc``
   ``jasp.quantum_gate``     ``quake.<gate>``  (dispatched via gate_mapping)
   ``jasp.measure``          ``quake.mz`` + ``quake.discriminate``
   ``jasp.reset``            ``quake.reset``
   ``jasp.create_qk``        *(dropped; function attr added)*
   ``jasp.consume_qk``       *(dropped)*
   ``jasp.parity``           *(left in place / not lowered)*
   ========================  ==========================================

2. **QuantumState threading elimination** – after the jasp-op rewrites,
   ``!jasp.QuantumState`` values are removed from:

   * op operands and results (done in-line during rewriting);
   * ``scf.while`` / ``scf.if`` block arguments and yield/condition operands;
   * function argument lists and return types.

Algorithm
---------
Ops are processed in **forward order** within each block.  For each jasp op,
the output ``!jasp.QuantumState`` result (``qst_out``) is threaded backwards
by calling ``qst_out.replace_all_uses_with(qst_in)`` before erasing the op.
This ensures that downstream ops that were waiting for the "updated" qst now
reference the *same* qst they supplied (or the initial function-arg qst),
making all intermediate qst values dead.

SCF ops are handled inside-out:

1. Block arg types are updated first (``QubitArray → !quake.veq<?>``,
   ``Qubit → !quake.ref``; QuantumState args are left for later removal).
2. Inner ops are then processed recursively.
3. Finally qst is stripped from the SCF op's structural components.
"""


import warnings
from typing import Sequence

from xdsl.dialects import arith, func, scf, tensor
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    FunctionType,
    IntegerAttr,
    StringAttr,
    i1,
    i64,
    f64,
    i32,
)
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    OpResult,
    Region,
    SSAValue,
)
from xdsl.rewriter import Rewriter

from qrisp.jasp.mlir.quake_lowering.quake_dialect import (
    AllocaVeqOp,
    ConcatOp,
    DeallocOp,
    DiscriminateOp,
    ExtractRefOp,
    MzOp,
    QuakeRefType,
    QuakeVeqType,
    QuakeMeasureType,
    ResetOp,
    SubVeqOp,
    VeqSizeOp,
    make_gate_op,
)
from qrisp.jasp.mlir.quake_lowering.gate_mapping import get_gate_info

# ---------------------------------------------------------------------------
# xDSL version compatibility
# ---------------------------------------------------------------------------


def _replace_all_uses_with(val: SSAValue, new_val: SSAValue) -> None:
    """Replace all uses of *val* with *new_val*.

    Provides compatibility between xDSL < 0.57 (which uses
    ``SSAValue.replace_by``) and xDSL >= 0.57 (which uses
    ``SSAValue.replace_all_uses_with``).
    """
    if hasattr(val, "replace_all_uses_with"):
        val.replace_all_uses_with(new_val)
    else:  # xDSL < 0.57
        val.replace_by(new_val)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers to identify Jasp types
# ---------------------------------------------------------------------------


def _is_qst(t: Attribute) -> bool:
    """Return True if *t* is ``!jasp.QuantumState``."""
    return t.name == "jasp.QuantumState"


def _is_qubit_array(t: Attribute) -> bool:
    """Return True if *t* is ``!jasp.QubitArray``."""
    return t.name == "jasp.QubitArray"


def _is_qubit(t: Attribute) -> bool:
    """Return True if *t* is ``!jasp.Qubit``."""
    return t.name == "jasp.Qubit"


def _is_jasp_type(t: Attribute) -> bool:
    return _is_qst(t) or _is_qubit_array(t) or _is_qubit(t)


def _quake_type_for(jasp_type: Attribute) -> Attribute | None:
    """Map a Jasp qubit type to its Quake equivalent, or None for qst."""
    if _is_qubit_array(jasp_type):
        return QuakeVeqType()
    if _is_qubit(jasp_type):
        return QuakeRefType()
    return None  # QuantumState → dropped


# ---------------------------------------------------------------------------
# Numeric-type helpers
# ---------------------------------------------------------------------------


def _is_qubit_type(t: Attribute) -> bool:
    """Return True if *t* is any qubit/veq type (Jasp or Quake)."""
    return (
        isinstance(t, (QuakeRefType, QuakeVeqType))
        or _is_qubit(t)
        or _is_qubit_array(t)
    )


def _is_numeric_type(t: Attribute) -> bool:
    """Return True if *t* is a scalar or rank-0 tensor of int/float."""
    from xdsl.dialects.builtin import IntegerType, TensorType

    scalar = t
    if isinstance(t, TensorType):
        if t.get_shape():  # only rank-0
            return False
        scalar = t.element_type
    return isinstance(scalar, IntegerType) or _is_float_type(scalar)


def _is_float_type(t: Attribute) -> bool:
    """Return True for any xDSL float type."""
    from xdsl.dialects.builtin import (
        Float16Type,
        Float32Type,
        Float64Type,
        BFloat16Type,
    )

    return isinstance(t, (Float16Type, Float32Type, Float64Type, BFloat16Type))


def _scalar_type_of(t: Attribute) -> Attribute:
    """Return the scalar element type (unwrap rank-0 tensor if needed)."""
    from xdsl.dialects.builtin import TensorType

    if isinstance(t, TensorType) and not t.get_shape():
        return t.element_type
    return t


def _coerce_to_f64(val: SSAValue, block: Block, insert_before) -> SSAValue:
    """Extract from tensor if needed, then cast int→f64 if needed.

    Returns an SSAValue of type f64.
    """
    from xdsl.dialects.builtin import IntegerType, TensorType

    # Step 1: unwrap rank-0 tensor → scalar
    scalar_type = _scalar_type_of(val.type)
    scalar = _extract_scalar(val, scalar_type, block, insert_before)

    # Step 2: already f64 → done
    if scalar.type == f64:
        return scalar

    # Step 3: other float → fpext/fptrunc to f64
    if _is_float_type(scalar.type):
        cast = arith.ExtFOp(scalar, f64)  # or SIToFPOp etc.
        block.insert_ops_before([cast], insert_before)
        return cast.result

    # Step 4: integer → sitofp to f64
    if isinstance(scalar.type, IntegerType):
        cast = arith.SIToFPOp(scalar, f64)
        block.insert_ops_before([cast], insert_before)
        return cast.result

    raise ValueError(f"Cannot coerce {scalar.type} to f64")


def _normalize_index_for_veq(
    veq: SSAValue, idx: SSAValue, block: Block, insert_before_op
) -> SSAValue:
    """Implement Python-style negative indexing on a veq index.

    If idx < 0, return idx + size(veq), else idx.

    All ops are inserted before `insert_before_op`.
    Assumes `idx` is an i64 scalar.
    """
    # len = quake.veq_size %veq
    size = VeqSizeOp(veq)

    zero = arith.ConstantOp(IntegerAttr(0, 64))
    is_neg = arith.CmpiOp(idx, zero.result, "slt")  # idx < 0 ?
    idx_plus_size = arith.AddiOp(idx, size.result)  # idx + len
    norm = arith.SelectOp(is_neg.result, idx_plus_size.result, idx)

    block.insert_ops_before(
        [size, zero, is_neg, idx_plus_size, norm],
        insert_before_op,
    )
    return norm.result


# ---------------------------------------------------------------------------
# Tensor-extract helper
# ---------------------------------------------------------------------------


def _extract_scalar(
    val: SSAValue, scalar_type: Attribute, block: Block, insert_before_op
) -> SSAValue:
    """Insert a ``tensor.extract %val[] : tensor<T>`` op and return the result.

    If *val* is already of *scalar_type* (not a tensor), return it unchanged.
    """
    if val.type == scalar_type:
        return val
    extract = tensor.ExtractOp(val, [], scalar_type)
    block.insert_ops_before([extract], insert_before_op)
    return extract.result


def _wrap_scalar(
    val: SSAValue, tensor_type: Attribute, block: Block, insert_before_op
) -> SSAValue:
    """Insert ``tensor.from_elements %val : tensor<T>`` and return the result.

    If *val* is already of *tensor_type*, return it unchanged.
    """
    if val.type == tensor_type:
        return val
    from_elem = tensor.FromElementsOp(operands=[[val]], result_types=[tensor_type])
    block.insert_ops_before([from_elem], insert_before_op)
    return from_elem.result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def lower_jasp_to_quake(module, execution_mode: str = "run") -> None:
    """In-place PASS 1: eliminate QuantumState and lower all Jasp ops to Quake.

    Parameters
    ----------
    module:
        An xDSL ``builtin.ModuleOp`` containing the Jasp IR emitted by
        :func:`~qrisp.jasp.mlir.mlir_emission.jaspr_to_mlir`.  The module is
        modified in-place.
    execution_mode:
        ``"run"`` (default) packs array measurement results into an ``i64``
        return value, suitable for ``cudaq.run``.  ``"sample"`` emits a raw
        ``quake.mz`` on the whole veq and strips classical return values from
        the function, suitable for ``cudaq.sample``.
    """
    for op in list(module.body.blocks[0].ops):
        if op.name == "func.func":
            _process_func(op, execution_mode)


# ---------------------------------------------------------------------------
# Function-level processing
# ---------------------------------------------------------------------------


def _process_func(func_op, execution_mode: str = "run") -> None:
    """Process a single ``func.func`` op: rewrite body, fix signature."""
    # Check if this function contains create_quantum_kernel → cudaq.kernel
    is_quantum = _func_has_any_jasp(func_op)

    for block in func_op.body.blocks:
        _process_block(block, execution_mode)

    _fix_func_signature(func_op, is_quantum, execution_mode)


def _func_has_any_jasp(func_op) -> bool:
    """Return True if the function body contains any jasp.* op."""
    for block in func_op.body.blocks:
        for op in block.ops:
            if op.name.startswith("jasp."):
                return True
    return False


def _fix_func_signature(func_op, mark_cudaq_kernel: bool = False, execution_mode: str = "run") -> None:
    """Remove ``!jasp.QuantumState`` from function argument/return types.

    Also updates the entry block's argument list and adds ``cudaq.kernel``
    + ``cudaq.entrypoint`` attributes when *mark_cudaq_kernel* is True.
    """
    old_ftype: FunctionType = func_op.function_type

    # Compute new input types (drop qst, update qubit types)
    new_inputs = []
    entry_block = func_op.body.blocks.first
    if entry_block is not None:
        for arg in list(entry_block.args):
            if _is_qst(arg.type):
                # Remove from block - the arg should have 0 uses by now
                entry_block.erase_arg(arg, safe_erase=False)
            elif _is_qubit_array(arg.type):
                Rewriter.replace_value_with_new_type(arg, QuakeVeqType())
                new_inputs.append(QuakeVeqType())
            elif _is_qubit(arg.type):
                Rewriter.replace_value_with_new_type(arg, QuakeRefType())
                new_inputs.append(QuakeRefType())
            else:
                new_inputs.append(arg.type)
    else:
        new_inputs = [t for t in old_ftype.inputs.data if not _is_qst(t)]

    # Compute new return types (drop qst, convert qubit types).
    # For sample execution_mode the kernel must return void – cudaq.sample collects
    # measurement results via the runtime, not as classical return values.
    if execution_mode == "sample":
        new_outputs = []
    else:
        new_outputs = [
            _quake_type_for(t) or t for t in old_ftype.outputs.data if not _is_qst(t)
        ]

    new_ftype = FunctionType.from_lists(new_inputs, new_outputs)
    func_op.function_type = new_ftype

    # if mark_cudaq_kernel:
    #    func_op.attributes["cudaq.kernel"] = StringAttr("true")
    #    func_op.attributes["cudaq.entrypoint"] = StringAttr("true")
    func_op.attributes["cudaq.kernel"] = StringAttr("true")

    # Only apply entrypoint to the host-callable main function
    if func_op.sym_name.data == "main":
        func_op.attributes["cudaq.entrypoint"] = StringAttr("true")


# ---------------------------------------------------------------------------
# Block-level processing
# ---------------------------------------------------------------------------


def _process_block(block: Block, execution_mode: str = "run") -> None:
    """Process all ops in *block* (forward order)."""
    for op in list(block.ops):
        _process_op(op, block, execution_mode)


# ---------------------------------------------------------------------------
# Op dispatch
# ---------------------------------------------------------------------------


def _process_op(op, block: Block, execution_mode: str = "run") -> None:
    """Dispatch a single op for lowering."""
    n = op.name

    if n == "jasp.create_quantum_kernel":
        _lower_create_quantum_kernel(op, block)
    elif n == "jasp.consume_quantum_kernel":
        _lower_consume_quantum_kernel(op, block)
    elif n == "jasp.create_qubits":
        _lower_create_qubits(op, block)
    elif n == "jasp.get_qubit":
        _lower_get_qubit(op, block)
    elif n == "jasp.get_size":
        _lower_get_size(op, block)
    elif n == "jasp.slice":
        _lower_slice(op, block)
    elif n == "jasp.fuse":
        _lower_fuse(op, block)
    elif n == "jasp.delete_qubits":
        _lower_delete_qubits(op, block)
    elif n == "jasp.quantum_gate":
        _lower_quantum_gate(op, block)
    elif n == "jasp.measure":
        _lower_measure(op, block, execution_mode)
    elif n == "jasp.reset":
        _lower_reset(op, block)
    elif n == "jasp.parity":
        # Intentionally not lowered – left in place.
        pass
    elif n == "scf.while":
        _process_scf_while(op, block, execution_mode)
    elif n == "scf.if":
        _process_scf_if(op, block, execution_mode)
    elif n == "scf.for":
        _process_scf_for(op, block, execution_mode)
    elif n == "scf.index_switch":
        _process_scf_index_switch(op, block, execution_mode)
    elif n == "func.return":
        _fix_return_op(op, execution_mode)
    elif n == "func.call":
        _fix_call_op(op)
    # All other ops (arith, tensor, linalg, …): no action needed.


# ---------------------------------------------------------------------------
# QuantumState threading helper
# ---------------------------------------------------------------------------


def _thread_qst(op) -> None:
    """For ops with a QuantumState result, thread it backwards.

    Finds all ``!jasp.QuantumState`` results of *op* and replaces their uses
    with the corresponding ``!jasp.QuantumState`` operand (the *input* qst).
    This makes qst_out dead so the op can be safely erased after rewriting.

    If there is no matching qst input (e.g. ``create_quantum_kernel``), the
    qst output is erased with ``safe_erase=False`` so pending uses are cleared.
    """
    qst_inputs = [v for v in op.operands if _is_qst(v.type)]
    qst_outputs = [r for r in op.results if _is_qst(r.type)]

    for i, qst_out in enumerate(qst_outputs):
        if i < len(qst_inputs):
            _replace_all_uses_with(qst_out, qst_inputs[i])
        else:
            # No corresponding input – erase all uses (create_quantum_kernel)
            qst_out.erase(safe_erase=False)


# ---------------------------------------------------------------------------
# Jasp op lowerings
# ---------------------------------------------------------------------------


def _lower_create_quantum_kernel(op, block: Block) -> None:
    """``jasp.create_quantum_kernel`` → dropped; mark enclosing func as kernel."""
    _thread_qst(op)
    Rewriter.erase_op(op)


def _lower_consume_quantum_kernel(op, block: Block) -> None:
    """``jasp.consume_quantum_kernel`` → dropped; replace bool result with True."""
    # Result is tensor<i1> – replace with constant True
    if op.results:
        true_const = arith.ConstantOp(
            DenseIntOrFPElementsAttr.from_list(op.results[0].type, [1])
        )
        block.insert_ops_before([true_const], op)
        _replace_all_uses_with(op.results[0], true_const.result)
    Rewriter.erase_op(op)


def _lower_create_qubits(op, block: Block) -> None:
    """``jasp.create_qubits %n_tensor, %qst`` → ``quake.alloca !quake.veq<?>[%n]``."""
    # Operands: [amount: tensor<i64>, qst_in]
    amount_tensor = op.operands[0]

    # Extract i64 scalar from tensor<i64>
    n = _extract_scalar(amount_tensor, i64, block, op)

    alloca = AllocaVeqOp(n)
    block.insert_ops_before([alloca], op)

    # Map QubitArray result → veq result
    for r in op.results:
        if _is_qubit_array(r.type):
            _replace_all_uses_with(r, alloca.result)

    _thread_qst(op)
    Rewriter.erase_op(op)


def _lower_get_qubit(op, block: Block) -> None:
    """``jasp.get_qubit %arr, %idx_tensor`` → ``quake.extract_ref %veq[%idx]``."""
    arr = op.operands[0]
    idx_tensor = op.operands[1]

    idx = _extract_scalar(idx_tensor, i64, block, op)

    # Normalize index: if idx < 0, use idx + size(arr)
    norm_idx = _normalize_index_for_veq(arr, idx, block, op)

    extract = ExtractRefOp(arr, norm_idx)
    block.insert_ops_before([extract], op)

    for r in op.results:
        if _is_qubit(r.type):
            _replace_all_uses_with(r, extract.result)

    Rewriter.erase_op(op)


def _lower_get_size(op, block: Block) -> None:
    """``jasp.get_size %arr`` → ``quake.veq_size %veq`` (returns tensor<i64>)."""
    arr = op.operands[0]
    veq_size = VeqSizeOp(arr)
    block.insert_ops_before([veq_size], op)

    # Wrap i64 result back into tensor<i64> for compatibility with downstream ops
    from xdsl.dialects.builtin import TensorType

    result_tensor_type = op.results[0].type
    wrapped = _wrap_scalar(veq_size.result, result_tensor_type, block, op)

    _replace_all_uses_with(op.results[0], wrapped)
    Rewriter.erase_op(op)


def _lower_slice(op, block: Block) -> None:
    """``jasp.slice %arr, %start, %end`` → ``quake.subveq %veq, %lo, %hi_inclusive``.

    jasp.slice uses exclusive upper bound (Python-style: [start, end)),
    while quake.subveq uses inclusive bounds [lo, hi].

    Assumptions:
    - start index is already a non-negative absolute index.
    - end index may be negative (Python slice semantics).
    """
    arr = op.operands[0]
    start_t = op.operands[1]
    end_t = op.operands[2]

    # start is already an absolute i64 index
    lo = _extract_scalar(start_t, i64, block, op)

    # end may be negative
    hi_raw = _extract_scalar(end_t, i64, block, op)

    # Get length of the veq: len = quake.veq_size %arr
    veq_size = VeqSizeOp(arr)

    # Constants 0 and 1
    zero = arith.ConstantOp(IntegerAttr(0, 64))
    one = arith.ConstantOp(IntegerAttr(1, 64))

    # hi_raw < 0 ?
    hi_is_neg = arith.CmpiOp(hi_raw, zero.result, "slt")

    # hi_plus_len = hi_raw + len
    hi_plus_len = arith.AddiOp(hi_raw, veq_size.result)

    # end_abs = hi_is_neg ? (hi_raw + len) : hi_raw
    hi_abs = arith.SelectOp(hi_is_neg.result, hi_plus_len.result, hi_raw)

    # Convert exclusive upper bound → inclusive upper bound: hi_inclusive = end_abs - 1
    hi_inclusive = arith.SubiOp(hi_abs.result, one.result)

    # Insert all arithmetic before the jasp.slice op
    block.insert_ops_before(
        [veq_size, zero, one, hi_is_neg, hi_plus_len, hi_abs, hi_inclusive],
        op,
    )

    # Lower to quake.subveq
    subveq = SubVeqOp(arr, lo, hi_inclusive.result)
    block.insert_ops_before([subveq], op)

    # Replace jasp slice result with the quake.subveq result
    for r in op.results:
        if _is_qubit_array(r.type):
            _replace_all_uses_with(r, subveq.result)

    Rewriter.erase_op(op)


def _lower_fuse(op, block: Block) -> None:
    """``jasp.fuse %a, %b`` → ``quake.concat %a, %b``."""
    operands = [v for v in op.operands if not _is_qst(v.type)]
    concat = ConcatOp(operands)
    block.insert_ops_before([concat], op)

    for r in op.results:
        if _is_qubit_array(r.type):
            _replace_all_uses_with(r, concat.result)

    Rewriter.erase_op(op)


def _lower_delete_qubits(op, block: Block) -> None:
    """``jasp.delete_qubits %arr, %qst`` → ``quake.dealloc %veq``."""
    arr = op.operands[0]
    dealloc = DeallocOp(arr)
    block.insert_ops_before([dealloc], op)

    _thread_qst(op)
    Rewriter.erase_op(op)


def _lower_quantum_gate(op, block: Block) -> None:
    """``jasp.quantum_gate "name" (%qubits, %params...), %qst`` → Quake gate op."""
    gate_name: str = op.attributes.get("gate_type") or op.attributes.get(
        "gate_name", StringAttr("")
    )
    if hasattr(gate_name, "data"):
        gate_name = gate_name.data

    # Split operands: qubits first, then float params, then qst
    qubit_operands = []
    param_operands = []
    for v in op.operands:
        if _is_qst(v.type):
            continue
        elif _is_qubit_type(v.type):
            qubit_operands.append(v)
        elif _is_numeric_type(v.type):
            param_operands.append(_coerce_to_f64(v, block, op))
        else:
            qubit_operands.append(v)

    gate_info = get_gate_info(gate_name)
    if gate_info is None:
        warnings.warn(
            f"Unsupported Jasp gate '{gate_name}' – removing jasp.quantum_gate from IR).",
            stacklevel=4,
        )
        _thread_qst(op)
        # Don't erase – leave the op (minus qst threading) in place.
        # Re-insert op without qst
        _strip_qst_from_op(op, block)
        Rewriter.erase_op(op)
        return

    # Determine controls vs targets
    num_ctrl = gate_info.num_controls
    if num_ctrl == -1:
        # All-but-last are controls (mcx family)
        controls = qubit_operands[:-1]
        targets = qubit_operands[-1:]
    elif num_ctrl == 0:
        controls = []
        targets = qubit_operands
    else:
        controls = qubit_operands[:num_ctrl]
        targets = qubit_operands[num_ctrl:]

    final_params = list(param_operands[: gate_info.num_params])

    # ---- Emit gate(s) ----------------------------------------------------
    if gate_info.emit is not None:
        # Multi-op decomposition path
        ops = gate_info.emit(controls, final_params, targets)
        if not ops:
            warnings.warn(
                f"Gate '{gate_name}' emit() returned empty list – skipping.",
                stacklevel=4,
            )
            _thread_qst(op)
            _strip_qst_from_op(op, block)
            return
        block.insert_ops_before(ops, op)
    else:
        # Single-op path
        gate_op = make_gate_op(gate_name, controls, final_params, targets)
        if gate_op is None:
            warnings.warn(
                f"Gate '{gate_name}' not in Quake gate class table – skipping.",
                stacklevel=4,
            )
            _thread_qst(op)
            _strip_qst_from_op(op, block)
            return
        block.insert_ops_before([gate_op], op)

    _thread_qst(op)
    Rewriter.erase_op(op)


def _strip_qst_from_op(op, block: Block) -> None:
    """Remove QuantumState from an op's operands in-place (fallback path)."""
    non_qst = [v for v in op.operands if not _is_qst(v.type)]
    op.operands = non_qst


def _lower_measure(op, block: Block, execution_mode: str = "run") -> None:
    """``jasp.measure %q, %qst`` → ``quake.mz %q`` + ``quake.discriminate``."""
    qubit_val = op.operands[0]  # !jasp.Qubit or !jasp.QubitArray

    mz = MzOp(qubit_val)
    new_ops: list = [mz]

    # Determine what type we need to output
    meas_result = op.results[0]  # tensor<i1> or tensor<i64>
    is_array = _is_qubit_array(qubit_val.type) or isinstance(
        qubit_val.type, QuakeVeqType
    )

    if not is_array and execution_mode != "sample":
        # Run execution_mode, single qubit: discriminate → i1, then wrap to tensor<i1>
        disc = DiscriminateOp(mz.result)
        new_ops.append(disc)
        block.insert_ops_before(new_ops, op)
        scalar_bit = disc.result

        # Wrap back to tensor<i1> if downstream expects it
        wrapped = _wrap_scalar(scalar_bit, meas_result.type, block, op)
        _replace_all_uses_with(meas_result, wrapped)
    elif not is_array and execution_mode == "sample":
        # Sample execution_mode, single qubit: emit raw quake.mz (→ !quake.measure) and
        # use a zero i1 placeholder to keep SSA valid.
        block.insert_ops_before(new_ops, op)
        zero_const = arith.ConstantOp(
            DenseIntOrFPElementsAttr.from_list(meas_result.type, [0])
        )
        block.insert_ops_before([zero_const], op)
        _replace_all_uses_with(meas_result, zero_const.result)
    elif execution_mode == "sample":
        # Sample execution_mode: emit a single quake.mz on the whole veq.
        # Its !cc.stdvec<!quake.measure> result is consumed by the cudaq runtime;
        # we do not need the classical value here.  A zero i64 placeholder keeps
        # SSA valid for all downstream passes; _fix_return_op will strip it from
        # the function return so the kernel ends up returning void.
        block.insert_ops_before(new_ops, op)
        zero_const = arith.ConstantOp(
            DenseIntOrFPElementsAttr.from_list(meas_result.type, [0])
        )
        block.insert_ops_before([zero_const], op)
        _replace_all_uses_with(meas_result, zero_const.result)
    else:
        # Run execution_mode: bit-pack discriminated bits into an i64 via scf.for loop.

        # 1. Get the size of the array
        veq_size = VeqSizeOp(qubit_val)
        block.insert_ops_before([veq_size], op)

        # 2. Set up loop constants (0, 1)
        c0 = arith.ConstantOp(IntegerAttr(0, 64))
        c1 = arith.ConstantOp(IntegerAttr(1, 64))
        block.insert_ops_before([c0, c1], op)

        # 3. Create the loop body block
        # Args: iv (index), acc (accumulated i64 value)
        loop_body = Block(arg_types=[i64, i64])
        iv, acc = loop_body.args

        # 4. Populate the loop body
        extract = ExtractRefOp(qubit_val, iv)
        mz_single = MzOp(extract.result)
        disc = DiscriminateOp(mz_single.result)

        # Zero-extend the i1 to i64
        extui = arith.ExtUIOp(disc.result, i64)

        # Shift the bit left by 'iv' positions (assuming little-endian packing)
        shift = arith.ShLIOp(extui.result, iv)

        # Accumulate using bitwise OR
        new_acc = arith.OrIOp(acc, shift.result)

        # Yield the new accumulator to the next iteration
        yield_op = scf.YieldOp(new_acc.result)

        loop_body.add_ops([extract, mz_single, disc, extui, shift, new_acc, yield_op])

        # 5. Create the scf.for loop
        for_op = scf.ForOp(
            c0.result, veq_size.result, c1.result, [c0.result], Region([loop_body])
        )
        block.insert_ops_before([for_op], op)

        # 6. Wrap the final packed i64 back to tensor<i64> for downstream compatibility
        wrapped = _wrap_scalar(for_op.results[0], meas_result.type, block, op)
        _replace_all_uses_with(meas_result, wrapped)

    _thread_qst(op)
    Rewriter.erase_op(op)


def _lower_reset(op, block: Block) -> None:
    """``jasp.reset %q, %qst`` → ``quake.reset %q``."""
    qubit_val = op.operands[0]
    reset_op = ResetOp(qubit_val)
    block.insert_ops_before([reset_op], op)
    _thread_qst(op)
    Rewriter.erase_op(op)


# ---------------------------------------------------------------------------
# func.return and func.call fixup
# ---------------------------------------------------------------------------


def _fix_return_op(op, execution_mode: str = "run") -> None:
    """Remove ``!jasp.QuantumState`` values from ``func.return``."""
    if execution_mode == "sample":
        # cudaq.sample kernels return void; drop all return operands.
        op.operands = []
        return
    non_qst = [v for v in op.operands if not _is_qst(v.type)]
    op.operands = non_qst


def _fix_call_op(op) -> None:
    """
    Remove ``!jasp.QuantumState`` values from ``func.call`` operands and results.
    Convert qubit types in the results to their Quake equivalents.
    """
    new_operands = [v for v in op.operands if not _is_qst(v.type)]

    # Convert result types: drop QST, map QubitArray→veq, Qubit→ref
    new_result_types = []
    for t in op.result_types:
        if _is_qst(t):
            continue
        elif _is_qubit_array(t):
            new_result_types.append(QuakeVeqType())
        elif _is_qubit(t):
            new_result_types.append(QuakeRefType())
        else:
            new_result_types.append(t)

    # Create the new call operation
    new_call = func.CallOp(op.callee, new_operands, new_result_types)

    # Map old results to new results, ignoring the deleted QuantumState results
    replacement_results = []
    new_res_idx = 0
    for old_res in op.results:
        if _is_qst(old_res.type):
            replacement_results.append(None)
        else:
            replacement_results.append(new_call.results[new_res_idx])
            new_res_idx += 1

    # Replace the op and update SSA users
    Rewriter.replace_op(op, new_call, replacement_results, safe_erase=False)


# ---------------------------------------------------------------------------
# SCF op processing
# ---------------------------------------------------------------------------


def _update_non_qst_block_arg_types(block: Block) -> None:
    """Update QubitArray/Qubit block args to their Quake equivalents."""
    for arg in list(block.args):
        if _is_qubit_array(arg.type):
            Rewriter.replace_value_with_new_type(arg, QuakeVeqType())
        elif _is_qubit(arg.type):
            Rewriter.replace_value_with_new_type(arg, QuakeRefType())
        # QuantumState args: leave for _strip_qst_from_* to handle


def _process_scf_while(while_op, outer_block: Block, execution_mode: str = "run") -> None:
    """Lower a ``scf.while`` op: update types, recurse, then strip qst."""
    # 1. Update block arg types in both regions
    for region in while_op.regions:
        for b in region.blocks:
            _update_non_qst_block_arg_types(b)

    # 2. Process inner blocks recursively
    for region in while_op.regions:
        for b in region.blocks:
            _process_block(b, execution_mode)

    # 3. Strip qst from the SCF while structure
    _strip_qst_from_while(while_op, outer_block)


def _strip_qst_from_while(while_op, outer_block: Block) -> None:
    """Remove QuantumState from a ``scf.while`` and reconstruct without it."""
    before_block = while_op.before_region.blocks.first
    after_block = while_op.after_region.blocks.first

    # a) Remove qst from scf.condition operands
    for op in before_block.ops:
        if op.name == "scf.condition":
            non_qst = [v for v in op.operands if not _is_qst(v.type)]
            op.operands = non_qst
            break

    # b) Remove qst from scf.yield operands (after-region)
    for op in after_block.ops:
        if op.name == "scf.yield":
            non_qst = [v for v in op.operands if not _is_qst(v.type)]
            op.operands = non_qst
            break

    # c) Erase qst block args (should now have 0 uses)
    for b in [before_block, after_block]:
        for arg in list(b.args):
            if _is_qst(arg.type):
                b.erase_arg(arg, safe_erase=False)

    # d) Collect non-qst operands and result types for the new while op
    # Also convert QubitArray/Qubit types to Quake equivalents.
    non_qst_ops = [v for v in while_op.arguments if not _is_qst(v.type)]
    non_qst_res_types = [
        _quake_type_for(t) or t for t in while_op.res.types if not _is_qst(t)
    ]

    # Detach regions and create new scf.while (if anything changed)
    has_qst = any(_is_qst(v.type) for v in while_op.arguments) or any(
        _is_qst(t) for t in while_op.res.types
    )
    if not has_qst:
        return  # Nothing to do

    # Save region references before detaching (detach changes the regions tuple)
    before_ref = while_op.regions[0]
    after_ref = while_op.regions[1]
    before = while_op.detach_region(before_ref)
    after = while_op.detach_region(after_ref)

    from xdsl.dialects.scf import WhileOp

    new_while = WhileOp(non_qst_ops, non_qst_res_types, before, after)

    # Map old non-qst results → new results; qst results → None (erase)
    new_results: list = []
    new_idx = 0
    for old_res in while_op.res:
        if _is_qst(old_res.type):
            new_results.append(None)
        else:
            new_results.append(new_while.res[new_idx])
            new_idx += 1

    Rewriter.replace_op(while_op, new_while, new_results, safe_erase=False)


def _process_scf_if(if_op, outer_block: Block, execution_mode: str = "run") -> None:
    """Lower a ``scf.if`` op: recurse into branches, then strip qst."""
    # 1. Update block arg types (scf.if branches typically have no block args,
    #    but handle in case of unusual lowering patterns)
    for region in if_op.regions:
        for b in region.blocks:
            _update_non_qst_block_arg_types(b)

    # 2. Process inner ops
    for region in if_op.regions:
        for b in region.blocks:
            _process_block(b, execution_mode)

    # 3. Strip qst from scf.if yields and reconstruct
    _strip_qst_from_if(if_op, outer_block)


def _strip_qst_from_if(if_op, outer_block: Block) -> None:
    """Remove QuantumState from a ``scf.if`` and reconstruct without it."""
    # Remove qst from yields in both branches
    for region in if_op.regions:
        for b in region.blocks:
            for op in b.ops:
                if op.name == "scf.yield":
                    non_qst = [v for v in op.operands if not _is_qst(v.type)]
                    op.operands = non_qst

    # Check if if_op has qst in result types
    qst_result_types = [t for t in if_op.result_types if _is_qst(t)]
    if not qst_result_types:
        return  # Nothing to do

    non_qst_res_types = [
        _quake_type_for(t) or t for t in if_op.result_types if not _is_qst(t)
    ]

    # Detach regions
    true_region = if_op.detach_region(if_op.regions[0])
    false_region = if_op.detach_region(if_op.regions[0])  # now index 0 again

    from xdsl.dialects.scf import IfOp

    cond = if_op.operands[0]
    new_if = IfOp(cond, non_qst_res_types, true_region, false_region)

    # Map results
    new_results: list = []
    new_idx = 0
    for old_res in if_op.results:
        if _is_qst(old_res.type):
            new_results.append(None)
        else:
            new_results.append(new_if.results[new_idx])
            new_idx += 1

    Rewriter.replace_op(if_op, new_if, new_results, safe_erase=False)


def _process_scf_for(for_op, outer_block: Block, execution_mode: str = "run") -> None:
    """Lower a ``scf.for`` op: update types, recurse, strip qst."""
    for region in for_op.regions:
        for b in region.blocks:
            _update_non_qst_block_arg_types(b)

    for region in for_op.regions:
        for b in region.blocks:
            _process_block(b, execution_mode)

    _strip_qst_from_for(for_op, outer_block)


def _strip_qst_from_for(for_op, outer_block: Block) -> None:
    """Remove QuantumState from a ``scf.for`` and reconstruct without it."""
    body_block = for_op.regions[0].blocks.first

    # Remove qst from scf.yield in body
    for op in body_block.ops:
        if op.name == "scf.yield":
            non_qst = [v for v in op.operands if not _is_qst(v.type)]
            op.operands = non_qst

    # Erase qst block args (induction variable is first arg, then iter args)
    for arg in list(body_block.args):
        if _is_qst(arg.type):
            body_block.erase_arg(arg, safe_erase=False)

    has_qst = any(_is_qst(t) for t in for_op.result_types)
    if not has_qst:
        return

    non_qst_init_args = [v for v in list(for_op.operands)[3:] if not _is_qst(v.type)]
    non_qst_res_types = [
        _quake_type_for(t) or t for t in for_op.result_types if not _is_qst(t)
    ]

    body = for_op.detach_region(for_op.regions[0])

    from xdsl.dialects.scf import ForOp

    new_for = ForOp(
        for_op.operands[0],  # lb
        for_op.operands[1],  # ub
        for_op.operands[2],  # step
        non_qst_init_args,
        body,
    )

    new_results: list = []
    new_idx = 0
    for old_res in for_op.results:
        if _is_qst(old_res.type):
            new_results.append(None)
        else:
            new_results.append(new_for.res[new_idx])
            new_idx += 1

    Rewriter.replace_op(for_op, new_for, new_results, safe_erase=False)


def _process_scf_index_switch(op, outer_block: Block, execution_mode: str = "run") -> None:
    """Lower a ``scf.index_switch`` op by recursing into its regions."""
    for region in op.regions:
        for b in region.blocks:
            _update_non_qst_block_arg_types(b)

    for region in op.regions:
        for b in region.blocks:
            _process_block(b, execution_mode)

    # Strip qst from yields and results
    for region in op.regions:
        for b in region.blocks:
            for inner_op in b.ops:
                if inner_op.name == "scf.yield":
                    non_qst = [v for v in inner_op.operands if not _is_qst(v.type)]
                    inner_op.operands = non_qst

    has_qst = any(_is_qst(t) for t in op.result_types)
    if not has_qst:
        return

    # For index_switch, we need to create a new op without qst results.
    # Since this is complex and rarely used in quantum kernels, we leave the
    # qst result as erased (safe_erase=False) and let downstream passes clean up.
    for res in op.results:
        if _is_qst(res.type):
            res.erase(safe_erase=False)
