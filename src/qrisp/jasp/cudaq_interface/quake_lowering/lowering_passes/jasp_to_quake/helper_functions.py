# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Provide helper functions for Jasp-to-Quake lowering."""

from xdsl.dialects import arith, func, tensor
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    IntegerAttr,
    IntegerType,
    TensorType,
    f64,
)
from xdsl.ir import (
    Attribute,
    Operation,
    SSAValue,
)
from xdsl.pattern_rewriter import (
    PatternRewriter,
)
from xdsl.rewriter import InsertPoint
from xdsl.traits import Pure

from qrisp.jasp.cudaq_interface.quake_lowering.dialects.quake_dialect import (
    QuakeRefType,
    QuakeVeqType,
    VeqSizeOp,
    _make_gate_op,
)
from qrisp.jasp.cudaq_interface.quake_lowering.lowering_passes.ir_helpers import _is_float_type, _is_scalar_tensor
from qrisp.jasp.cudaq_interface.quake_lowering.lowering_passes.jasp_to_quake.gate_mapping import GateInfo
from qrisp.jasp.mlir.xdsl_dialect import (
    QuantumGateOp,
    QuantumStateType,
    QubitArrayType,
    QubitType,
)

# ---------------------------------------------------------------------------
# Helpers to identify Jasp types
# ---------------------------------------------------------------------------


def _is_qst(t: Attribute) -> bool:
    """Return True if *t* is ``!jasp.QuantumState``."""
    return isinstance(t, QuantumStateType)


def _is_qubit_array(t: Attribute) -> bool:
    """Return True if *t* is ``!jasp.QubitArray``."""
    return isinstance(t, QubitArrayType)


def _is_qubit(t: Attribute) -> bool:
    """Return True if *t* is ``!jasp.Qubit``."""
    return isinstance(t, QubitType)


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
    return isinstance(t, (QuakeRefType, QuakeVeqType)) or _is_qubit(t) or _is_qubit_array(t)


def _is_numeric_type(t: Attribute) -> bool:
    """Return True if *t* is a scalar or rank-0 tensor of int/float."""
    scalar = t
    if isinstance(t, TensorType):
        if t.get_shape():  # only rank-0
            return False
        scalar = t.element_type
    return isinstance(scalar, IntegerType) or _is_float_type(scalar)


def _scalar_type_of(t: Attribute) -> Attribute:
    """Return the scalar element type (unwrap rank-0 tensor if needed)."""
    if _is_scalar_tensor(t):
        return t.element_type
    return t


# ---------------------------------------------------------------------------
# Numeric-type helpers for rewriter
# ---------------------------------------------------------------------------


def _coerce_to_f64_for_rewriter(val: SSAValue, rewriter: PatternRewriter) -> SSAValue:
    """Extract from tensor if needed, then cast to f64."""
    scalar_type = _scalar_type_of(val.type)
    scalar = _extract_scalar_for_rewriter(val, scalar_type, rewriter)

    if scalar.type == f64:
        return scalar

    if _is_float_type(scalar.type):
        cast = arith.ExtFOp(scalar, f64)
        rewriter.insert_op(cast, InsertPoint.before(rewriter.current_operation))
        return cast.result

    if isinstance(scalar.type, IntegerType):
        if scalar.type == IntegerType(1):
            cast = arith.UIToFPOp(scalar, f64)
        else:
            cast = arith.SIToFPOp(scalar, f64)
        rewriter.insert_op(cast, InsertPoint.before(rewriter.current_operation))
        return cast.result

    raise ValueError(f"Cannot coerce {scalar.type} to f64")


def _try_get_constant_int(value: SSAValue) -> int | None:
    """Return the compile-time integer value feeding *value*, if any.

    Sees through a single ``tensor.extract`` (the standard rank-0
    tensor-scalar round-trip emitted for jasp indices) down to the
    underlying ``arith.constant`` (scalar or dense-tensor), else None.
    """
    owner = value.owner
    if isinstance(owner, tensor.ExtractOp):
        owner = owner.tensor.owner

    if not isinstance(owner, arith.ConstantOp):
        return None

    attr = owner.value
    if isinstance(attr, IntegerAttr):
        return attr.value.data
    if isinstance(attr, DenseIntOrFPElementsAttr):
        values = list(attr.get_values())
        if len(values) == 1:
            return values[0]
    return None


def _normalize_index_for_veq_rewriter(
    veq: SSAValue,
    idx: SSAValue,
    rewriter: PatternRewriter,
    size: SSAValue | None = None,
) -> SSAValue:
    """Python-style negative indexing: if idx < 0, return idx + size(veq).

    When ``idx`` is already a compile-time constant, its sign is known
    statically, so the cmpi/select scaffold is skipped: a non-negative
    constant is used as-is, and a negative one only needs the addi (no
    cmpi/select needed since the branch is resolved).

    ``size`` is an already-emitted ``quake.veq_size`` result for *veq*. A
    caller that normalizes several indices against the same register passes
    it so that only one ``quake.veq_size`` is emitted.
    """
    const_idx = _try_get_constant_int(idx)
    if const_idx is not None and const_idx >= 0:
        return idx

    size_ops = []
    if size is None:
        size_op = VeqSizeOp(veq)
        size_ops.append(size_op)
        size = size_op.result

    idx_plus_size = arith.AddiOp(idx, size)
    if const_idx is not None:
        rewriter.insert_op([*size_ops, idx_plus_size], InsertPoint.before(rewriter.current_operation))
        return idx_plus_size.result

    zero = arith.ConstantOp(IntegerAttr(0, 64))
    is_neg = arith.CmpiOp(idx, zero.result, "slt")
    norm = arith.SelectOp(is_neg.result, idx_plus_size.result, idx)

    rewriter.insert_op(
        [*size_ops, zero, is_neg, idx_plus_size, norm],
        InsertPoint.before(rewriter.current_operation),
    )
    return norm.result


def _normalize_slice_bounds_for_veq_rewriter(
    veq: SSAValue, start: SSAValue, stop: SSAValue, rewriter: PatternRewriter
) -> tuple[SSAValue, SSAValue]:
    """Python slice-bound normalization for ``veq[start:stop]``.

    Both bounds get negative-index folding against the register size, the
    start is then clamped up to 0 and the (still exclusive) stop clamped down
    to the size. A caller that treats ``hi <= lo`` as an empty slice therefore
    only ever reaches ``quake.subveq`` with ``0 <= lo < hi <= size``, making
    the inclusive upper bound ``hi - 1`` a valid qubit index.

    Out-of-range in the other direction needs no clamp: a start past the end
    and a stop still negative after folding both produce ``hi <= lo``, which
    is the empty case.
    """
    size = VeqSizeOp(veq)
    rewriter.insert_op(size, InsertPoint.before(rewriter.current_operation))

    lo = _normalize_index_for_veq_rewriter(veq, start, rewriter, size=size.result)
    hi = _normalize_index_for_veq_rewriter(veq, stop, rewriter, size=size.result)

    start_const = _try_get_constant_int(start)
    if start_const is None or start_const < 0:
        # A start that was non-negative to begin with needs no lower clamp.
        zero = arith.ConstantOp(IntegerAttr(0, 64))
        lo_clamped = arith.MaxSIOp(lo, zero.result)
        rewriter.insert_op([zero, lo_clamped], InsertPoint.before(rewriter.current_operation))
        lo = lo_clamped.result

    stop_const = _try_get_constant_int(stop)
    if stop_const is None or stop_const >= 0:
        # A negative stop folds to ``stop + size``, which is below the size.
        hi_clamped = arith.MinSIOp(hi, size.result)
        rewriter.insert_op(hi_clamped, InsertPoint.before(rewriter.current_operation))
        hi = hi_clamped.result

    return lo, hi


# ---------------------------------------------------------------------------
# Tensor-extract helpers for rewriter
# ---------------------------------------------------------------------------


def _extract_scalar_for_rewriter(val: SSAValue, scalar_type: Attribute, rewriter: PatternRewriter) -> SSAValue:
    """Insert ``tensor.extract`` if val is a tensor; return scalar SSAValue."""
    if val.type == scalar_type:
        return val
    extract = tensor.ExtractOp(val, [], scalar_type)
    rewriter.insert_op(extract, InsertPoint.before(rewriter.current_operation))
    return extract.result


def _wrap_scalar_for_rewriter(val: SSAValue, tensor_type: Attribute, rewriter: PatternRewriter) -> SSAValue:
    """Insert ``tensor.from_elements`` to wrap scalar into tensor."""
    if val.type == tensor_type:
        return val
    from_elem = tensor.FromElementsOp(operands=[[val]], result_types=[tensor_type])
    rewriter.insert_op(from_elem, InsertPoint.before(rewriter.current_operation))
    return from_elem.result


# ---------------------------------------------------------------------------
# Quantum-gate helpers
# ---------------------------------------------------------------------------


def _classify_gate_operands(op: QuantumGateOp, rewriter: PatternRewriter) -> tuple[list[SSAValue], list[SSAValue]]:
    """Separate quantum-gate operands into qubits and converted parameters."""
    qubit_operands: list[SSAValue] = []
    param_operands: list[SSAValue] = []
    for operand in op.operands:
        if _is_qst(operand.type):
            continue
        if _is_qubit_type(operand.type):
            qubit_operands.append(operand)
        elif _is_numeric_type(operand.type):
            param_operands.append(_coerce_to_f64_for_rewriter(operand, rewriter))
        else:
            qubit_operands.append(operand)
    return qubit_operands, param_operands


def _split_gate_operands(qubit_operands: list[SSAValue], gate_info: GateInfo) -> tuple[list[SSAValue], list[SSAValue]]:
    """Split gate qubits into controls and targets according to gate metadata."""
    if gate_info.num_controls == -1:
        return qubit_operands[:-1], qubit_operands[-1:]
    if gate_info.num_controls == 0:
        return [], qubit_operands
    return qubit_operands[: gate_info.num_controls], qubit_operands[gate_info.num_controls :]


def _emit_gate(
    gate_name: str,
    gate_info: GateInfo,
    gate_operands: tuple[list[SSAValue], list[SSAValue], list[SSAValue]],
    rewriter: PatternRewriter,
) -> None:
    """Create and insert either a custom gate decomposition or a Quake gate."""
    controls, param_operands, targets = gate_operands
    final_params = param_operands[: gate_info.num_params]
    if gate_info.emit is not None:
        emitted_ops = gate_info.emit(controls, final_params, targets)
        if not emitted_ops:
            raise RuntimeError(f"Gate '{gate_name}' emit() returned empty list.")
        rewriter.insert_op(emitted_ops, InsertPoint.before(rewriter.current_operation))
        return

    gate_op = _make_gate_op(gate_name, controls, final_params, targets)
    if gate_op is None:
        raise RuntimeError(f"Gate '{gate_name}' not in Quake gate class table.")
    rewriter.insert_op(gate_op, InsertPoint.before(rewriter.current_operation))


# ---------------------------------------------------------------------------
# Sample-mode measurement helpers
# ---------------------------------------------------------------------------


def _is_entry_return(op: Operation) -> bool:
    """Return whether *op* is the ``func.return`` of the entry function."""
    if not isinstance(op, func.ReturnOp):
        return False
    parent = op.parent_op()
    return isinstance(parent, func.FuncOp) and parent.sym_name.data == "main"


def _first_impure_consumer(value: SSAValue) -> Operation | None:
    """Return the first side-effecting consumer of *value*, following pure ops.

    The entry function's ``func.return`` is not a consumer here: sample mode
    strips it together with every classical value flowing into it, so pure
    computation that ends there dies with it.
    """
    seen: set[Operation] = set()
    worklist = [value]
    while worklist:
        for use in worklist.pop().uses:
            user = use.operation
            if user in seen or _is_entry_return(user):
                continue
            seen.add(user)
            if not user.has_trait(Pure):
                return user
            worklist.extend(user.results)
    return None
