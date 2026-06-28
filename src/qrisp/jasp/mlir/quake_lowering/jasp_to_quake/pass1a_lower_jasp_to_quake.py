"""********************************************************************************
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
PASS 1a – Jasp→Quake op lowering.

Replaces all ``jasp.*`` ops with their Quake equivalents using a
``PatternRewriteWalker``.  QuantumState values are threaded backwards
(``qst_out.replace_all_uses_with(qst_in)``) inside each pattern, making
all QST values dead.  The actual *structural* removal of QST from SCF ops
and function signatures is handled by the separate PASS 1b.

Op mapping:

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
``jasp.create_quantum_kernel``        dropped in PASS 1b
``jasp.consume_quantum_kernel``       dropped in PASS 1b
========================  ==========================================
"""

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    IntegerAttr,
    ModuleOp,
    i64,
)
from xdsl.ir import (
    Block,
    Operation,
    Region,
    SSAValue,
)
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from qrisp.jasp.mlir.quake_lowering.jasp_to_quake.quake_dialect import (
    AllocaVeqOp,
    ConcatOp,
    DeallocOp,
    DiscriminateOp,
    ExtractRefOp,
    MzOp,
    QuakeVeqType,
    ResetOp,
    SubVeqOp,
    VeqSizeOp,
    make_gate_op,
)
from qrisp.jasp.mlir.quake_lowering.jasp_to_quake.gate_mapping import get_gate_info
from qrisp.jasp.mlir.xdsl_dialect import (
    CreateQubitsOp,
    DeleteQubitsOp,
    FuseOp,
    GetQubitOp,
    GetSizeOp,
    MeasureOp as JaspMeasureOp,
    QuantumGateOp,
    ResetOp as JaspResetOp,
    SliceOp,
)
from qrisp.jasp.mlir.quake_lowering.jasp_to_quake.helper_functions import (
    _is_qst,
    _is_qubit_array,
    _is_qubit,
    _is_qubit_type,
    _is_numeric_type,
    _extract_scalar_for_rewriter,
    _coerce_to_f64_for_rewriter,
    _normalize_index_for_veq_rewriter,
    _wrap_scalar_for_rewriter,
)


# ===========================================================================
# Public entry point
# ===========================================================================


def lower_jasp_to_quake(module: ModuleOp, execution_mode: str = "run") -> None:
    """In-place PASS 1: lower all ``jasp.*`` ops to Quake equivalents.

    After this pass, no ``jasp.*`` ops remain.  ``!jasp.QuantumState`` values
    are dead (all uses redirected) but may still appear structurally in SCF
    ops and function signatures — those are cleaned up by PASS 2.

    Parameters
    ----------
    module:
        An xDSL ``builtin.ModuleOp`` containing the JASP MLIR IR.
        Modified in-place.
    execution_mode:
        ``"run"`` (default) packs array measurement results into an ``i64``.
        ``"sample"`` emits raw ``quake.mz`` and uses zero placeholders.
    """

    patterns: list[RewritePattern] = [
        LowerCreateQubits(),
        LowerGetQubit(),
        LowerGetSize(),
        LowerSlice(),
        LowerFuse(),
        LowerDeleteQubits(),
        LowerQuantumGate(),
        LowerMeasure(execution_mode),
        LowerReset(),
    ]

    applier = GreedyRewritePatternApplier(patterns)
    walker = PatternRewriteWalker(applier, apply_recursively=True)
    walker.rewrite_module(module)


# ===========================================================================
# QST threading helper
# ===========================================================================


def _thread_qst(op: Operation) -> None:
    """Thread QST backwards: replace each qst_out with its corresponding qst_in.

    This makes QST results dead so the op can be safely erased.
    For ops without a matching QST input (e.g. ``create_quantum_kernel``),
    the QST output is left with no replacement (erased via safe_erase=False).
    """
    qst_inputs = [v for v in op.operands if _is_qst(v.type)]
    qst_outputs = [r for r in op.results if _is_qst(r.type)]

    for i, qst_out in enumerate(qst_outputs):
        if i < len(qst_inputs):
            qst_out.replace_all_uses_with(qst_inputs[i])


# ===========================================================================
# Rewrite Patterns
# ===========================================================================


class LowerCreateQubits(RewritePattern):
    """``jasp.create_qubits %n_tensor, %qst`` → ``quake.alloca !quake.veq<?>[%n]``."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CreateQubitsOp, rewriter: PatternRewriter) -> None:
        amount_tensor = op.operands[0]
        n = _extract_scalar_for_rewriter(amount_tensor, i64, rewriter)

        alloca = AllocaVeqOp(n)
        rewriter.insert_op(alloca, InsertPoint.before(rewriter.current_operation))

        for r in op.results:
            if _is_qubit_array(r.type):
                r.replace_all_uses_with(alloca.result)

        _thread_qst(op)
        rewriter.erase_op(op)


class LowerGetQubit(RewritePattern):
    """``jasp.get_qubit %arr, %idx_tensor`` → ``quake.extract_ref %veq[%idx]``."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GetQubitOp, rewriter: PatternRewriter) -> None:
        arr = op.operands[0]
        idx_tensor = op.operands[1]

        idx = _extract_scalar_for_rewriter(idx_tensor, i64, rewriter)
        norm_idx = _normalize_index_for_veq_rewriter(arr, idx, rewriter)

        extract = ExtractRefOp(arr, norm_idx)
        rewriter.insert_op(extract, InsertPoint.before(rewriter.current_operation))

        for r in op.results:
            if _is_qubit(r.type):
                r.replace_all_uses_with(extract.result)

        rewriter.erase_op(op)


class LowerGetSize(RewritePattern):
    """``jasp.get_size %arr`` → ``quake.veq_size %veq`` wrapped to tensor<i64>."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GetSizeOp, rewriter: PatternRewriter) -> None:
        arr = op.operands[0]
        veq_size = VeqSizeOp(arr)
        rewriter.insert_op(veq_size, InsertPoint.before(rewriter.current_operation))

        result_tensor_type = op.results[0].type
        wrapped = _wrap_scalar_for_rewriter(veq_size.result, result_tensor_type, rewriter)

        op.results[0].replace_all_uses_with(wrapped)
        rewriter.erase_op(op)


class LowerSlice(RewritePattern):
    """``jasp.slice %arr, %start, %end`` → ``quake.subveq %veq, %lo, %hi``.

    jasp.slice uses exclusive upper bound (Python-style: [start, end)),
    while quake.subveq uses inclusive bounds [lo, hi].

    Assumptions:
    - start index is already a non-negative absolute index.
    - end index may be negative (Python slice semantics).
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: SliceOp, rewriter: PatternRewriter) -> None:
        arr = op.operands[0]
        start_t = op.operands[1]
        end_t = op.operands[2]

        lo = _extract_scalar_for_rewriter(start_t, i64, rewriter)
        hi_raw = _extract_scalar_for_rewriter(end_t, i64, rewriter)
        hi_norm = _normalize_index_for_veq_rewriter(arr, hi_raw, rewriter)

        # Exclusive → inclusive: hi_inclusive = hi_norm - 1
        one = arith.ConstantOp(IntegerAttr(1, 64))
        hi_inclusive = arith.SubiOp(hi_norm, one.result)
        rewriter.insert_op([one, hi_inclusive], InsertPoint.before(rewriter.current_operation))

        subveq = SubVeqOp(arr, lo, hi_inclusive.result)
        rewriter.insert_op(subveq, InsertPoint.before(rewriter.current_operation))

        for r in op.results:
            if _is_qubit_array(r.type):
                r.replace_all_uses_with(subveq.result)

        rewriter.erase_op(op)


class LowerFuse(RewritePattern):
    """``jasp.fuse %a, %b`` → ``quake.concat %a, %b``."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuseOp, rewriter: PatternRewriter) -> None:
        operands = [v for v in op.operands if not _is_qst(v.type)]
        concat = ConcatOp(operands)
        rewriter.insert_op(concat, InsertPoint.before(rewriter.current_operation))

        for r in op.results:
            if _is_qubit_array(r.type):
                r.replace_all_uses_with(concat.result)

        rewriter.erase_op(op)


class LowerDeleteQubits(RewritePattern):
    """``jasp.delete_qubits %arr, %qst`` → ``quake.dealloc %veq``."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DeleteQubitsOp, rewriter: PatternRewriter) -> None:
        arr = op.operands[0]
        dealloc = DeallocOp(arr)
        rewriter.insert_op(dealloc, InsertPoint.before(rewriter.current_operation))

        _thread_qst(op)
        rewriter.erase_op(op)


class LowerQuantumGate(RewritePattern):
    """``jasp.quantum_gate "name" (%qubits, %params), %qst`` → Quake gate op(s)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: QuantumGateOp, rewriter: PatternRewriter) -> None:
        qubit_operands: list[SSAValue] = []
        param_operands: list[SSAValue] = []
        for v in op.operands:
            if _is_qst(v.type):
                continue
            elif _is_qubit_type(v.type):
                qubit_operands.append(v)
            elif _is_numeric_type(v.type):
                param_operands.append(_coerce_to_f64_for_rewriter(v, rewriter))
            else:
                qubit_operands.append(v)

        gate_name = op.gate_type.data
        gate_info = get_gate_info(gate_name)
        if gate_info is None:
            raise NotImplementedError(f"Lowering failed: Unsupported Jasp gate '{gate_name}'.")

        num_ctrl = gate_info.num_controls
        if num_ctrl == -1:
            controls = qubit_operands[:-1]
            targets = qubit_operands[-1:]
        elif num_ctrl == 0:
            controls = []
            targets = qubit_operands
        else:
            controls = qubit_operands[:num_ctrl]
            targets = qubit_operands[num_ctrl:]

        final_params = list(param_operands[: gate_info.num_params])

        if gate_info.emit is not None:
            ops = gate_info.emit(controls, final_params, targets)
            if not ops:
                raise RuntimeError(f"Gate '{gate_name}' emit() returned empty list.")
            rewriter.insert_op(ops, InsertPoint.before(rewriter.current_operation))
        else:
            gate_op = make_gate_op(gate_name, controls, final_params, targets)
            if gate_op is None:
                raise RuntimeError(f"Gate '{gate_name}' not in Quake gate class table.")
            rewriter.insert_op(gate_op, InsertPoint.before(rewriter.current_operation))

        _thread_qst(op)
        rewriter.erase_op(op)


class LowerMeasure(RewritePattern):
    """``jasp.measure`` → Quake measurement ops.

    The output depends on whether the operand is a single qubit or a qubit
    array, and on *execution_mode*:

    Single qubit, ``"run"`` mode
        ``quake.mz %ref → !quake.measure`` followed by
        ``quake.discriminate → i1``, wrapped back to ``tensor<i1>``.

    Single qubit, ``"sample"`` mode
        ``quake.mz %ref → !quake.measure`` only.  The classical result is
        replaced by a zero ``tensor<i1>`` placeholder so that downstream SSA
        uses remain valid; the placeholder is stripped from the function
        return by :func:`_fix_return_op`.

    Qubit array, ``"run"`` mode
        A ``scf.for`` loop extracts each qubit, applies ``quake.mz`` +
        ``quake.discriminate``, and bit-packs the results into a single
        ``i64`` (little-endian), which is wrapped to ``tensor<i64>``.

    Qubit array, ``"sample"`` mode
        ``quake.mz %veq → !cc.stdvec<!quake.measure>`` only.  The classical
        result is replaced by a zero ``tensor<i64>`` placeholder; stripped
        from the return by :func:`_fix_return_op`.
    """

    def __init__(self, execution_mode: str = "run"):
        super().__init__()
        self.execution_mode = execution_mode

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: JaspMeasureOp, rewriter: PatternRewriter) -> None:
        qubit_val = op.operands[0]
        meas_result = op.results[0]
        is_array = _is_qubit_array(qubit_val.type) or isinstance(qubit_val.type, QuakeVeqType)

        if not is_array and self.execution_mode != "sample":
            self._lower_single_qubit_run(qubit_val, meas_result, rewriter)
        elif not is_array and self.execution_mode == "sample":
            self._lower_single_qubit_sample(qubit_val, meas_result, rewriter)
        elif self.execution_mode == "sample":
            self._lower_array_sample(qubit_val, meas_result, rewriter)
        else:
            self._lower_array_run(qubit_val, meas_result, rewriter)

        _thread_qst(op)
        rewriter.erase_op(op)

    def _lower_single_qubit_run(self, qubit_val, meas_result, rewriter):
        mz = MzOp(qubit_val)
        disc = DiscriminateOp(mz.result)
        rewriter.insert_op([mz, disc], InsertPoint.before(rewriter.current_operation))
        wrapped = _wrap_scalar_for_rewriter(disc.result, meas_result.type, rewriter)
        meas_result.replace_all_uses_with(wrapped)

    def _lower_single_qubit_sample(self, qubit_val, meas_result, rewriter):
        mz = MzOp(qubit_val)
        zero_const = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(meas_result.type, [0]))
        rewriter.insert_op([mz, zero_const], InsertPoint.before(rewriter.current_operation))
        meas_result.replace_all_uses_with(zero_const.result)

    def _lower_array_sample(self, qubit_val, meas_result, rewriter):
        mz = MzOp(qubit_val)
        zero_const = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(meas_result.type, [0]))
        rewriter.insert_op([mz, zero_const], InsertPoint.before(rewriter.current_operation))
        meas_result.replace_all_uses_with(zero_const.result)

    def _lower_array_run(self, qubit_val, meas_result, rewriter):
        veq_size = VeqSizeOp(qubit_val)
        c0 = arith.ConstantOp(IntegerAttr(0, 64))
        c1 = arith.ConstantOp(IntegerAttr(1, 64))
        rewriter.insert_op([veq_size, c0, c1], InsertPoint.before(rewriter.current_operation))

        loop_body = Block(arg_types=[i64, i64])
        iv, acc = loop_body.args

        extract = ExtractRefOp(qubit_val, iv)
        mz_single = MzOp(extract.result)
        disc = DiscriminateOp(mz_single.result)
        extui = arith.ExtUIOp(disc.result, i64)
        shift = arith.ShLIOp(extui.result, iv)
        new_acc = arith.OrIOp(acc, shift.result)
        yield_op = scf.YieldOp(new_acc.result)

        loop_body.add_ops([extract, mz_single, disc, extui, shift, new_acc, yield_op])

        for_op = scf.ForOp(c0.result, veq_size.result, c1.result, [c0.result], Region([loop_body]))
        rewriter.insert_op(for_op, InsertPoint.before(rewriter.current_operation))

        wrapped = _wrap_scalar_for_rewriter(for_op.results[0], meas_result.type, rewriter)
        meas_result.replace_all_uses_with(wrapped)


class LowerReset(RewritePattern):
    """``jasp.reset %q, %qst`` → ``quake.reset %q``."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: JaspResetOp, rewriter: PatternRewriter) -> None:
        qubit_val = op.operands[0]
        reset_op = ResetOp(qubit_val)
        rewriter.insert_op(reset_op, InsertPoint.before(rewriter.current_operation))

        _thread_qst(op)
        rewriter.erase_op(op)
