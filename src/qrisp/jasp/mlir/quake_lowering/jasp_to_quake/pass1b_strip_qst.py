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
PASS 1b – QuantumState structural elimination.

After PASS 1a has lowered all ``jasp.*`` ops and threaded QST backwards
(making all QST values dead), excecpt for the quantum kernel creation/consumption ops, this pass
removes all remaining structural traces of ``!jasp.QuantumState`` from the IR.

Quantum kernels are handled specially: the ``jasp.create_quantum_kernel`` op is dropped, and the
``jasp.consume_quantum_kernel`` op is replaced with a constant True tensor. 
A ``jasp.create_quantum_kernel`` op has a single result of type ``!jasp.QuantumState``. 
A ``jasp.consume_quantum_kernel`` op has a single operand of type ``!jasp.QuantumState``and a single result of type ``tensor<i1>``.
Both ops enclose a ``func.call`` marking the quantum kernel function:

    %0 = jasp.create_quantum_kernel -> !jasp.QuantumState
    %1, %2 = func.call @quantum_kernel(%0) : (!jasp.QuantumState) -> (tensor<i1>, !jasp.QuantumState)
    %3 = jasp.consume_quantum_kernel %2 : !jasp.QuantumState -> tensor<i1>

This pass removes the remaining structural traces of ``!jasp.QuantumState`` from:

- ``scf.while`` init operands, block arguments, yields, conditions, and results
- ``scf.if`` yields and results
- ``scf.for`` init args, block arguments, yields, and results
- ``scf.index_switch`` yields and results
- ``func.call`` operands and results
- ``func.return`` operands
- ``func.func`` argument lists, return types, and attributes

"""

from xdsl.dialects import arith, func
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    FunctionType,
    ModuleOp,
    StringAttr,
)
from xdsl.dialects.scf import ConditionOp, ForOp, IfOp, IndexSwitchOp, WhileOp, YieldOp
from xdsl.ir import (
    Attribute,
    Block,
    Operation,
    SSAValue,
)
from xdsl.pattern_rewriter import (
    InsertPoint,
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import Rewriter

from qrisp.jasp.mlir.quake_lowering.jasp_to_quake.quake_dialect import (
    QuakeRefType,
    QuakeVeqType,
)
from qrisp.jasp.mlir.quake_lowering.jasp_to_quake.helper_functions import (
    _is_qst,
    _is_qubit_array,
    _is_qubit,
    _quake_type_for,
)
from qrisp.jasp.mlir.xdsl_dialect import (
    ConsumeQuantumKernelOp,
    CreateQuantumKernelOp,
)


# ===========================================================================
# Public entry point
# ===========================================================================


def strip_qst(module: ModuleOp, execution_mode: str = "run") -> None:
    """In-place PASS 1b: remove all structural traces of ``!jasp.QuantumState``.

    Parameters
    ----------
    module:
        An xDSL ``builtin.ModuleOp`` that has already been processed by PASS 1a.
        Modified in-place.
    execution_mode:
        ``"run"`` or ``"sample"`` — affects function return type handling.
    """
    patterns: list[RewritePattern] = [
        LowerCreateQuantumKernel(),
        LowerConsumeQuantumKernel(),
        StripQSTFromWhile(),
        StripQSTFromIf(),
        StripQSTFromFor(),
        StripQSTFromIndexSwitch(),
        StripQSTFromReturn(execution_mode),
        StripQSTFromCall(),
        StripQSTFromFunc(execution_mode),
    ]

    applier = GreedyRewritePatternApplier(patterns)
    walker = PatternRewriteWalker(applier, apply_recursively=True)
    walker.rewrite_module(module)


# ===========================================================================
# Shared helpers
# ===========================================================================


def _op_has_qst(op: Operation) -> bool:
    """Return True if *op* has QST in operands or results."""
    return any(_is_qst(v.type) for v in op.operands) or any(_is_qst(t) for t in op.result_types)


def _update_block_arg_types(block: Block) -> None:
    """Erase QST block args; update QubitArray/Qubit args to Quake types."""
    for arg in list(block.args):
        if _is_qst(arg.type):
            block.erase_arg(arg, safe_erase=False)
        elif _is_qubit_array(arg.type):
            Rewriter.replace_value_with_new_type(arg, QuakeVeqType())
        elif _is_qubit(arg.type):
            Rewriter.replace_value_with_new_type(arg, QuakeRefType())


def _strip_qst_from_terminators(op: Operation) -> None:
    """Remove QST from yield/condition operands in all regions of *op*."""
    for region in op.regions:
        for block in region.blocks:
            for inner_op in block.ops:
                if isinstance(inner_op, (YieldOp, ConditionOp)):
                    inner_op.operands = [v for v in inner_op.operands if not _is_qst(v.type)]


def _non_qst_result_types(op: Operation) -> list[Attribute]:
    """Compute new result types: drop QST, convert Jasp qubit types to Quake."""
    return [_quake_type_for(t) or t for t in op.result_types if not _is_qst(t)]


def _build_result_mapping(old_results, new_results_iter) -> list[SSAValue | None]:
    """Map old results to new results, returning None for QST positions."""
    mapping: list[SSAValue | None] = []
    new_idx = 0
    for old_res in old_results:
        if _is_qst(old_res.type):
            mapping.append(None)
        else:
            mapping.append(new_results_iter[new_idx])
            new_idx += 1
    return mapping


# ===========================================================================
# Quantum Kernel Patterns
# ===========================================================================


class LowerCreateQuantumKernel(RewritePattern):
    """``jasp.create_quantum_kernel`` → dropped (after QST uses are gone)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CreateQuantumKernelOp, rewriter: PatternRewriter) -> None:
        qst_outputs = [r for r in op.results if _is_qst(r.type)]
        if any(r.uses for r in qst_outputs):
            return  # Not ready yet — wait for StripQSTFromCall to fire
        rewriter.erase_op(op)


class LowerConsumeQuantumKernel(RewritePattern):
    """``jasp.consume_quantum_kernel`` → constant True tensor."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ConsumeQuantumKernelOp, rewriter: PatternRewriter) -> None:
        if op.results:
            true_const = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(op.results[0].type, [1]))
            op.results[0].replace_all_uses_with(true_const.result)
            rewriter.insert_op(true_const, InsertPoint.before(rewriter.current_operation))
        rewriter.erase_op(op)


# ===========================================================================
# SCF Patterns
# ===========================================================================


class StripQSTFromWhile(RewritePattern):
    """Strip ``!jasp.QuantumState`` from ``scf.while``."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: WhileOp, rewriter: PatternRewriter) -> None:
        if not _op_has_qst(op):
            return

        # Update block args and terminators
        for region in op.regions:
            for block in region.blocks:
                _update_block_arg_types(block)
        _strip_qst_from_terminators(op)

        # Rebuild
        new_res_types = _non_qst_result_types(op)
        before_region = op.detach_region(op.regions[0])
        after_region = op.detach_region(op.regions[0])

        non_qst_inits = [v for v in op.arguments if not _is_qst(v.type)]
        new_while = WhileOp(non_qst_inits, new_res_types, before_region, after_region)

        new_results = _build_result_mapping(op.results, new_while.results)
        rewriter.replace_matched_op(new_while, new_results, safe_erase=False)


class StripQSTFromIf(RewritePattern):
    """Strip ``!jasp.QuantumState`` from ``scf.if``."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: IfOp, rewriter: PatternRewriter) -> None:
        if not _op_has_qst(op):
            return

        for region in op.regions:
            for block in region.blocks:
                _update_block_arg_types(block)
        _strip_qst_from_terminators(op)

        new_res_types = _non_qst_result_types(op)
        then_region = op.detach_region(op.regions[0])
        else_region = op.detach_region(op.regions[0])

        new_if = IfOp(op.operands[0], new_res_types, then_region, else_region)

        new_results = _build_result_mapping(op.results, new_if.results)
        rewriter.replace_matched_op(new_if, new_results, safe_erase=False)


class StripQSTFromFor(RewritePattern):
    """Strip ``!jasp.QuantumState`` from ``scf.for``."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ForOp, rewriter: PatternRewriter) -> None:
        if not _op_has_qst(op):
            return

        for region in op.regions:
            for block in region.blocks:
                _update_block_arg_types(block)
        _strip_qst_from_terminators(op)

        new_res_types = _non_qst_result_types(op)
        body_region = op.detach_region(op.regions[0])

        # scf.for operands: [lb, ub, step, init_args...]
        non_qst_init_args = [v for v in list(op.operands)[3:] if not _is_qst(v.type)]
        new_for = ForOp(
            op.operands[0],  # lb
            op.operands[1],  # ub
            op.operands[2],  # step
            non_qst_init_args,
            body_region,
        )

        new_results = _build_result_mapping(op.results, new_for.results)
        rewriter.replace_matched_op(new_for, new_results, safe_erase=False)


class StripQSTFromIndexSwitch(RewritePattern):
    """Strip ``!jasp.QuantumState`` from ``scf.index_switch``."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: IndexSwitchOp, rewriter: PatternRewriter) -> None:
        if not _op_has_qst(op):
            return

        for region in op.regions:
            for block in region.blocks:
                _update_block_arg_types(block)
        _strip_qst_from_terminators(op)

        # Erase QST results directly
        for res in op.results:
            if _is_qst(res.type):
                res.erase(safe_erase=False)


# ===========================================================================
# Function-level Patterns
# ===========================================================================


class StripQSTFromReturn(RewritePattern):
    """Remove QST from ``func.return`` operands."""

    def __init__(self, execution_mode: str = "run"):
        super().__init__()
        self.execution_mode = execution_mode

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.ReturnOp, rewriter: PatternRewriter) -> None:
        if not any(_is_qst(v.type) for v in op.operands):
            return

        if self.execution_mode == "sample":
            parent_func = op.parent_block().parent_region().parent_op()
            if isinstance(parent_func, func.FuncOp) and parent_func.sym_name.data == "main":
                op.operands = []
                return

        op.operands = [v for v in op.operands if not _is_qst(v.type)]


class StripQSTFromCall(RewritePattern):
    """Remove QST from ``func.call`` operands and results."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter) -> None:
        if not _op_has_qst(op):
            return

        new_operands = [v for v in op.operands if not _is_qst(v.type)]

        new_result_types: list[Attribute] = []
        for t in op.result_types:
            if _is_qst(t):
                continue
            elif _is_qubit_array(t):
                new_result_types.append(QuakeVeqType())
            elif _is_qubit(t):
                new_result_types.append(QuakeRefType())
            else:
                new_result_types.append(t)

        new_call = func.CallOp(op.callee, new_operands, new_result_types)

        new_results = _build_result_mapping(op.results, new_call.results)
        rewriter.replace_matched_op(new_call, new_results, safe_erase=False)


class StripQSTFromFunc(RewritePattern):
    """Fix ``func.func`` signature: remove QST args/returns, add cudaq attrs."""

    def __init__(self, execution_mode: str = "run"):
        super().__init__()
        self.execution_mode = execution_mode

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter) -> None:
        old_ftype: FunctionType = op.function_type

        # Check if there's anything to do
        has_qst_in_sig = any(_is_qst(t) for t in old_ftype.inputs.data) or any(
            _is_qst(t) for t in old_ftype.outputs.data
        )
        needs_attrs = "cudaq.kernel" not in op.attributes

        if not has_qst_in_sig and not needs_attrs:
            return

        # Update entry block args
        new_inputs: list[Attribute] = []
        entry_block = op.body.blocks.first
        if entry_block is not None:
            for arg in list(entry_block.args):
                if _is_qst(arg.type):
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

        # Compute new return types
        # For sample execution_mode the kernel must return void – cudaq.sample collects
        # measurement results via the runtime, not as return values.
        if self.execution_mode == "sample" and op.sym_name.data == "main":
            new_outputs: list[Attribute] = []
        else:
            new_outputs = [_quake_type_for(t) or t for t in old_ftype.outputs.data if not _is_qst(t)]

        op.function_type = FunctionType.from_lists(new_inputs, new_outputs)
        op.attributes["cudaq.kernel"] = StringAttr("true")
        if op.sym_name.data == "main":
            op.attributes["cudaq.entrypoint"] = StringAttr("true")
