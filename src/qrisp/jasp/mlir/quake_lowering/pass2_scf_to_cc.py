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
PASS 2 – SCF → CC dialect lowering (pure-SSA approach).

Replaces structured control-flow ops (``scf.if``, ``scf.for``, ``scf.while``)
with their CC-dialect equivalents (``cc.if``, ``cc.loop``) using pure SSA
value threading via block arguments.

Strategy
--------
- ``cc.if`` carries result values via ``cc.continue %val : type`` in each
  branch (matching native CUDA-Q output).
- ``cc.loop`` uses block arguments + ``cc.condition %cond(%args...)`` in the
  while-region, and ``cc.continue %args...`` in body/step regions to thread
  loop-carried state.

To comply with CUDA-Q's strict scalar requirements for classical control flow,
any loop-carried ``tensor<T>`` values are automatically unwrapped to scalars
(``T``) at region boundaries using ``tensor.extract``, and re-wrapped after
the op using ``tensor.from_elements``.  These wrappers are cleaned up by
Pass 3 (tensor unwrap).

CUDA-Q Hotfix
-------------
A known CUDA-Q compiler bug causes inclusive loop bounds (``sge``, ``sle``) in
``cc.loop`` while-conditions to terminate prematurely.  The hotfix rewrites
``A sge B`` to ``A sgt (B-1)`` and ``A sle B`` to ``A slt (B+1)`` before emitting the condition.
"""

from typing import Optional, List, Any

from xdsl.dialects import scf, arith, tensor
from xdsl.dialects.scf import IfOp, ForOp, WhileOp, YieldOp, ConditionOp
from xdsl.dialects.builtin import IntegerAttr, i1, i64, ModuleOp, TensorType
from xdsl.ir import Block, Region, SSAValue, Operation, Attribute

from xdsl.rewriter import Rewriter, InsertPoint
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)

from qrisp.jasp.mlir.quake_lowering.cc_dialect import (
    CcConditionOp,
    CcContinueOp,
    CcIfOp,
    CcLoopOp,
)

# ===================================================================
# Public entry point
# ===================================================================


def lower_scf_to_cc(module: ModuleOp) -> None:
    """In-place PASS 2: lower SCF structured control flow to the CC dialect using Pattern Rewriters."""
    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier(
            [
                ScfIfPattern(),
                ScfWhilePattern(),
                ScfForPattern(),
            ]
        ),
        walk_regions_first=True,  # Enables bottom-up traversal (replaces manual _process_region recursion)
    )
    walker.rewrite_module(module)


# ===================================================================
# IR search helpers
# ===================================================================


def _find_condition_op(block: Block) -> Optional[ConditionOp]:
    """Return the ``scf.condition`` op in *block*, or None."""
    for op in block.ops:
        if isinstance(op, ConditionOp):
            return op
    return None


def _find_trailing_yield(block: Block) -> Optional[YieldOp]:
    """Return the trailing ``scf.yield`` in *block*, or None."""
    ops = list(block.ops)
    if ops and isinstance(ops[-1], YieldOp):
        return ops[-1]
    return None


# ===================================================================
# Tensor unwrap/wrap utilities
# ===================================================================


def _is_rank0_tensor(t: Attribute) -> bool:
    """Return True if *t* is a rank-0 TensorType."""
    return isinstance(t, TensorType) and len(t.get_shape()) == 0


def _get_element_type(t: Attribute) -> Optional[Attribute]:
    """Extract the element type from a TensorType."""
    if isinstance(t, TensorType):
        return t.element_type
    return None


def _unwrap(val: SSAValue, insert_hook: Operation) -> SSAValue:
    """Extract a scalar from a rank-0 tensor; no-op for scalars."""
    if _is_rank0_tensor(val.type):
        elem_type = _get_element_type(val.type)
        if elem_type is None:
            return val
        ext = tensor.ExtractOp(val, [], elem_type)
        block = insert_hook.parent_block
        if callable(block):
            block = block()
        block.insert_ops_before([ext], insert_hook)
        return ext.results[0]
    return val


def _rewrite_region_args_for_tensors(region: Region) -> Region:
    """Create a new Region with scalar block args, wrapping back to tensors inside.

    If a block argument was ``tensor<T>``, the new block gets a ``T`` arg and
    a ``tensor.from_elements`` is inserted to bridge existing uses.
    """
    block = region.blocks[0]
    new_arg_types: List[Attribute] = []
    needs_wrap = False

    for arg in block.args:
        if _is_rank0_tensor(arg.type):
            elem_type = _get_element_type(arg.type)
            if elem_type:
                new_arg_types.append(elem_type)
                needs_wrap = True
            else:
                new_arg_types.append(arg.type)
        else:
            new_arg_types.append(arg.type)

    if not needs_wrap:
        return region

    new_block = Block(arg_types=new_arg_types)

    for old_arg, new_arg in zip(block.args, new_block.args):
        if _is_rank0_tensor(old_arg.type):
            wrap = tensor.FromElementsOp.create(
                operands=[new_arg],
                result_types=[old_arg.type],
            )
            new_block.add_op(wrap)
            old_arg.replace_all_uses_with(wrap.results[0])
        else:
            old_arg.replace_all_uses_with(new_arg)

    for op in list(block.ops):
        op.detach()
        new_block.add_op(op)

    return Region([new_block])


# ===================================================================
# CUDA-Q backend bug hotfix
# ===================================================================


def _apply_cudaq_while_hotfix(block: Block) -> None:
    """Rewrite ``A sge B`` → ``A sgt (B-1)`` and ``A sle B`` → ``A slt (B+1)`` to work around a CUDA-Q bug.

    The CUDA-Q compiler mishandles inclusive loop bounds (``sge``, ``sle``) in
    ``cc.loop`` while-conditions, causing premature loop termination.
    Fixed by: https://github.com/NVIDIA/cuda-quantum/pull/4412 (not yet in 0.14.2)
    """
    for op in list(block.ops):
        if isinstance(op, arith.CmpiOp):
            pred_attr = op.predicate
            if pred_attr is None:
                continue

            val = getattr(pred_attr, "value", pred_attr)
            pred_val = getattr(val, "data", val)
            pred_val_str = str(pred_val)

            if pred_val == 5 or "sge" in pred_val_str:
                lhs = op.operands[0]
                rhs = op.operands[1]

                one = arith.ConstantOp(IntegerAttr(1, i64))
                rhs_new = arith.SubiOp(rhs, one.result)
                new_cmp = arith.CmpiOp(lhs, rhs_new.result, "sgt")

                block.insert_ops_before([one, rhs_new, new_cmp], op)
                op.results[0].replace_all_uses_with(new_cmp.results[0])
                Rewriter.erase_op(op, safe_erase=False)

            if pred_val == 3 or "sle" in pred_val_str:
                lhs = op.operands[0]
                rhs = op.operands[1]

                one = arith.ConstantOp(IntegerAttr(1, i64))
                rhs_new = arith.AddiOp(rhs, one.result)
                new_cmp = arith.CmpiOp(lhs, rhs_new.result, "slt")

                block.insert_ops_before([one, rhs_new, new_cmp], op)
                op.results[0].replace_all_uses_with(new_cmp.results[0])
                Rewriter.erase_op(op, safe_erase=False)


# ===================================================================
# Rewrite Patterns
# ===================================================================


class ScfIfPattern(RewritePattern):
    """Pattern to convert ``scf.if`` → ``cc.if`` with SSA result threading."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, if_op: IfOp, rewriter: PatternRewriter) -> None:
        cond = if_op.cond

        # Determine CC result types (unwrap tensors)
        result_types: List[Attribute] = []
        for t in if_op.result_types:
            if _is_rank0_tensor(t):
                elem_t = _get_element_type(t)
                result_types.append(elem_t if elem_t else t)
            else:
                result_types.append(t)

        # Detach regions
        true_region = if_op.detach_region(if_op.regions[0])
        false_region = if_op.detach_region(if_op.regions[0]) if len(if_op.regions) > 0 else Region([Block()])

        # Replace scf.yield → cc.continue (with unwrapped operands) inside detached regions
        for region in (true_region, false_region):
            for block in region.blocks:
                for op in list(block.ops):
                    if isinstance(op, scf.YieldOp):
                        unwrapped = [_unwrap(v, op) for v in op.operands]
                        cc_continue = CcContinueOp(*unwrapped)
                        block.insert_ops_before([cc_continue], op)
                        Rewriter.erase_op(op, safe_erase=False)

        # Build cc.if with result types
        cc_if = CcIfOp(cond, result_types, true_region, false_region)
        rewriter.insert_op(cc_if, InsertPoint.before(if_op))

        # Replace original results (re-wrap tensors if needed)
        for i, res in enumerate(if_op.results):
            orig_type = res.type
            scalar_res = cc_if.res[i]
            if _is_rank0_tensor(orig_type):
                wrap = tensor.FromElementsOp.create(
                    operands=[scalar_res],
                    result_types=[orig_type],
                )
                rewriter.insert_op(wrap, InsertPoint.before(if_op))
                res.replace_all_uses_with(wrap.results[0])
            else:
                res.replace_all_uses_with(scalar_res)

        rewriter.erase_op(if_op)


class ScfForPattern(RewritePattern):
    """Pattern to convert ``scf.for`` → ``cc.loop`` with SSA block arguments."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, for_op: ForOp, rewriter: PatternRewriter) -> None:
        lb = for_op.operands[0]
        ub = for_op.operands[1]
        step = for_op.operands[2]
        iter_args = list(for_op.operands[3:])

        init_args = [lb] + iter_args

        # Helper to unwrap values directly in the outer block scope via the rewriter
        def _unwrap_outer(val: SSAValue) -> SSAValue:
            if _is_rank0_tensor(val.type):
                elem_type = _get_element_type(val.type)
                if elem_type:
                    ext = tensor.ExtractOp(val, [], elem_type)
                    rewriter.insert_op(ext, InsertPoint.before(for_op))
                    return ext.results[0]
            return val

        unwrapped_init_args = [_unwrap_outer(arg) for arg in init_args]
        arg_types = [arg.type for arg in unwrapped_init_args]

        unwrapped_ub = _unwrap_outer(ub)
        unwrapped_step = _unwrap_outer(step)

        # ---- 1. While Region (Condition): iv < ub ----------------------------
        cond_region = Region([Block(arg_types=arg_types)])
        cond_block = cond_region.blocks[0]
        iv_cond = cond_block.args[0]

        cmp = arith.CmpiOp(iv_cond, unwrapped_ub, "slt")
        cond_block.add_op(cmp)
        cond_block.add_op(CcConditionOp(cmp.result, *cond_block.args))

        # ---- 2. Body Region: rewrite from scf.for body ----------------------
        body_region = for_op.detach_region(for_op.regions[0])
        body_region = _rewrite_region_args_for_tensors(body_region)
        body_block = body_region.blocks[0]

        yield_op = _find_trailing_yield(body_block)
        if yield_op is not None:
            unwrapped_yields = [_unwrap(v, yield_op) for v in yield_op.operands]
            cc_continue = CcContinueOp(body_block.args[0], *unwrapped_yields)
            body_block.insert_ops_before([cc_continue], yield_op)
            Rewriter.erase_op(yield_op, safe_erase=False)

        # ---- 3. Step Region: iv += step --------------------------------------
        step_region = Region([Block(arg_types=arg_types)])
        step_block = step_region.blocks[0]

        iv_step = step_block.args[0]
        next_iv = arith.AddiOp(iv_step, unwrapped_step)
        step_block.add_op(next_iv)

        next_args = [next_iv.result] + list(step_block.args[1:])
        step_block.add_op(CcContinueOp(*next_args))

        # ---- 4. Build cc.loop ------------------------------------------------
        cc_loop = CcLoopOp(
            arguments=unwrapped_init_args,
            result_types=arg_types,
            while_region=cond_region,
            body_region=body_region,
            step_region=step_region,
        )
        rewriter.insert_op(cc_loop, InsertPoint.before(for_op))

        # ---- 5. Replace iter-arg results (skip IV at index 0) ----------------
        loop_results = list(cc_loop.res)[1:]
        for i, res in enumerate(for_op.results):
            orig_type = res.type
            scalar_res = loop_results[i]
            if _is_rank0_tensor(orig_type):
                wrap = tensor.FromElementsOp.create(
                    operands=[scalar_res],
                    result_types=[orig_type],
                )
                rewriter.insert_op(wrap, InsertPoint.before(for_op))
                res.replace_all_uses_with(wrap.results[0])
            else:
                res.replace_all_uses_with(scalar_res)

        rewriter.erase_op(for_op)


class ScfWhilePattern(RewritePattern):
    """Pattern to convert ``scf.while`` → ``cc.loop`` with SSA block arguments."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, while_op: WhileOp, rewriter: PatternRewriter) -> None:
        init_args = list(while_op.operands)

        def _unwrap_outer(val: SSAValue) -> SSAValue:
            if _is_rank0_tensor(val.type):
                elem_type = _get_element_type(val.type)
                if elem_type:
                    ext = tensor.ExtractOp(val, [], elem_type)
                    rewriter.insert_op(ext, InsertPoint.before(while_op))
                    return ext.results[0]
            return val

        unwrapped_init_args = [_unwrap_outer(arg) for arg in init_args]
        arg_types = [arg.type for arg in unwrapped_init_args]

        # Detach regions
        before_region = while_op.detach_region(while_op.regions[0])
        after_region = while_op.detach_region(while_op.regions[0])

        # ---- 1. While Region (Condition) ------------------------------------
        before_region = _rewrite_region_args_for_tensors(before_region)
        before_block = before_region.blocks[0]

        # Explicitly scope the hotfix ONLY to the while-condition check block
        _apply_cudaq_while_hotfix(before_block)

        cond_op = _find_condition_op(before_block)

        cond_val = cond_op.operands[0]
        forwarded = list(cond_op.operands)[1:]
        unwrapped_forwarded = [_unwrap(f, cond_op) for f in forwarded]
        cc_cond = CcConditionOp(cond_val, *unwrapped_forwarded)
        before_block.insert_ops_before([cc_cond], cond_op)
        Rewriter.erase_op(cond_op, safe_erase=False)

        # ---- 2. Body Region --------------------------------------------------
        after_region = _rewrite_region_args_for_tensors(after_region)
        after_block = after_region.blocks[0]

        yield_op = _find_trailing_yield(after_block)

        unwrapped_yields = [_unwrap(v, yield_op) for v in yield_op.operands]
        step_arg_types = [u.type for u in unwrapped_yields]
        cc_continue = CcContinueOp(*unwrapped_yields)
        after_block.insert_ops_before([cc_continue], yield_op)
        Rewriter.erase_op(yield_op, safe_erase=False)

        # ---- 3. Step Region (passthrough) ------------------------------------
        step_region = Region([Block(arg_types=step_arg_types)])
        step_block = step_region.blocks[0]
        step_block.add_op(CcContinueOp(*step_block.args))

        # ---- 4. Build cc.loop ------------------------------------------------
        cc_loop = CcLoopOp(
            arguments=unwrapped_init_args,
            result_types=arg_types,
            while_region=before_region,
            body_region=after_region,
            step_region=step_region,
        )
        rewriter.insert_op(cc_loop, InsertPoint.before(while_op))

        # ---- 5. Replace results (re-wrap tensors if needed) ------------------
        for i, res in enumerate(while_op.results):
            orig_type = res.type
            scalar_res = cc_loop.res[i]
            if _is_rank0_tensor(orig_type):
                wrap = tensor.FromElementsOp.create(
                    operands=[scalar_res],
                    result_types=[orig_type],
                )
                rewriter.insert_op(wrap, InsertPoint.before(while_op))
                res.replace_all_uses_with(wrap.results[0])
            else:
                res.replace_all_uses_with(scalar_res)

        rewriter.erase_op(while_op)
