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
A known CUDA-Q compiler bug causes inclusive loop bounds (``sge``) in
``cc.loop`` while-conditions to terminate prematurely.  The hotfix rewrites
``A sge B`` to ``A sgt (B-1)`` before emitting the condition.
"""

from xdsl.dialects import scf, arith, tensor
from xdsl.dialects.builtin import IntegerAttr, i1, i64
from xdsl.ir import Block, Region, SSAValue, Operation
from xdsl.rewriter import Rewriter

from qrisp.jasp.mlir.quake_lowering.cc_dialect import (
    CcConditionOp,
    CcContinueOp,
    CcIfOp,
    CcLoopOp,
)


# ===================================================================
# Public entry point
# ===================================================================


def lower_scf_to_cc(module) -> None:
    """In-place PASS 2: lower SCF structured control flow to the CC dialect."""
    for op in list(module.body.blocks[0].ops):
        if op.name == "func.func":
            _process_region(op.body)


# ===================================================================
# xDSL compatibility shims
# ===================================================================


def _replace_all_uses_with(val: SSAValue, new_val: SSAValue) -> None:
    if hasattr(val, "replace_all_uses_with"):
        val.replace_all_uses_with(new_val)
    else:
        val.replace_by(new_val)


def _get_results(op) -> list:
    return list(getattr(op, "results", getattr(op, "res", [])))


# ===================================================================
# IR search helpers
# ===================================================================


def _find_condition_op(block: Block):
    """Return the ``scf.condition`` op in *block*, or None."""
    for op in block.ops:
        if op.name == "scf.condition":
            return op
    return None


def _find_trailing_yield(block: Block):
    """Return the trailing ``scf.yield`` in *block*, or None."""
    ops = list(block.ops)
    if ops and ops[-1].name == "scf.yield":
        return ops[-1]
    return None


# ===================================================================
# Region traversal
# ===================================================================


def _process_region(region: Region) -> None:
    for block in region.blocks:
        for op in list(block.ops):
            if op.name == "scf.if":
                _convert_scf_if(op, block)
            elif op.name == "scf.while":
                _convert_scf_while(op, block)
            elif op.name == "scf.for":
                _convert_scf_for(op, block)
            else:
                for reg in op.regions:
                    _process_region(reg)


# ===================================================================
# Tensor unwrap/wrap utilities
# ===================================================================


def _is_rank0_tensor(t) -> bool:
    """Return True if *t* is a rank-0 TensorType."""
    return getattr(t, "name", "") == "tensor" and not t.get_shape()

def _is_tensor(t) -> bool:
    """Return True if *t* is a rank-0 TensorType (only these get unwrapped)."""
    return _is_rank0_tensor(t)


def _get_element_type(t):
    """Extract the element type from a TensorType."""
    if hasattr(t, "get_element_type"):
        return t.get_element_type()
    if hasattr(t, "element_type"):
        return t.element_type
    if hasattr(t, "parameters"):
        return t.parameters[0]
    return None


def _unwrap(val: SSAValue, insert_hook: Operation) -> SSAValue:
    """Extract a scalar from a rank-0 tensor; no-op for scalars."""
    if _is_tensor(val.type):
        elem_type = _get_element_type(val.type)
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
    new_arg_types = []
    needs_wrap = False

    for arg in block.args:
        if _is_tensor(arg.type):
            new_arg_types.append(_get_element_type(arg.type))
            needs_wrap = True
        else:
            new_arg_types.append(arg.type)

    if not needs_wrap:
        return region

    new_block = Block(arg_types=new_arg_types)

    for old_arg, new_arg in zip(block.args, new_block.args):
        if _is_tensor(old_arg.type):
            wrap = tensor.FromElementsOp.create(
                operands=[new_arg],
                result_types=[old_arg.type],
            )
            new_block.add_op(wrap)
            _replace_all_uses_with(old_arg, wrap.results[0])
        else:
            _replace_all_uses_with(old_arg, new_arg)

    for op in list(block.ops):
        op.detach()
        new_block.add_op(op)

    return Region([new_block])


# ===================================================================
# CUDA-Q backend bug hotfix
# ===================================================================


def _apply_cudaq_while_hotfix(block: Block) -> None:
    """Rewrite ``A sge B`` → ``A sgt (B-1)`` to work around a CUDA-Q bug.

    The CUDA-Q compiler mishandles inclusive loop bounds (``sge``) in
    ``cc.loop`` while-conditions, causing premature loop termination.
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
                _replace_all_uses_with(op.results[0], new_cmp.results[0])
                Rewriter.erase_op(op, safe_erase=False)


# ===================================================================
# scf.if → cc.if (with SSA results)
# ===================================================================


def _convert_scf_if(if_op, outer_block: Block) -> None:
    """Convert ``scf.if`` → ``cc.if`` with SSA result threading.

    Result values are carried by ``cc.continue %val`` in each branch.
    Rank-0 tensors are unwrapped to scalars at the boundary.
    """
    # Recurse into nested SCF ops first
    for region in if_op.regions:
        _process_region(region)

    cond = if_op.cond

    # Determine CC result types (unwrap tensors)
    result_types = []
    for t in if_op.result_types:
        if _is_tensor(t):
            result_types.append(_get_element_type(t))
        else:
            result_types.append(t)

    # Detach regions
    true_region = if_op.detach_region(if_op.regions[0])
    false_region = (
        if_op.detach_region(if_op.regions[0])
        if len(if_op.regions) > 0
        else Region([Block()])
    )

    # Replace scf.yield → cc.continue (with unwrapped operands)
    for region in (true_region, false_region):
        for block in region.blocks:
            for op in list(block.ops):
                if op.name == "scf.yield":
                    unwrapped = [_unwrap(v, op) for v in op.operands]
                    cc_continue = CcContinueOp(*unwrapped)
                    block.insert_ops_before([cc_continue], op)
                    Rewriter.erase_op(op, safe_erase=False)

    # Build cc.if with result types
    cc_if = CcIfOp(cond, result_types, true_region, false_region)
    outer_block.insert_ops_before([cc_if], if_op)

    # Replace original results (re-wrap tensors if needed)
    for i, res in enumerate(_get_results(if_op)):
        orig_type = res.type
        scalar_res = cc_if.res[i]
        if _is_tensor(orig_type):
            wrap = tensor.FromElementsOp.create(
                operands=[scalar_res],
                result_types=[orig_type],
            )
            outer_block.insert_ops_before([wrap], if_op)
            _replace_all_uses_with(res, wrap.results[0])
        else:
            _replace_all_uses_with(res, scalar_res)

    Rewriter.erase_op(if_op, safe_erase=False)


# ===================================================================
# scf.for → cc.loop
# ===================================================================


def _convert_scf_for(for_op, outer_block: Block) -> None:
    """Convert ``scf.for`` → ``cc.loop`` with SSA block arguments.

    The induction variable (IV) is carried as the first block arg;
    iter-args follow.  The step region increments the IV.
    """
    # Recurse first
    for region in for_op.regions:
        _process_region(region)

    lb = for_op.operands[0]
    ub = for_op.operands[1]
    step = for_op.operands[2]
    iter_args = list(for_op.operands[3:])

    # Initial args: [lb, iter_arg0, iter_arg1, ...]
    init_args = [lb] + iter_args
    unwrapped_init_args = [_unwrap(arg, for_op) for arg in init_args]
    arg_types = [arg.type for arg in unwrapped_init_args]

    unwrapped_ub = _unwrap(ub, for_op)
    unwrapped_step = _unwrap(step, for_op)

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
        # Forward: [iv, yielded_iter_args...]
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
    outer_block.insert_ops_before([cc_loop], for_op)

    # ---- 5. Replace iter-arg results (skip IV at index 0) ----------------
    loop_results = list(cc_loop.res)[1:]  # skip IV result
    for i, res in enumerate(_get_results(for_op)):
        orig_type = res.type
        scalar_res = loop_results[i]
        if _is_tensor(orig_type):
            wrap = tensor.FromElementsOp.create(
                operands=[scalar_res],
                result_types=[orig_type],
            )
            outer_block.insert_ops_before([wrap], for_op)
            _replace_all_uses_with(res, wrap.results[0])
        else:
            _replace_all_uses_with(res, scalar_res)

    Rewriter.erase_op(for_op, safe_erase=False)


# ===================================================================
# scf.while → cc.loop
# ===================================================================


def _convert_scf_while(while_op, outer_block: Block) -> None:
    """Convert ``scf.while`` → ``cc.loop`` with SSA block arguments.

    The before-region becomes the while-region (condition check);
    the after-region becomes the body-region.  A passthrough step-region
    is synthesized.
    """
    # Recurse first
    for region in while_op.regions:
        _process_region(region)

    init_args = list(while_op.operands)
    unwrapped_init_args = [_unwrap(arg, while_op) for arg in init_args]
    arg_types = [arg.type for arg in unwrapped_init_args]

    # Detach regions
    before_region = while_op.detach_region(while_op.regions[0])
    after_region = while_op.detach_region(while_op.regions[0])

    # ---- 1. While Region (Condition) ------------------------------------
    before_region = _rewrite_region_args_for_tensors(before_region)
    before_block = before_region.blocks[0]

    _apply_cudaq_while_hotfix(before_block)

    cond_op = _find_condition_op(before_block)
    if cond_op is not None:
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
    step_arg_types = [arg.type for arg in after_block.args]
    if yield_op is not None:
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
    outer_block.insert_ops_before([cc_loop], while_op)

    # ---- 5. Replace results (re-wrap tensors if needed) ------------------
    for i, res in enumerate(_get_results(while_op)):
        orig_type = res.type
        scalar_res = cc_loop.res[i]
        if _is_tensor(orig_type):
            wrap = tensor.FromElementsOp.create(
                operands=[scalar_res],
                result_types=[orig_type],
            )
            outer_block.insert_ops_before([wrap], while_op)
            _replace_all_uses_with(res, wrap.results[0])
        else:
            _replace_all_uses_with(res, scalar_res)

    Rewriter.erase_op(while_op, safe_erase=False)