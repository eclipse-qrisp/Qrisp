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
PASS 2 – SCF → CC dialect lowering.

Replaces structured control-flow ops (``scf.if``, ``scf.for``, ``scf.while``)
with their CC-dialect equivalents (``cc.if``, ``cc.loop``).

Strategy
--------
Loop-carried SSA values are promoted to stack memory via a
:class:`_MemorySlot` abstraction (``cc.alloca`` / ``cc.store`` /
``cc.load``), so the resulting ``cc.loop`` has **no** SSA results and no
block arguments in its regions.  This matches the style emitted by the
CUDA-Q compiler.

Memory slot creation automatically **unwraps rank-0 tensors** to scalars
(``tensor<T>`` → ``T``) because CUDA-Q's ``cc.alloca`` only supports scalar
types.  The wrappers (``tensor.extract`` / ``tensor.from_elements``) are
inserted transparently and cleaned up by Pass 3 (tensor unwrap).

For ``scf.while``, an **invariant-detection** step classifies each
loop-carried value as *invariant* (captured from the enclosing scope,
no alloca) or *variant* (promoted to a memory slot).  This avoids illegal
``cc.alloca !quake.veq<?>`` ops that would crash the CUDA-Q runtime.

Architecture
------------
Both ``scf.for`` and ``scf.while`` lowerings share the same memory-slot
infrastructure:

* ``_MemorySlot`` – data class encapsulating an alloca + tensor-wrapping
  metadata
* ``_create_slot`` – alloca + init store (with tensor unwrap)
* ``_load_from_slot`` – ``cc.load`` [+ ``tensor.from_elements``]
* ``_store_to_slot`` – [``tensor.extract`` +] ``cc.store``
* ``_rewrite_block_to_region`` – replace block args with loads / captures,
  move ops to a new region

Important
---------
Ops that are **permanently removed** (``scf.condition``, ``scf.yield``,
``scf.for``, ``scf.while``) must use ``Rewriter.erase_op(op,
safe_erase=False)`` — **not** ``op.detach()``.  The ``detach()`` method
removes the op from its parent block but does **not** release its operand
references, leaving phantom uses that prevent downstream dead-code
elimination from cleaning up dead ``tensor.from_elements`` ops.

``op.detach()`` is only correct when **moving** an op between blocks
(followed immediately by ``new_block.add_op(op)``).
"""

"""
PASS 2 – SCF → CC dialect lowering.

Replaces structured control-flow ops (``scf.if``, ``scf.for``, ``scf.while``)
with their CC-dialect equivalents (``cc.if``, ``cc.loop``) using pure SSA 
semantics. 

To comply with CUDA-Q's strict scalar requirements for classical control flow,
any loop-carried `tensor<T>` values are automatically unwrapped to scalars 
(`T`) at the boundaries using `tensor.extract`, and re-wrapped inside the 
regions and after the loop using `tensor.from_elements`.
"""

from xdsl.dialects import scf, arith, tensor
from xdsl.ir import Block, Region, SSAValue, Operation
from xdsl.rewriter import Rewriter

from qrisp.jasp.mlir.quake_lowering.cc_dialect import (
    CcConditionOp,
    CcContinueOp,
    CcIfOp,
    CcLoopOp,
)


def lower_scf_to_cc(module) -> None:
    for op in list(module.body.blocks[0].ops):
        if op.name == "func.func":
            _process_region(op.body)


def _replace_all_uses_with(val: SSAValue, new_val: SSAValue) -> None:
    if hasattr(val, "replace_all_uses_with"):
        val.replace_all_uses_with(new_val)
    else:
        val.replace_by(new_val)


def _get_arith_addi():
    cls = getattr(arith, "Addi", None) or getattr(arith, "AddiOp", None)
    return cls


def _get_results(op) -> list:
    return list(getattr(op, "results", getattr(op, "res", [])))


def _find_condition_op(block: Block):
    for op in block.ops:
        if op.name == "scf.condition":
            return op
    return None


def _find_trailing_yield(block: Block):
    ops = list(block.ops)
    if ops and ops[-1].name == "scf.yield":
        return ops[-1]
    return None


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
# Tensor Unwrap/Wrap Utilities for CUDA-Q Compatibility
# ===================================================================

def _is_tensor(t) -> bool:
    return getattr(t, "name", "") == "tensor"


def _get_element_type(t):
    if hasattr(t, "get_element_type"):
        return t.get_element_type()
    if hasattr(t, "element_type"):
        return t.element_type
    if hasattr(t, "parameters"):
        return t.parameters[0]
    return None


def _unwrap(val: SSAValue, insert_hook: Operation) -> SSAValue:
    """Extracts a scalar from a rank-0 tensor."""
    if _is_tensor(val.type):
        elem_type = _get_element_type(val.type)
        # Use the native xDSL tensor.ExtractOp
        ext = tensor.ExtractOp(val, [], elem_type)
        
        block = insert_hook.parent_block
        if callable(block):
            block = block()
            
        block.insert_ops_before([ext], insert_hook)
        return ext.results[0]
    return val


def _rewrite_region_args_for_tensors(region: Region) -> Region:
    """
    Creates a new Region with a new Block whose arguments are purely scalars.
    Moves operations over and wraps the new block arguments in tensor.from_elements
    so the interior body operations remain valid.
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
    
    # Insert wrappers in the new block and replace uses
    for old_arg, new_arg in zip(block.args, new_block.args):
        if _is_tensor(old_arg.type):
            # Use the native xDSL tensor.FromElementsOp
            wrap = tensor.FromElementsOp.create(
                operands=[new_arg], 
                result_types=[old_arg.type]
            )
            new_block.add_op(wrap)
            _replace_all_uses_with(old_arg, wrap.results[0])
        else:
            _replace_all_uses_with(old_arg, new_arg)
            
    # Move all operations from the old block to the new block safely
    for op in list(block.ops):
        op.detach()
        new_block.add_op(op)
        
    return Region([new_block])


# ===================================================================
# scf.if → cc.if
# ===================================================================

def _convert_scf_if(if_op, outer_block: Block) -> None:
    for region in if_op.regions:
        _process_region(region)

    cond = if_op.cond

    result_types = []
    for t in if_op.result_types:
        if _is_tensor(t):
            result_types.append(_get_element_type(t))
        else:
            result_types.append(t)

    true_region = if_op.detach_region(if_op.regions[0])
    false_region = if_op.detach_region(if_op.regions[0]) if len(if_op.regions) > 1 else Region([Block()])

    for region in (true_region, false_region):
        for block in region.blocks:
            for op in list(block.ops):
                if op.name == "scf.yield":
                    unwrapped_yields = [_unwrap(opnd, op) for opnd in op.operands]
                    cc_continue = CcContinueOp(*unwrapped_yields)
                    block.insert_ops_before([cc_continue], op)
                    Rewriter.erase_op(op, safe_erase=False)

    cc_if = CcIfOp(cond, result_types, true_region, false_region)
    outer_block.insert_ops_before([cc_if], if_op)
    
    for i, res in enumerate(_get_results(if_op)):
        orig_type = res.type
        scalar_res = cc_if.res[i]
        if _is_tensor(orig_type):
            wrap = tensor.FromElementsOp.create(
                operands=[scalar_res], 
                result_types=[orig_type]
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
    for region in for_op.regions:
        _process_region(region)

    lb = for_op.operands[0]
    ub = for_op.operands[1]
    step = for_op.operands[2]
    iter_args = list(for_op.operands[3:])

    init_args = [lb] + iter_args
    unwrapped_init_args = [_unwrap(arg, for_op) for arg in init_args]
    arg_types = [arg.type for arg in unwrapped_init_args]

    unwrapped_ub = _unwrap(ub, for_op)
    unwrapped_step = _unwrap(step, for_op)

    # ---- 1. While Region (Condition) ----
    cond_region = Region([Block(arg_types=arg_types)])
    cond_block = cond_region.blocks[0]
    iv_cond = cond_block.args[0]
    
    cmp = arith.CmpiOp(iv_cond, unwrapped_ub, "slt")
    cond_block.add_op(cmp)
    cond_block.add_op(CcConditionOp(cmp.result, *cond_block.args))

    # ---- 2. Do Region (Body) ----
    body_region = for_op.detach_region(for_op.regions[0])
    body_region = _rewrite_region_args_for_tensors(body_region)
    body_block = body_region.blocks[0]
    
    yield_op = _find_trailing_yield(body_block)
    if yield_op is not None:
        unwrapped_yields = [_unwrap(opnd, yield_op) for opnd in yield_op.operands]
        cc_continue = CcContinueOp(body_block.args[0], *unwrapped_yields)
        body_block.insert_ops_before([cc_continue], yield_op)
        Rewriter.erase_op(yield_op, safe_erase=False)

    # ---- 3. Step Region (Increment) ----
    step_region = Region([Block(arg_types=arg_types)])
    step_block = step_region.blocks[0]
    
    iv_step = step_block.args[0]
    addi_cls = _get_arith_addi()
    next_iv = addi_cls(iv_step, unwrapped_step)
    step_block.add_op(next_iv)
    
    next_args = [next_iv.result] + list(step_block.args[1:])
    step_block.add_op(CcContinueOp(*next_args))

    # ---- 4. Build cc.loop ----
    cc_loop = CcLoopOp(
        arguments=unwrapped_init_args,
        result_types=arg_types,
        while_region=cond_region,
        body_region=body_region,
        step_region=step_region,
    )
    outer_block.insert_ops_before([cc_loop], for_op)

    # Replace iter_arg results (skipping IV result)
    loop_results = list(cc_loop.res)[1:]
    for i, res in enumerate(_get_results(for_op)):
        orig_type = res.type
        scalar_res = loop_results[i]
        if _is_tensor(orig_type):
            wrap = tensor.FromElementsOp.create(
                operands=[scalar_res], 
                result_types=[orig_type]
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
    for region in while_op.regions:
        _process_region(region)

    init_args = list(while_op.operands)
    unwrapped_init_args = [_unwrap(arg, while_op) for arg in init_args]
    arg_types = [arg.type for arg in unwrapped_init_args]

    before_region = while_op.detach_region(while_op.regions[0])
    after_region = while_op.detach_region(while_op.regions[0])

    # ---- 1. While Region (Condition) ----
    before_region = _rewrite_region_args_for_tensors(before_region)
    before_block = before_region.blocks[0]

    cond_op = _find_condition_op(before_block)
    if cond_op is not None:
        cond_val = cond_op.operands[0]
        forwarded = list(cond_op.operands)[1:]
        unwrapped_forwarded = [_unwrap(f, cond_op) for f in forwarded]
        cc_cond = CcConditionOp(cond_val, *unwrapped_forwarded)
        before_block.insert_ops_before([cc_cond], cond_op)
        Rewriter.erase_op(cond_op, safe_erase=False)

    # ---- 2. Do Region (Body) ----
    after_region = _rewrite_region_args_for_tensors(after_region)
    after_block = after_region.blocks[0]

    yield_op = _find_trailing_yield(after_block)
    step_args_types = [arg.type for arg in after_block.args]
    if yield_op is not None:
        unwrapped_yields = [_unwrap(opnd, yield_op) for opnd in yield_op.operands]
        step_args_types = [u.type for u in unwrapped_yields]
        cc_continue = CcContinueOp(*unwrapped_yields)
        after_block.insert_ops_before([cc_continue], yield_op)
        Rewriter.erase_op(yield_op, safe_erase=False)

    # ---- 3. Step Region (Passthrough) ----
    step_region = Region([Block(arg_types=step_args_types)])
    step_block = step_region.blocks[0]
    step_block.add_op(CcContinueOp(*step_block.args))

    # ---- 4. Build cc.loop ----
    cc_loop = CcLoopOp(
        arguments=unwrapped_init_args,
        result_types=arg_types,
        while_region=before_region,
        body_region=after_region,
        step_region=step_region,
    )
    outer_block.insert_ops_before([cc_loop], while_op)

    for i, res in enumerate(_get_results(while_op)):
        orig_type = res.type
        scalar_res = cc_loop.res[i]
        if _is_tensor(orig_type):
            wrap = tensor.FromElementsOp.create(
                operands=[scalar_res], 
                result_types=[orig_type]
            )
            outer_block.insert_ops_before([wrap], while_op)
            _replace_all_uses_with(res, wrap.results[0])
        else:
            _replace_all_uses_with(res, scalar_res)

    Rewriter.erase_op(while_op, safe_erase=False)