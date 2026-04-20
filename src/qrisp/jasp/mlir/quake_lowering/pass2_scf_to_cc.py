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

Replaces structured control-flow ops (``scf.if``, ``scf.while``) with their
CC-dialect equivalents (``cc.if``, ``cc.loop``).  This keeps the IR in
structured form, suitable for the CUDA-Q kernel compilation pipeline.

Design notes
------------
* ``scf.if`` with **no results** maps trivially to ``cc.if``.
* ``scf.if`` with results is converted by introducing ``cc.alloca`` /
  ``cc.store`` / ``cc.load`` patterns for each yielded value – *not* yet
  implemented; such ops are left in SCF form (a warning is emitted).
* ``scf.while`` with **no results** (after Pass 1 qst removal) maps to
  ``cc.loop``.
* ``scf.while`` with non-void results is similarly not yet supported and is
  left in place.

Extending
---------
To add support for SCF ops with results, implement
:func:`_convert_yielded_value` and call it from the converter functions.
"""


import warnings
from typing import Sequence

from xdsl.dialects import scf, arith
from xdsl.dialects.builtin import i1
from xdsl.ir import Block, Region, SSAValue
from xdsl.rewriter import Rewriter

from qrisp.jasp.mlir.quake_lowering.cc_dialect import (
    CcConditionOp,
    CcContinueOp,
    CcIfOp,
    CcLoopOp,
    CcAllocaOp,
    CcStoreOp,
    CcLoadOp,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def lower_scf_to_cc(module) -> None:
    """In-place PASS 2: lower SCF structured control flow to the CC dialect.

    Parameters
    ----------
    module:
        An xDSL ``builtin.ModuleOp`` that has already been processed by
        :func:`~qrisp.jasp.mlir.quake_lowering.pass1_jasp_to_quake.lower_jasp_to_quake`.
    """
    for op in list(module.body.blocks[0].ops):
        if op.name == "func.func":
            _process_region(op.body)

# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

def _replace_all_uses_with(val: SSAValue, new_val: SSAValue) -> None:
    """Provide compatibility between xDSL < 0.57 and xDSL >= 0.57."""
    if hasattr(val, "replace_all_uses_with"):
        val.replace_all_uses_with(new_val)
    else:  
        val.replace_by(new_val)

# ---------------------------------------------------------------------------
# Recursive region/block/op processing
# ---------------------------------------------------------------------------


def _process_region(region: Region) -> None:
    for block in region.blocks:
        _process_block(block)


def _process_block(block: Block) -> None:
    for op in list(block.ops):
        _process_op(op, block)

"""
def _process_op(op, block: Block) -> None:
    if op.name == "scf.if":
        _convert_scf_if(op, block)
    elif op.name == "scf.while":
        _convert_scf_while(op, block)
    elif op.name in ("scf.for", "scf.index_switch"):
        # Recurse into nested regions; leave the op itself in place.
        for region in op.regions:
            _process_region(region)
    else:
        # Recurse into any nested regions (e.g. linalg.generic bodies).
        for region in op.regions:
            _process_region(region)
"""
def _process_op(op, block: Block) -> None:
    if op.name == "scf.if":
        _convert_scf_if(op, block)
    elif op.name == "scf.while":
        _convert_scf_while(op, block)
    elif op.name == "scf.for":
        _convert_scf_for(op, block)
    elif op.name == "scf.index_switch":
        # Recurse into nested regions; leave the op itself in place.
        for region in op.regions:
            _process_region(region)
    else:
        # Recurse into any nested regions (e.g. linalg.generic bodies).
        for region in op.regions:
            _process_region(region)

# ---------------------------------------------------------------------------
# scf.if → cc.if
# ---------------------------------------------------------------------------


def _convert_scf_if(if_op, outer_block: Block) -> None:
    """Convert ``scf.if`` to ``cc.if``.

    Only no-result ``scf.if`` ops are converted.  Ops with results are left
    in SCF form with a warning.
    """
    # Recurse into branches first so nested SCF ops are also lowered.
    for region in if_op.regions:
        _process_region(region)

    if list(if_op.result_types):
        # Has results – not yet supported.
        warnings.warn(
            "scf.if with results cannot be lowered to cc.if yet; left in SCF form.",
            stacklevel=4,
        )
        return

    cond = if_op.cond

    # Convert scf.yield terminators to cc.continue inside both regions
    true_region = if_op.detach_region(if_op.regions[0])
    false_region = if_op.detach_region(if_op.regions[0])

    _replace_yield_with_continue(true_region)
    _replace_yield_with_continue(false_region)

    cc_if = CcIfOp(cond, true_region, false_region)
    Rewriter.replace_op(if_op, cc_if, [], safe_erase=False)


def _replace_yield_with_continue(region: Region) -> None:
    """Replace ``scf.yield`` (no-op yield) with ``cc.continue`` in *region*."""
    for block in region.blocks:
        for op in list(block.ops):
            if op.name == "scf.yield" and not list(op.operands):
                continue_op = CcContinueOp()
                block.insert_ops_before([continue_op], op)
                Rewriter.erase_op(op, safe_erase=False)
            elif op.name == "scf.yield":
                # Has operands – can't trivially replace; leave for now.
                pass


# ---------------------------------------------------------------------------
# scf.while → cc.loop
# ---------------------------------------------------------------------------


def _convert_scf_while(while_op, outer_block: Block) -> None:
    """Convert ``scf.while`` to ``cc.loop``.

    Only no-result ``scf.while`` ops (after Pass 1 qst removal) are converted.
    """
    # Recurse first
    for region in while_op.regions:
        _process_region(region)

    result_types = list(while_op.res.types)
    if result_types:
        warnings.warn(
            "scf.while with results cannot be lowered to cc.loop yet; left in SCF form.",
            stacklevel=4,
        )
        return

    # Build the cc.loop while-region (condition) and body-region
    before = while_op.detach_region(while_op.before_region)
    after = while_op.detach_region(while_op.after_region)

    # Convert scf.condition → cc.condition inside before-region
    _convert_condition_op(before)

    # Convert scf.yield → cc.continue inside after-region
    _replace_yield_with_continue(after)

    cc_loop = CcLoopOp(
        arguments=[],
        result_types=[],
        while_region=before,
        body_region=after,
    )
    Rewriter.replace_op(while_op, cc_loop, [], safe_erase=False)


def _convert_condition_op(region: Region) -> None:
    """Replace ``scf.condition`` with ``cc.condition`` inside *region*."""
    for block in region.blocks:
        for op in list(block.ops):
            if op.name == "scf.condition":
                cond = op.operands[0]
                extra_args = list(op.operands)[1:]  # loop-carried (should be empty)
                cc_cond = CcConditionOp(cond, *extra_args)
                block.insert_ops_before([cc_cond], op)
                Rewriter.erase_op(op, safe_erase=False)

# ---------------------------------------------------------------------------
# scf.for → cc.loop
# ---------------------------------------------------------------------------

def _convert_scf_for(for_op, outer_block: Block) -> None:
    """Convert ``scf.for`` to ``cc.loop``.
    
    Uses the cc.alloca / cc.store / cc.load pattern to support yielded values 
    and the loop induction variable.
    """
    # 1. Recurse first to process inner SCF/QST ops
    for region in for_op.regions:
        _process_region(region)

    lb = for_op.operands[0]
    ub = for_op.operands[1]
    step = for_op.operands[2]
    iter_args = list(for_op.operands[3:])

    # Extract block args from the original loop body
    body_block = for_op.regions[0].blocks[0]
    iv_arg = body_block.args[0]
    iter_block_args = list(body_block.args[1:])

    # 2. Allocate and initialize memory for iter_args and Induction Variable (IV)
    allocas = []
    for arg, init_val in zip(iter_block_args, iter_args):
        alloca = CcAllocaOp(arg.type)
        store = CcStoreOp(init_val, alloca.result)
        outer_block.insert_ops_before([alloca, store], for_op)
        allocas.append(alloca)

    iv_alloca = CcAllocaOp(iv_arg.type)
    iv_store = CcStoreOp(lb, iv_alloca.result)
    outer_block.insert_ops_before([iv_alloca, iv_store], for_op)

    # 3. Build the condition region: `while (IV < UB)`
    cond_region = Region([Block()])
    cond_block = cond_region.blocks[0]
    
    cond_iv_load = CcLoadOp(iv_alloca.result)
    cond_block.add_op(cond_iv_load)
    
    cmp_op = arith.CmpiOp(cond_iv_load.result, ub, "slt")
    cond_block.add_op(cmp_op)
    
    cond_op = CcConditionOp(cmp_op.result)
    cond_block.add_op(cond_op)

    # 4. Build the loop body region
    body_region = Region([Block()])
    new_body_block = body_region.blocks[0]

    # Load IV at the start of the body block
    iv_load = CcLoadOp(iv_alloca.result)
    new_body_block.add_op(iv_load)
    _replace_all_uses_with(iv_arg, iv_load.result)

    # Load iter args at the start of the body block
    for i, iter_b_arg in enumerate(iter_block_args):
        iload = CcLoadOp(allocas[i].result)
        new_body_block.add_op(iload)
        _replace_all_uses_with(iter_b_arg, iload.result)

    # Move all original operations from scf.for body into new_body_block
    for op in list(body_block.ops):
        op.detach()
        new_body_block.add_op(op)

    # 5. Process scf.yield: Replace with cc.store, IV increment, and cc.continue
    # FIX: Use list(ops)[-1] because .last_op doesn't exist in older xDSL
    body_ops = list(new_body_block.ops)
    yield_op = body_ops[-1] if body_ops else None
    
    if yield_op and yield_op.name == "scf.yield":
        # Store the yielded variables back into memory
        stores_to_insert = []
        for i, operand in enumerate(yield_op.operands):
            store = CcStoreOp(operand, allocas[i].result)
            stores_to_insert.append(store)
        
        if stores_to_insert:
            new_body_block.insert_ops_before(stores_to_insert, yield_op)

        # Increment IV: next_iv = iv + step
        # FIX: Check for both Addi and AddiOp depending on the exact xDSL patch
        addi_cls = getattr(arith, "Addi", getattr(arith, "AddiOp", None))
        next_iv = addi_cls(iv_load.result, step)
        
        iv_update_store = CcStoreOp(next_iv.result, iv_alloca.result)
        new_body_block.insert_ops_before([next_iv, iv_update_store], yield_op)

        # Insert continue and safely detach the original yield
        continue_op = CcContinueOp()
        new_body_block.insert_ops_before([continue_op], yield_op)
        yield_op.detach()  

    # 6. Create the final cc.loop op
    cc_loop = CcLoopOp(
        arguments=[],
        result_types=[],
        while_region=cond_region,
        body_region=body_region
    )
    outer_block.insert_ops_before([cc_loop], for_op)

    # 7. Extract the final accumulated results and replace uses of scf.for
    final_loads = []
    for alloca in allocas:
        load = CcLoadOp(alloca.result)
        final_loads.append(load)
        outer_block.insert_ops_before([load], for_op)

    # FIX: Check for both .results and .res depending on xDSL patch
    loop_results = getattr(for_op, "results", getattr(for_op, "res", []))
    for i, res in enumerate(loop_results):
        _replace_all_uses_with(res, final_loads[i].result)

    # FIX: Safely detach the old scf.for instead of using the Rewriter
    for_op.detach()