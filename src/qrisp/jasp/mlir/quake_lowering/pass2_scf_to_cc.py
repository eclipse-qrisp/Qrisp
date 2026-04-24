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
Loop-carried SSA values are promoted to memory via ``cc.alloca`` /
``cc.store`` / ``cc.load``, so the resulting ``cc.loop`` has **no** SSA
results and no block arguments in its regions.  This matches the style
emitted by the CUDA-Q compiler.

The promotion is factored into five reusable helpers (see *Shared helpers*
section) that are called by both ``_convert_scf_for`` and
``_convert_scf_while``.
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


# ===================================================================
# Public entry point
# ===================================================================

def lower_scf_to_cc(module) -> None:
    """In-place PASS 2: lower SCF structured control flow to the CC dialect.

    Parameters
    ----------
    module:
        An xDSL ``builtin.ModuleOp`` already processed by Pass 1.
    """
    for op in list(module.body.blocks[0].ops):
        if op.name == "func.func":
            _process_region(op.body)


# ===================================================================
# xDSL compatibility helpers
# ===================================================================

def _replace_all_uses_with(val: SSAValue, new_val: SSAValue) -> None:
    """Compat shim for xDSL < 0.57 vs >= 0.57."""
    if hasattr(val, "replace_all_uses_with"):
        val.replace_all_uses_with(new_val)
    else:
        val.replace_by(new_val)


def _get_arith_addi():
    """Return the arith addi op class, fail-fast if missing."""
    cls = getattr(arith, "Addi", None) or getattr(arith, "AddiOp", None)
    if cls is None:
        raise ImportError(
            "Cannot find arith.Addi or arith.AddiOp in this xDSL version"
        )
    return cls


def _get_results(op) -> list:
    """Return the SSA results of *op* as a list (compat shim)."""
    return list(getattr(op, "results", getattr(op, "res", [])))


# ===================================================================
# Recursive region / block / op traversal
# ===================================================================

def _process_region(region: Region) -> None:
    for block in region.blocks:
        _process_block(block)


def _process_block(block: Block) -> None:
    for op in list(block.ops):
        _process_op(op, block)


def _process_op(op, block: Block) -> None:
    if op.name == "scf.if":
        _convert_scf_if(op, block)
    elif op.name == "scf.while":
        _convert_scf_while(op, block)
    elif op.name == "scf.for":
        _convert_scf_for(op, block)
    else:
        # Recurse into any nested regions (func bodies, linalg, etc.).
        for region in op.regions:
            _process_region(region)


# ===================================================================
# Shared helpers: alloca / store / load memory promotion for loops
# ===================================================================

def _alloca_and_init(
    outer_block: Block,
    init_values: Sequence[SSAValue],
    insert_before,
) -> list:
    """Create ``cc.alloca`` + ``cc.store`` for each value in *init_values*.

    All ops are inserted into *outer_block* just before *insert_before*.
    Returns a list of :class:`CcAllocaOp` (one per value).
    """
    allocas = []
    for init_val in init_values:
        alloca = CcAllocaOp(init_val.type)
        store = CcStoreOp(init_val, alloca.result)
        outer_block.insert_ops_before([alloca, store], insert_before)
        allocas.append(alloca)
    return allocas


def _rewrite_block_with_loads(
    old_block: Block,
    allocas: Sequence,
) -> tuple:
    """Create a fresh ``Region([Block()])``, insert ``cc.load`` for each alloca,
    replace the corresponding *old_block* arg uses, and move all ops.

    The new block has **no** block arguments.

    Returns ``(new_region, new_block, loads)``.

    .. note::
       ``_replace_all_uses_with`` is called while the consuming ops still
       reside in *old_block*.  This is safe because SSA use-def replacement
       only updates pointers; the ops are moved immediately afterwards.
    """
    new_region = Region([Block()])
    new_block = new_region.blocks[0]

    loads = []
    for barg, alloca in zip(list(old_block.args), allocas):
        load = CcLoadOp(alloca.result)
        new_block.add_op(load)
        _replace_all_uses_with(barg, load.result)
        loads.append(load)

    for op in list(old_block.ops):
        op.detach()
        new_block.add_op(op)

    return new_region, new_block, loads


def _replace_yield_with_stores_and_continue(
    block: Block,
    allocas: Sequence,
) -> None:
    """Replace the trailing ``scf.yield`` with stores to *allocas* +
    ``cc.continue``.

    If the block has no trailing ``scf.yield``, this is a no-op.
    """
    ops = list(block.ops)
    if not ops or ops[-1].name != "scf.yield":
        return
    yield_op = ops[-1]

    for operand, alloca in zip(yield_op.operands, allocas):
        block.insert_ops_before([CcStoreOp(operand, alloca.result)], yield_op)

    block.insert_ops_before([CcContinueOp()], yield_op)
    yield_op.detach()


def _replace_condition_with_stores(
    block: Block,
    allocas: Sequence,
) -> None:
    """Replace ``scf.condition(%c) %fwd…`` with stores of forwarded args +
    ``cc.condition(%c)``."""
    for op in list(block.ops):
        if op.name == "scf.condition":
            cond = op.operands[0]
            forwarded = list(op.operands)[1:]
            for val, alloca in zip(forwarded, allocas):
                block.insert_ops_before(
                    [CcStoreOp(val, alloca.result)], op
                )
            block.insert_ops_before([CcConditionOp(cond)], op)
            op.detach()
            break  # exactly one scf.condition per block


def _load_final_and_replace(
    outer_block: Block,
    allocas: Sequence,
    original_results: Sequence[SSAValue],
    insert_before,
) -> None:
    """After a loop: load final values from *allocas* and replace
    *original_results*."""
    for alloca, res in zip(allocas, original_results):
        load = CcLoadOp(alloca.result)
        outer_block.insert_ops_before([load], insert_before)
        _replace_all_uses_with(res, load.result)


# ===================================================================
# scf.if  →  cc.if
# ===================================================================

def _convert_scf_if(if_op, outer_block: Block) -> None:
    """Convert ``scf.if`` to ``cc.if``.

    Only no-result ``scf.if`` is converted.  Ops with results are left in
    SCF form (a warning is emitted).
    """
    # Recurse into branches first so nested SCF ops are also lowered.
    for region in if_op.regions:
        _process_region(region)

    if list(if_op.result_types):
        warnings.warn(
            "scf.if with results cannot be lowered to cc.if yet; "
            "left in SCF form.",
            stacklevel=4,
        )
        return

    cond = if_op.cond

    # After detaching regions[0], the former regions[1] shifts to index 0.
    true_region = if_op.detach_region(if_op.regions[0])
    false_region = if_op.detach_region(if_op.regions[0])

    _replace_yield_with_continue(true_region)
    _replace_yield_with_continue(false_region)

    cc_if = CcIfOp(cond, true_region, false_region)
    Rewriter.replace_op(if_op, cc_if, [], safe_erase=False)


def _replace_yield_with_continue(region: Region) -> None:
    """Replace no-operand ``scf.yield`` with ``cc.continue`` (for ``cc.if``)."""
    for block in region.blocks:
        for op in list(block.ops):
            if op.name == "scf.yield" and not list(op.operands):
                block.insert_ops_before([CcContinueOp()], op)
                Rewriter.erase_op(op, safe_erase=False)
            elif op.name == "scf.yield":
                warnings.warn(
                    "scf.yield with operands inside cc.if region; "
                    "left in place.",
                    stacklevel=4,
                )


# ===================================================================
# scf.for  →  cc.loop
# ===================================================================

def _convert_scf_for(for_op, outer_block: Block) -> None:
    """Convert ``scf.for`` to ``cc.loop`` using alloca/store/load promotion.

    The induction variable (IV) and every iter-arg are promoted to memory.
    A synthetic condition region (``iv < ub``) is built; the body region is
    moved from the original ``scf.for``.
    """
    # Recurse first.
    for region in for_op.regions:
        _process_region(region)

    lb, ub, step = for_op.operands[0], for_op.operands[1], for_op.operands[2]
    iter_args = list(for_op.operands[3:])
    body_block = for_op.regions[0].blocks[0]
    iv_arg = body_block.args[0]

    # ---- 1. Alloca + init ------------------------------------------------
    iter_allocas = _alloca_and_init(outer_block, iter_args, for_op)
    iv_alloca = _alloca_and_init(outer_block, [lb], for_op)[0]

    # ---- 2. Synthetic condition region: while (iv < ub) -------------------
    cond_region = Region([Block()])
    cond_block = cond_region.blocks[0]

    cond_iv_load = CcLoadOp(iv_alloca.result)
    cond_block.add_op(cond_iv_load)
    cmp = arith.CmpiOp(cond_iv_load.result, ub, "slt")
    cond_block.add_op(cmp)
    cond_block.add_op(CcConditionOp(cmp.result))

    # ---- 3. Body region: load IV + iter-args, move ops --------------------
    all_allocas = [iv_alloca] + iter_allocas
    body_region, new_body_block, loads = _rewrite_block_with_loads(
        body_block, all_allocas
    )
    iv_load = loads[0]  # needed for the IV increment below

    # ---- 4. Replace scf.yield: stores + IV increment + continue -----------
    ops = list(new_body_block.ops)
    yield_op = ops[-1] if ops and ops[-1].name == "scf.yield" else None

    if yield_op is not None:
        # Store yielded iter-arg values back to their slots.
        for operand, alloca in zip(yield_op.operands, iter_allocas):
            new_body_block.insert_ops_before(
                [CcStoreOp(operand, alloca.result)], yield_op
            )

        # IV increment: next_iv = iv + step; store back.
        addi_cls = _get_arith_addi()
        next_iv = addi_cls(iv_load.result, step)
        iv_store = CcStoreOp(next_iv.result, iv_alloca.result)
        new_body_block.insert_ops_before([next_iv, iv_store], yield_op)

        new_body_block.insert_ops_before([CcContinueOp()], yield_op)
        yield_op.detach()

    # ---- 5. Create cc.loop ------------------------------------------------
    cc_loop = CcLoopOp(
        arguments=[],
        result_types=[],
        while_region=cond_region,
        body_region=body_region,
    )
    outer_block.insert_ops_before([cc_loop], for_op)

    # ---- 6. Final loads + replace results ---------------------------------
    _load_final_and_replace(
        outer_block, iter_allocas, _get_results(for_op), for_op
    )

    # ---- 7. Remove original -----------------------------------------------
    for_op.detach()


# ===================================================================
# scf.while  →  cc.loop
# ===================================================================

def _convert_scf_while(while_op, outer_block: Block) -> None:
    """Convert ``scf.while`` to ``cc.loop`` using alloca/store/load promotion.

    Handles **both** no-result and with-result ``scf.while`` ops uniformly.
    When the loop carries N values:

    * N ``cc.alloca`` + ``cc.store`` pairs are inserted before the loop.
    * The *before* (condition) region loads from the slots, computes the
      condition, stores the forwarded values, and emits ``cc.condition``.
    * The *after* (body) region loads from the slots, executes the body,
      stores the yielded values, and emits ``cc.continue``.
    * N ``cc.load`` ops after the loop replace the original SSA results.

    When N = 0 the helpers degenerate to no-ops and the result is equivalent
    to the previous no-result-only implementation.
    """
    # Recurse first.
    for region in while_op.regions:
        _process_region(region)

    init_args = list(while_op.operands)
    before_block = while_op.regions[0].blocks[0]
    after_block = while_op.regions[1].blocks[0]

    # ---- 1. Alloca + init -------------------------------------------------
    allocas = _alloca_and_init(outer_block, init_args, while_op)

    # ---- 2. Before (condition) region -------------------------------------
    #   load → original condition ops → store forwarded → cc.condition
    before_region, new_before_block, _ = _rewrite_block_with_loads(
        before_block, allocas
    )
    _replace_condition_with_stores(new_before_block, allocas)

    # ---- 3. After (body) region -------------------------------------------
    #   load → original body ops → store yields → cc.continue
    after_region, new_after_block, _ = _rewrite_block_with_loads(
        after_block, allocas
    )
    _replace_yield_with_stores_and_continue(new_after_block, allocas)

    # ---- 4. Create cc.loop ------------------------------------------------
    cc_loop = CcLoopOp(
        arguments=[],
        result_types=[],
        while_region=before_region,
        body_region=after_region,
    )
    outer_block.insert_ops_before([cc_loop], while_op)

    # ---- 5. Final loads + replace results ---------------------------------
    _load_final_and_replace(
        outer_block, allocas, _get_results(while_op), while_op
    )

    # ---- 6. Remove original -----------------------------------------------
    while_op.detach()