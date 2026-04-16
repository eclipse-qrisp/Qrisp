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

from xdsl.dialects import scf
from xdsl.dialects.builtin import i1
from xdsl.ir import Block, Region, SSAValue
from xdsl.rewriter import Rewriter

from qrisp.jasp.mlir.quake_lowering.cc_dialect import (
    CcConditionOp,
    CcContinueOp,
    CcIfOp,
    CcLoopOp,
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
# Recursive region/block/op processing
# ---------------------------------------------------------------------------


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
    elif op.name in ("scf.for", "scf.index_switch"):
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
