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

"""Rewrite dynamic Quake vector allocations to static allocations."""

# Static-size quake.alloca rewriting.
# ===============================================
#
# Rewrites quake.alloca !quake.veq<?>[%n : i64] into quake.alloca
# !quake.veq<N> whenever %n is fed by an arith.constant.
#
# Uses of the (now statically-typed) result that are not guaranteed to
# accept either veq flavor interchangeably – i.e. anything other than the
# Quake ops in _TYPE_TRANSPARENT_CONSUMERS, such as control-flow block
# arguments (cc.loop/cc.if, or their pre-lowering scf counterparts) or
# function boundaries (func.call/func.return), whose types were already
# independently fixed to the dynamic !quake.veq<?> earlier in the
# pipeline – are routed through an inserted quake.relax_size cast back to
# !quake.veq<?>, so the alloca itself can always be staticized regardless
# of how its result is later used.
#
# This runs after scalar_tensor_unwrap, which is
# responsible for folding the tensor.extract/tensor.from_elements
# round-trips that Jasp emits for register sizes down to a bare
# arith.constant. Only once that folding has happened can this pass
# reliably recognize a constant-sized allocation.

from xdsl.context import Context
from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerAttr, ModuleOp
from xdsl.ir import Use
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from qrisp.jasp.cudaq_interface.quake_lowering.dialects.quake_dialect import (
    AllocaOp,
    ConcatOp,
    DeallocOp,
    ExtractRefOp,
    MzOp,
    QuakeVeqType,
    RelaxSizeOp,
    ResetOp,
    SubVeqOp,
    VeqSizeOp,
)

# Ops whose printed operand type is always derived dynamically from the
# operand's actual ``.type`` (see their ``print`` methods in quake_dialect.py),
# rather than from a separately-declared, possibly-stale type (as is the case
# for e.g. scf/cc control-flow block arguments, function signatures, or
# func.call/func.return operands). Uses by these ops can keep referencing the
# staticized alloca result directly; all other uses are routed through a
# ``quake.relax_size`` cast (see :func:`staticize_veq_alloca`).
_TYPE_TRANSPARENT_CONSUMERS = (
    ExtractRefOp,
    DeallocOp,
    VeqSizeOp,
    SubVeqOp,
    ConcatOp,
    MzOp,
    ResetOp,
)


# ===================================================================
# Public entry point
# ===================================================================


def _staticize_veq_alloca(module: ModuleOp) -> None:
    """Applies the StaticVeqAllocaPass to the given module."""
    StaticVeqAllocaPass().apply(Context(), module)


# ------------------------------------------------------------------ #
# Pattern: quake.alloca !quake.veq<?>[%c] -> quake.alloca !quake.veq<N>
# ------------------------------------------------------------------ #
class StaticizeAllocaSize(RewritePattern):
    """Replaces a dynamically-sized veq alloca fed by a constant with a

    statically-sized one, inserting a ``quake.relax_size`` cast for any uses
    that require the dynamic ``!quake.veq<?>`` type (see module docstring).
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AllocaOp, rewriter: PatternRewriter) -> None:
        """Replace constant-sized dynamic allocations with static allocations."""
        size_operands = list(op.size)
        if len(size_operands) != 1:
            return  # single-qubit alloca, nothing to staticize

        veq_type = op.result.type
        if not isinstance(veq_type, QuakeVeqType) or veq_type.is_static:
            return  # already static (or unexpected result type)

        owner = size_operands[0].owner
        if not isinstance(owner, arith.ConstantOp):
            return

        value = owner.value
        if not isinstance(value, IntegerAttr):
            return

        n = value.value.data
        if n < 0:
            return

        uses_needing_relax: list[Use] = [
            use for use in op.result.uses if not isinstance(use.operation, _TYPE_TRANSPARENT_CONSUMERS)
        ]

        new_alloca = AllocaOp(n)
        if uses_needing_relax:
            relax_op = RelaxSizeOp(new_alloca.result)
            rewriter.replace_matched_op([new_alloca, relax_op], [new_alloca.results[0]])
            for use in uses_needing_relax:
                use.operation.operands[use.index] = relax_op.result
        else:
            rewriter.replace_matched_op(new_alloca)


# ------------------------------------------------------------------ #
# Pass
# ------------------------------------------------------------------ #
class StaticVeqAllocaPass(ModulePass):
    """xDSL ``ModulePass`` wrapper around :func:`staticize_veq_alloca`."""

    name = "static-veq-alloca"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        """Apply static register allocation rewriting to the module."""
        PatternRewriteWalker(
            GreedyRewritePatternApplier([StaticizeAllocaSize()]),
            apply_recursively=False,
        ).rewrite_module(op)
