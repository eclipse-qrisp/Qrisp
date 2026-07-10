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
Classical-to-LLVM lowering for the JASP MLIR pipeline.

This module converts all non-quantum MLIR dialects (``arith``, ``tensor``,
``scf``, ``math``) into the ``llvm`` dialect, leaving only ``jasp.*``
(quantum) and ``llvm.*`` (classical) ops in the module.

Each dialect group has its own ``RewritePattern`` subclass(es), registered in
the ``classical_to_llvm`` orchestrator.
"""

from collections.abc import Sequence

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    RewritePattern,
)


# ==============================================================================
# Orchestrator
# ==============================================================================


def classical_to_llvm(xdsl_ctx: Context, xdsl_module: builtin.ModuleOp) -> None:
    """Lower all classical MLIR dialects to the LLVM dialect in-place.

    Applies rewrite patterns for ``arith``, ``tensor``, ``scf``, and ``math``
    dialects, replacing each operation with its ``llvm.*`` equivalent.
    ``jasp.*`` operations are left untouched.

    Parameters
    ----------
    xdsl_ctx : Context
        The xDSL context in which the module resides. Must have the LLVM
        dialect loaded.
    xdsl_module : builtin.ModuleOp
        The xDSL module to rewrite. Modified in-place.
    """
    # TODO: populate with all pattern instances
    patterns: list[RewritePattern] = []

    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier(patterns),
        apply_recursively=True,
        walk_reverse=False,
    )
    walker.rewrite_module(xdsl_module)


# ==============================================================================
# arith → llvm  —  integer arithmetic
# ==============================================================================


class ArithAddiToLLVM(RewritePattern):
    """Lower ``arith.addi`` to ``llvm.add``.

    Matches ``arith.AddiOp`` and replaces it with ``llvm.AddOp`` using the
    same operands and result type.
    """


class ArithSubiToLLVM(RewritePattern):
    """Lower ``arith.subi`` to ``llvm.sub``.

    Matches ``arith.SubiOp`` and replaces it with ``llvm.SubOp``.
    """


class ArithMuliToLLVM(RewritePattern):
    """Lower ``arith.muli`` to ``llvm.mul``.

    Matches ``arith.MuliOp`` and replaces it with ``llvm.MulOp``.
    """


class ArithDivuiToLLVM(RewritePattern):
    """Lower ``arith.divui`` (unsigned division) to ``llvm.udiv``.

    Matches ``arith.DivuiOp`` and replaces it with ``llvm.UDivOp``.
    """


class ArithDivsiToLLVM(RewritePattern):
    """Lower ``arith.divsi`` (signed division) to ``llvm.sdiv``.

    Matches ``arith.DivsiOp`` and replaces it with ``llvm.SDivOp``.
    """


class ArithRemuiToLLVM(RewritePattern):
    """Lower ``arith.remui`` (unsigned remainder) to ``llvm.urem``.

    Matches ``arith.RemuiOp`` and replaces it with ``llvm.URemOp``.
    """


class ArithRemsiToLLVM(RewritePattern):
    """Lower ``arith.remsi`` (signed remainder) to ``llvm.srem``.

    Matches ``arith.RemsiOp`` and replaces it with ``llvm.SRemOp``.
    """


class ArithCmpiToLLVM(RewritePattern):
    """Lower ``arith.cmpi`` to ``llvm.icmp``.

    Matches ``arith.CmpiOp`` and replaces it with ``llvm.ICmpOp``.
    The integer predicate values (0-9) use the same encoding in both
    dialects, so the predicate attribute is forwarded as-is.
    """


class ArithAndToLLVM(RewritePattern):
    """Lower ``arith.andi`` to ``llvm.and``.

    Matches ``arith.AndOp`` and replaces it with ``llvm.AndOp``.
    """


class ArithOrToLLVM(RewritePattern):
    """Lower ``arith.ori`` to ``llvm.or``.

    Matches ``arith.OrOp`` and replaces it with ``llvm.OrOp``.
    """


class ArithXOrToLLVM(RewritePattern):
    """Lower ``arith.xori`` to ``llvm.xor``.

    Matches ``arith.XOrOp`` and replaces it with ``llvm.XOrOp``.
    """


class ArithShliToLLVM(RewritePattern):
    """Lower ``arith.shli`` (shift left) to ``llvm.shl``.

    Matches ``arith.ShliOp`` and replaces it with ``llvm.ShlOp``.
    """


class ArithShrsiToLLVM(RewritePattern):
    """Lower ``arith.shrsi`` (arithmetic/signed shift right) to ``llvm.ashr``.

    Matches ``arith.ShrsiOp`` and replaces it with ``llvm.AShrOp``.
    """


class ArithShruiToLLVM(RewritePattern):
    """Lower ``arith.shrui`` (logical/unsigned shift right) to ``llvm.lshr``.

    Matches ``arith.ShruiOp`` and replaces it with ``llvm.LShrOp``.
    """


# ==============================================================================
# arith → llvm  —  floating-point arithmetic
# ==============================================================================


class ArithAddfToLLVM(RewritePattern):
    """Lower ``arith.addf`` to ``llvm.fadd``.

    Matches ``arith.AddfOp`` and replaces it with ``llvm.FAddOp``.
    """


class ArithSubfToLLVM(RewritePattern):
    """Lower ``arith.subf`` to ``llvm.fsub``.

    Matches ``arith.SubfOp`` and replaces it with ``llvm.FSubOp``.
    """


class ArithMulfToLLVM(RewritePattern):
    """Lower ``arith.mulf`` to ``llvm.fmul``.

    Matches ``arith.MulfOp`` and replaces it with ``llvm.FMulOp``.
    """


class ArithDivfToLLVM(RewritePattern):
    """Lower ``arith.divf`` to ``llvm.fdiv``.

    Matches ``arith.DivfOp`` and replaces it with ``llvm.FDivOp``.
    """


class ArithNegfToLLVM(RewritePattern):
    """Lower ``arith.negf`` to ``llvm.fneg``.

    Matches ``arith.NegfOp`` and replaces it with ``llvm.FNegOp``.
    """


# ==============================================================================
# arith → llvm  —  type casts
# ==============================================================================


class ArithExtSIToLLVM(RewritePattern):
    """Lower ``arith.extsi`` (sign-extending integer cast) to ``llvm.sext``.

    Matches ``arith.ExtSIOp`` and replaces it with ``llvm.SExtOp``.
    """


class ArithExtUIToLLVM(RewritePattern):
    """Lower ``arith.extui`` (zero-extending integer cast) to ``llvm.zext``.

    Matches ``arith.ExtUIOp`` and replaces it with ``llvm.ZExtOp``.
    """


class ArithTruncIToLLVM(RewritePattern):
    """Lower ``arith.trunci`` (integer truncation) to ``llvm.trunc``.

    Matches ``arith.TruncIOp`` and replaces it with ``llvm.TruncOp``.
    """


class ArithIndexCastToLLVM(RewritePattern):
    """Lower ``arith.index_cast`` to ``llvm.inttoptr`` or a direct cast.

    Matches ``arith.IndexCastOp``. Since the LLVM dialect does not have an
    ``index`` type, this cast converts to/from ``i64``. If the source is
    ``index``, cast from ``index`` → ``i64`` → target type. If the result is
    ``index``, cast from source type → ``i64`` → ``index`` (represented as
    ``i64`` in LLVM).
    """


# ==============================================================================
# arith → llvm  —  select / constants
# ==============================================================================


class ArithSelectToLLVM(RewritePattern):
    """Lower ``arith.select`` to ``llvm.select``.

    Matches ``arith.SelectOp`` and replaces it with ``llvm.SelectOp`` with
    the same condition, true value, and false value.
    """


class ArithConstantToLLVM(RewritePattern):
    """Lower ``arith.constant`` to ``llvm.mlir.constant``.

    Matches ``arith.ConstantOp`` and replaces it with ``llvm.ConstantOp``.
    The value attribute (integer or float) is forwarded as-is; the result
    type is forwarded to the LLVM constant op.
    """


# ==============================================================================
# tensor → llvm
# ==============================================================================


class TensorExtractToLLVM(RewritePattern):
    """Lower ``tensor.extract`` (0-d tensor scalar extraction) to ``llvm.extractvalue``.

    Matches zero-dimensional ``tensor.ExtractOp`` (i.e., extracting the scalar
    from a rank-0 tensor). Replaces it with ``llvm.ExtractValueOp`` on the
    tensor value. Higher-rank tensor extracts are not handled here and will
    need a separate pass or a more general lowering.
    """


class TensorFromElementsToLLVM(RewritePattern):
    """Lower ``tensor.from_elements`` to ``llvm.undef`` + ``llvm.insertvalue``.

    Matches ``tensor.FromElementsOp``. Creates an ``llvm.undef`` value of the
    result type, then inserts each element at its position using
    ``llvm.insertvalue``. This only handles the scalar-wrapping case (0-d
    tensor from one scalar element) and single-element 1-d tensors.
    """


class TensorEmptyToLLVM(RewritePattern):
    """Lower ``tensor.empty`` to ``llvm.undef`` (poison value).

    Matches ``tensor.EmptyOp`` and replaces it with ``llvm.UndefOp`` of the
    corresponding LLVM type. Since ``tensor.empty`` creates an uninitialized
    tensor, ``llvm.undef`` (or poison) is the closest semantic equivalent.
    """


# ==============================================================================
# scf → llvm  —  structured control flow to CFG
# ==============================================================================


class ScfIfToLLVM(RewritePattern):
    """Lower ``scf.if`` to unstructured LLVM control-flow graph (CFG).

    This is the most complex rewrite in the pass. ``scf.if %cond -> (T)`` is
    converted to a conditional branch followed by a merge block with phi nodes.

    Before::

        %result = scf.if %cond -> (i32) {
            %v = arith.addi %a, %b : i32
            scf.yield %v : i32
        } else {
            %w = arith.subi %a, %b : i32
            scf.yield %w : i32
        }

    After (unstructured CFG)::

        ^bb0:
          cond_br %cond, ^then, ^else

        ^then:
          %v = llvm.add %a, %b : i32
          br ^merge(%v)

        ^else:
          %w = llvm.sub %a, %b : i32
          br ^merge(%w)

        ^merge(%phi):
          ...  (use %phi instead of %result)

    Quantum types (``!jasp.QuantumState``) may appear as block arguments
    carried through the ``scf.if`` — these must be forwarded through the
    branch operands unchanged.

    .. note::

        xDSL v0.59.0 does not have ``llvm.br`` (unconditional branch).
        Use ``cf.br`` from the ``cf`` dialect, or work around it with
        ``llvm.cond_br`` using a constant ``true`` condition.
    """


class ScfWhileToLLVM(RewritePattern):
    """Lower ``scf.while`` to unstructured LLVM CFG (loop).

    ``scf.while`` has two regions: a ``before`` region (computes the
    condition) and an ``after`` region (computes the next iteration values).

    The lowering produces:

        ^entry:
          br ^cond(carried_values...)

        ^cond(%args...):
          ... compute condition ...
          cond_br %condition, ^body(%args...), ^done(%args...)

        ^body(%args...):
          ... compute next iteration ...
          br ^cond(next_args...)

        ^done(%results...):
          ...  (use %results instead of the while op's results)

    Quantum types carried through the loop must be forwarded unchanged.
    """


# ==============================================================================
# math → llvm  —  math intrinsics
# ==============================================================================


class MathSqrtToLLVM(RewritePattern):
    """Lower ``math.sqrt`` to ``llvm.intr.sqrt``.

    Matches ``math.SqrtOp`` and replaces it with an LLVM intrinsic call
    (``llvm.intr.sqrt``) or ``llvm.FSqrtOp`` if available in the dialect.
    """


class MathAbsIToLLVM(RewritePattern):
    """Lower ``math.absi`` to ``llvm.abs``.

    Matches ``math.AbsIOp`` and replaces it with ``llvm.AbsOp`` (integer
    absolute value).
    """


class MathAbsfToLLVM(RewritePattern):
    """Lower ``math.absf`` to ``llvm.fabs``.

    Matches ``math.AbsFOp`` and replaces it with ``llvm.FAbsOp`` (floating-
    point absolute value).
    """


class MathSinToLLVM(RewritePattern):
    """Lower ``math.sin`` to an LLVM intrinsic or libm call.

    Matches ``math.SinOp`` and replaces it with ``llvm.intr.sin`` or a
    call to the ``sin`` libm function.
    """


class MathCosToLLVM(RewritePattern):
    """Lower ``math.cos`` to an LLVM intrinsic or libm call.

    Matches ``math.CosOp`` and replaces it with ``llvm.intr.cos`` or a
    call to the ``cos`` libm function.
    """


class MathExpToLLVM(RewritePattern):
    """Lower ``math.exp`` to an LLVM intrinsic or libm call.

    Matches ``math.ExpOp`` and replaces it with ``llvm.intr.exp``.
    """


class MathLogToLLVM(RewritePattern):
    """Lower ``math.log`` to an LLVM intrinsic or libm call.

    Matches ``math.LogOp`` and replaces it with ``llvm.intr.log``.
    """
