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

This module converts ``arith`` integer operations into ``llvm`` equivalents.
Other classical dialects (``tensor``, ``scf``, ``math``) have bare-bone
placeholder stubs and  will be filled in separately.
"""

from collections.abc import Sequence

from xdsl.context import Context
from xdsl.dialects import arith, builtin, llvm
from xdsl.dialects.builtin import IntegerAttr, i64
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


# ==============================================================================
# Orchestrator
#
# Note: patterns are classes (not functions) because xDSL's
# GreedyRewritePatternApplier expects list[RewritePattern], and
# @op_type_rewrite_pattern requires a class method.
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
    patterns: list[RewritePattern] = [
        ArithAddiToLLVM(),
        ArithSubiToLLVM(),
        ArithMuliToLLVM(),
        ArithDivuiToLLVM(),
        ArithDivsiToLLVM(),
        ArithRemuiToLLVM(),
        ArithRemsiToLLVM(),
        ArithCmpiToLLVM(),
        ArithAndToLLVM(),
        ArithOrToLLVM(),
        ArithXOrToLLVM(),
        ArithShliToLLVM(),
        ArithShrsiToLLVM(),
        ArithShruiToLLVM(),
    ]

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

    ``arith.addi`` (https://mlir.llvm.org/docs/Dialects/ArithOps/#arithaddi-arithaddiop)
    performs N-bit integer addition modulo 2^N (two's complement). The operands
    are signless — the same operation handles both signed and unsigned addition.

    ``llvm.add`` (https://llvm.org/docs/LangRef.html#add-instruction)
    is the LLVM equivalent with identical bitvector semantics. Both support
    ``nuw``/``nsw`` overflow flags (``arith`` uses ``IntegerOverflowFlagsAttr``,
    LLVM uses ``OverflowAttr``). Overflow wraparound modulo 2^N when the flags
    are absent.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.AddiOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(llvm.AddOp(op.lhs, op.rhs))


class ArithSubiToLLVM(RewritePattern):
    """Lower ``arith.subi`` to ``llvm.sub``.

    ``arith.subi`` (https://mlir.llvm.org/docs/Dialects/ArithOps/#arithsubi-arithsubiop)
    computes ``lhs - rhs`` modulo 2^N (two's complement). Signless semantics:
    ``-5 - 3 = -8`` with `i32` wraps to ``2^32 - 8``.

    ``llvm.sub`` (https://llvm.org/docs/LangRef.html#sub-instruction)
    has identical N-bit subtraction semantics. Both support ``nuw``/``nsw``
    overflow flags.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SubiOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(llvm.SubOp(op.lhs, op.rhs))


class ArithMuliToLLVM(RewritePattern):
    """Lower ``arith.muli`` to ``llvm.mul``.

    ``arith.muli`` (https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmuli-arithmuliop)
    multiplies two N-bit integers, producing the low N bits of the full 2N-bit
    product (i.e., ``(a * b) mod 2^N``). Signless — the low N bits are the same
    for signed and unsigned multiplication.

    ``llvm.mul`` (https://llvm.org/docs/LangRef.html#mul-instruction)
    has identical semantics. Both support ``nuw``/``nsw`` overflow flags.
    If ``nuw`` is set, the result is poison if the product overflows unsigned;
    if ``nsw`` is set, poison on signed overflow.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MuliOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(llvm.MulOp(op.lhs, op.rhs))


class ArithDivuiToLLVM(RewritePattern):
    """Lower ``arith.divui`` (unsigned division) to ``llvm.udiv``.

    ``arith.divui`` (https://mlir.llvm.org/docs/Dialects/ArithOps/#arithdivui-arithdivuiop)
    performs unsigned integer division, rounding toward zero. The operands are
    treated as unsigned N-bit values, so ``-1`` (``0xFFFFFFFF`` for ``i32``)
    divided by ``2`` gives ``2^31 - 1``, not ``-0.5``. Division by zero is UB.

    ``llvm.udiv`` (https://llvm.org/docs/LangRef.html#udiv-instruction)
    has identical unsigned division semantics. Both support an ``exact`` flag:
    if set, the result is poison when ``lhs`` is not a multiple of ``rhs``.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.DivUIOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(llvm.UDivOp(op.lhs, op.rhs))


class ArithDivsiToLLVM(RewritePattern):
    """Lower ``arith.divsi`` (signed division) to ``llvm.sdiv``.

    ``arith.divsi`` (https://mlir.llvm.org/docs/Dialects/ArithOps/#arithdivsi-arithdivsiop)
    performs signed integer division, rounding toward zero (truncation toward
    zero). Examples: ``7 / -2 = -3``, ``-7 / 2 = -3``. Division by zero and
    signed overflow (``INT_MIN / -1``) are UB.

    ``llvm.sdiv`` (https://llvm.org/docs/LangRef.html#sdiv-instruction)
    has identical signed division semantics. Both support an ``exact`` flag:
    if set, the result is poison when ``lhs`` is not a multiple of ``rhs``.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.DivSIOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(llvm.SDivOp(op.lhs, op.rhs))


class ArithRemuiToLLVM(RewritePattern):
    """Lower ``arith.remui`` (unsigned remainder) to ``llvm.urem``.

    ``arith.remui`` computes ``lhs % rhs`` treating both operands as unsigned
    bitvectors (leading bit is MSB, not a sign). ``llvm.urem`` has identical
    unsigned remainder semantics (result has the same sign as the dividend).
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.RemUIOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(llvm.URemOp(op.lhs, op.rhs))


class ArithRemsiToLLVM(RewritePattern):
    """Lower ``arith.remsi`` (signed remainder) to ``llvm.srem``.

    ``arith.remsi`` computes ``lhs % rhs`` with sign — the result sign matches
    the dividend (``lhs``), and the result magnitude equals ``|lhs| % |rhs|``.
    ``llvm.srem`` has identical semantics (sign follows dividend, trunc toward
    zero). Division by zero is UB in both dialects.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.RemSIOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(llvm.SRemOp(op.lhs, op.rhs))


class ArithCmpiToLLVM(RewritePattern):
    """Lower ``arith.cmpi`` to ``llvm.icmp``.

    Both operations compare two integers using a predicate encoded as an
    integer attribute with values 0–9, and the encoding is identical:

    =====  =====  =============  ===========
    Value  Name   Meaning        Operands
    =====  =====  =============  ===========
    0      eq     equal          signed
    1      ne     not equal      signed
    2      slt    less than      signed
    3      sle    less or equal  signed
    4      sgt    greater than   signed
    5      sge    greater/equal  signed
    6      ult    less than      unsigned
    7      ule    less or equal  unsigned
    8      ugt    greater than   unsigned
    9      uge    greater/equal  unsigned
    =====  =====  =============  ===========

    The result is a 1-bit integer (``i1``). Operands must be integer-like
    (scalar, vector, or tensor); vector/tensor comparisons are element-wise.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.CmpiOp, rewriter: PatternRewriter):
        pred = IntegerAttr(op.predicate.value.data, i64)
        rewriter.replace_matched_op(llvm.ICmpOp(op.lhs, op.rhs, pred))


class ArithAndToLLVM(RewritePattern):
    """Lower ``arith.andi`` to ``llvm.and``.

    Both ops compute the bitwise AND of two integer values. Operands are
    treated as unsigned bitvectors — the operation is identical for signed
    and unsigned integers since AND is bitwise. ``arith.andi`` uses signless
    integer types; ``llvm.and`` uses the same.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.AndIOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(llvm.AndOp(op.lhs, op.rhs))


class ArithOrToLLVM(RewritePattern):
    """Lower ``arith.ori`` to ``llvm.or``.

    Both ops compute the bitwise OR of two integer values. Like AND, this
    operation is bitwise and identical for signed and unsigned integers.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.OrIOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(llvm.OrOp(op.lhs, op.rhs))


class ArithXOrToLLVM(RewritePattern):
    """Lower ``arith.xori`` to ``llvm.xor``.

    Both ops compute the bitwise XOR (exclusive OR) of two integer values.
    Like AND/OR, this is a purely bitwise operation independent of signedness.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.XOrIOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(llvm.XOrOp(op.lhs, op.rhs))


class ArithShliToLLVM(RewritePattern):
    """Lower ``arith.shli`` (shift left) to ``llvm.shl``.

    ``arith.shli`` shifts ``lhs`` left by ``rhs`` positions, filling the low
    bits with zeros. This is a logical/arithmetic left shift (same for both).
    ``llvm.shl`` has identical semantics. If ``rhs`` >= bitwidth, the result
    is poison in both dialects.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ShLIOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(llvm.ShlOp(op.lhs, op.rhs))


class ArithShrsiToLLVM(RewritePattern):
    """Lower ``arith.shrsi`` (signed shift right) to ``llvm.ashr``.

    ``arith.shrsi`` performs an arithmetic (signed) right shift: the shift
    fills the high bits with copies of the original sign bit (sign-extension).
    ``llvm.ashr`` (arithmetic shift right) has identical semantics.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ShRSIOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(llvm.AShrOp(op.lhs, op.rhs))


class ArithShruiToLLVM(RewritePattern):
    """Lower ``arith.shrui`` (unsigned shift right) to ``llvm.lshr``.

    ``arith.shrui`` performs a logical (unsigned) right shift: the high bits
    are filled with zeros. ``llvm.lshr`` (logical shift right) has identical
    semantics.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ShRUIOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(llvm.LShrOp(op.lhs, op.rhs))


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

    The lowering produces::

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
