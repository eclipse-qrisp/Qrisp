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
Optimization pass for simplifying verbose boolean condition chains in MLIR/xDSL.

General Context:
    The JAX-to-MLIR lowering frequently produces verbose operation chains to
    evaluate simple scalar and boolean conditions. This pass is part of a suite
    of rewrite patterns designed to collapse unnecessarily verbose chains.

Specific Problem:
    Booleans (i1) are often zero-extended into wider integers (e.g., i32, i64)
    only to be immediately compared against 0 or 1 to yield another boolean.
    For example: `measure (i1) → extui (i64) → cmpi eq 0 (i1) → extui (i32) → cmpi ne 0`.
    This whole sequence is semantically equivalent to a simple boolean NOT 
    of the original measurement.

Solution:
    A rewrite pattern that statically resolves comparisons of zero-extended
    booleans against 0 or 1. It bypasses the integer extension and folds the 
    chain directly into the original boolean condition, its logical NOT, or a 
    constant True/False.
"""


from enum import Enum
from xdsl.context import Context
from xdsl.dialects import builtin, arith
from xdsl.dialects.builtin import IntegerAttr, i1
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    PatternRewriteWalker,
    op_type_rewrite_pattern,
    GreedyRewritePatternApplier,
)
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.dead_code_elimination import DeadCodeElimination


def cmpi_extui_folding(xdsl_ctx: Context, xdsl_module: builtin.ModuleOp) -> None:
    """
    Applies custom rewrite patterns to fold redundant cmpi(extui(i1), 0) chains into the original i1 condition.

    Parameters
    ----------
    xdsl_ctx: Context
        The xDSL context in which the module resides.
    xdsl_module: builtin.ModuleOp
        The xDSL module to be rewritten. The transformation is applied
        greedily and recursively over the whole module.
    """
    patterns = [FoldCmpiExtui()]

    # Apply patterns using a greedy rewriter over the entire module.
    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier(patterns),
        apply_recursively=True,
        walk_reverse=False,
    )

    walker.rewrite_module(xdsl_module)
    CanonicalizePass().apply(xdsl_ctx, xdsl_module)
    DeadCodeElimination().apply(xdsl_ctx, xdsl_module)


# ====================================================================== #


class FoldCmpiExtui(RewritePattern):
    """
    Folds all comparisons of zero-extended booleans against 0 or 1.
    Since extui(i1) ∈ {0, 1}, every predicate is statically resolvable.
    """

    class _Result(Enum):
        IDENTITY = "identity"  # result is %x
        NOT = "not"  # result is NOT %x
        TRUE = "true"  # always true
        FALSE = "false"  # always false

    # Lookup table: (predicate_int, rhs_const) -> _Result
    #
    #   eq=0, ne=1, slt=2, sle=3, sgt=4, sge=5, ult=6, ule=7, ugt=8, uge=9
    #
    _FOLD_TABLE: dict[tuple[int, int], _Result] = {
        # ---- vs 0 ----
        (0, 0): _Result.NOT,  # eq  0 → NOT %x
        (1, 0): _Result.IDENTITY,  # ne  0 →     %x
        (2, 0): _Result.FALSE,  # slt 0 → false
        (3, 0): _Result.NOT,  # sle 0 → NOT %x
        (4, 0): _Result.IDENTITY,  # sgt 0 →     %x
        (5, 0): _Result.TRUE,  # sge 0 → true
        (6, 0): _Result.FALSE,  # ult 0 → false
        (7, 0): _Result.NOT,  # ule 0 → NOT %x
        (8, 0): _Result.IDENTITY,  # ugt 0 →     %x
        (9, 0): _Result.TRUE,  # uge 0 → true
        # ---- vs 1 ----
        (0, 1): _Result.IDENTITY,  # eq  1 →     %x
        (1, 1): _Result.NOT,  # ne  1 → NOT %x
        (2, 1): _Result.NOT,  # slt 1 → NOT %x
        (3, 1): _Result.TRUE,  # sle 1 → true
        (4, 1): _Result.FALSE,  # sgt 1 → false
        (5, 1): _Result.IDENTITY,  # sge 1 →     %x
        (6, 1): _Result.NOT,  # ult 1 → NOT %x
        (7, 1): _Result.TRUE,  # ule 1 → true
        (8, 1): _Result.FALSE,  # ugt 1 → false
        (9, 1): _Result.IDENTITY,  # uge 1 →     %x
    }

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.CmpiOp, rewriter: PatternRewriter):
        # 1. RHS must be constant 0 or 1
        rhs_op = op.rhs.owner
        if not isinstance(rhs_op, arith.ConstantOp) or not isinstance(
            rhs_op.value, IntegerAttr
        ):
            return
        rhs_val = rhs_op.value.value.data
        if rhs_val not in (0, 1):
            return

        # 2. LHS must be extui from i1
        lhs_op = op.lhs.owner
        if not isinstance(lhs_op, arith.ExtUIOp) or lhs_op.input.type != i1:
            return

        # 3. Look up the fold result
        pred = op.predicate.value.data
        key = (pred, rhs_val)
        result = self._FOLD_TABLE.get(key)
        if result is None:
            return  # unknown predicate

        orig_bool = lhs_op.input

        if result == self._Result.IDENTITY:
            # Replace with the original i1 directly
            rewriter.replace_matched_op([], [orig_bool])

        elif result == self._Result.NOT:
            # NOT via XOR with 1
            true_const = arith.ConstantOp.from_int_and_width(1, 1)
            xor_op = arith.XOrIOp(orig_bool, true_const.result)
            rewriter.replace_matched_op([true_const, xor_op], [xor_op.result])

        elif result == self._Result.TRUE:
            # Replace with constant true (1 : i1)
            true_const = arith.ConstantOp.from_int_and_width(1, 1)
            rewriter.replace_matched_op(true_const)

        elif result == self._Result.FALSE:
            # Replace with constant false (0 : i1)
            false_const = arith.ConstantOp.from_int_and_width(0, 1)
            rewriter.replace_matched_op(false_const)
