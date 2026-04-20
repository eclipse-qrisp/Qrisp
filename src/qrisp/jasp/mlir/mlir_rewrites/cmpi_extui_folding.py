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
Optimization passes for folding redundant scalar linalg.generic operations
and simplifying boolean condition chains in MLIR/xDSL.

Problem:
    The JAX-to-MLIR lowering sometimes produces verbose chains of 0-dimensional
    linalg.generic operations to evaluate simple boolean conditions, e.g.:

        measure (i1) → extui (i64) → cmpi eq 0 (i1) → extui (i32) → extract → cmpi ne 0

    This entire chain is semantically equivalent to a single NOT of the
    original measurement result.

Solution:
    Two composable rewrite patterns that, together with upstream canonicalize
    and DCE passes, collapse the chain:

    1. FoldScalarLinalgGeneric  — inlines 0-d linalg.generic into scalar ops
    2. FoldCmpiExtui            — folds cmpi(extui(i1), 0) into the original i1

Usage:
    >>> from xdsl.ir import Context
    >>> ctx = Context()
    >>> FoldRedundantMeasurementCondition().apply(ctx, module)

    Or as part of a pass pipeline:
        pass_pipeline = [
            FoldRedundantMeasurementCondition,
            Canonicalize,
            CommonSubexpressionElimination,
            DeadCodeElimination,
        ]
"""

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
    # Build the pattern set. Keep it small and focused so the greedy rewriter
    # converges quickly.
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

#====================================================================== #

class FoldCmpiExtui(RewritePattern):
    """
    Folds redundant comparisons of zero-extended booleans against 0 or 1.
    - cmpi ne, (extui %x), 0  ->  %x
    - cmpi eq, (extui %x), 1  ->  %x
    - cmpi eq, (extui %x), 0  ->  NOT %x
    - cmpi ne, (extui %x), 1  ->  NOT %x
    """
    
    _PRED_EQ = 0
    _PRED_NE = 1

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.CmpiOp, rewriter: PatternRewriter):
        # 1. Check if RHS is constant 0 or 1
        rhs_op = op.rhs.owner
        if not isinstance(rhs_op, arith.ConstantOp) or not isinstance(rhs_op.value, IntegerAttr):
            return
            
        rhs_val = rhs_op.value.value.data
        if rhs_val not in (0, 1):
            return
            
        # 2. Check if LHS is an extui originating from an i1
        lhs_op = op.lhs.owner
        if not isinstance(lhs_op, arith.ExtUIOp) or lhs_op.input.type != i1:
            return
            
        # 3. Determine if this reduces to `NOT x` or just `x`
        orig_bool = lhs_op.input
        pred = op.predicate.value.data
        
        # We need a NOT if checking `== 0` or `!= 1`
        needs_not = (pred == self._PRED_EQ and rhs_val == 0) or \
                    (pred == self._PRED_NE and rhs_val == 1)
        
        if needs_not:
            # Implement NOT via XOR with true
            true_const = arith.ConstantOp.from_int_and_width(1, 1)
            xor_op = arith.XOrIOp(orig_bool, true_const.result)
            
            rewriter.insert_op_before_matched_op(true_const)
            rewriter.replace_matched_op(xor_op)
        else:
            # Identity: replace with the original boolean
            rewriter.replace_matched_op([], [orig_bool])
