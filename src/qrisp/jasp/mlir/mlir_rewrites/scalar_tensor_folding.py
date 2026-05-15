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
    Three composable rewrite patterns that, together with upstream canonicalize
    and DCE passes, collapse the chain.
"""

from xdsl.context import Context
from xdsl.dialects import builtin, tensor
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    op_type_rewrite_pattern,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.dead_code_elimination import DeadCodeElimination


def scalar_tensor_folding(xdsl_ctx: Context, xdsl_module: builtin.ModuleOp) -> None:
    """
    Applies custom rewrite patterns to fold trivial 0-dimensional tensor operations
    into scalar operations, and eliminates dead tensor ops.

    Parameters
    ----------
    xdsl_ctx: Context
        The xDSL context in which the module resides.
    xdsl_module: builtin.ModuleOp
        The xDSL module to be rewritten. The transformation is applied
        greedily and recursively over the whole module.
    """
    patterns = [FoldExtractFromElements(), EraseDeadFromElements()]

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

class FoldExtractFromElements(RewritePattern):
    """Folds a tensor.extract from a 0-D tensor created by tensor.from_elements back into the original scalar value."""
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tensor.ExtractOp, rewriter: PatternRewriter) -> None:
        # Only handle 0-D case (no index operands)
        if len(op.indices) != 0:
            return

        defining_op = op.tensor.owner
        if not isinstance(defining_op, tensor.FromElementsOp):
            return

        # A scalar (0-D) from_elements has exactly one element
        if len(defining_op.elements) != 1:
            return

        # Replace extract result with the original scalar; erase extract op
        rewriter.replace_matched_op([], [defining_op.elements[0]])


class EraseDeadFromElements(RewritePattern):
    """
    Eliminates dead `tensor.from_elements` operations that have no active uses.

    Why is this necessary?
    In MLIR and xDSL, the standard `DeadCodeElimination` (DCE) pass will only 
    remove an unused operation if it possesses the `Pure` trait. This trait 
    guarantees to the compiler that the operation has no side effects (such as 
    memory allocation, I/O, or halting).

    If the xDSL implementation of `tensor.from_elements` lacks the `Pure` trait, 
    the standard DCE pass must conservatively assume that executing the operation 
    might produce side effects. Consequently, it will refuse to delete it, even 
    if its resulting tensor is completely unused.

    This rewrite pattern acts as a targeted DCE fallback. It explicitly checks 
    if the result of the `tensor.from_elements` operation has any active uses 
    in the IR; if the use count is zero, it safely erases the operation.
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tensor.FromElementsOp, rewriter: PatternRewriter) -> None:
        # If the result has zero uses, we are safe to delete it
        if not op.result.uses:
            rewriter.erase_op(op)
