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
from xdsl.dialects import builtin, linalg, tensor
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    op_type_rewrite_pattern,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)


def scalar_linalg_folding(xdsl_ctx: Context, xdsl_module: builtin.ModuleOp) -> None:
    """
    Applies custom rewrite patterns to fold a 0-dimensional `linalg.generic` with a single operation into
    scalar arithmetic wrapped in `tensor.extract` / `tensor.from_elements`.

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
    patterns = [FoldScalarLinalgGeneric()]

    # Apply patterns using a greedy rewriter over the entire module.
    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier(patterns),
        apply_recursively=True,
        walk_reverse=False,
    )

    walker.rewrite_module(xdsl_module)

#====================================================================== #

class FoldScalarLinalgGeneric(RewritePattern):
    """Fold a 0-dimensional `linalg.generic` with a single operation into
    scalar arithmetic wrapped in `tensor.extract` / `tensor.from_elements`.

    0-dimensional `linalg.generic` ops operate on scalar tensors (tensor<T>
    with no dimensions). They carry significant overhead — region, block
    arguments, affine maps — for what is semantically a single scalar
    instruction. This pattern eliminates that overhead.

    Preconditions (all must hold for the pattern to fire):
        - The generic has zero iterator types (i.e., all operands are 0-d).
        - The body contains exactly one computation op followed by a yield.

    Example:
        Before:
            %out = tensor.empty() : tensor<i64>
            %result = linalg.generic {
                indexing_maps = [affine_map<() -> ()>,
                                 affine_map<() -> ()>],
                iterator_types = []
            } ins(%in : tensor<i1>) outs(%out : tensor<i64>) {
            ^bb0(%arg0 : i1, %arg1 : i64):
                %v = arith.extui %arg0 : i1 to i64
                linalg.yield %v : i64
            } -> tensor<i64>

        After:
            %scalar = tensor.extract %in[] : tensor<i1>
            %v      = arith.extui %scalar : i1 to i64
            %result = tensor.from_elements %v : tensor<i64>
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.GenericOp, rewriter: PatternRewriter
    ) -> None:
        # ── Guard: only 0-d generics (no iteration dimensions) ──────────
        if len(op.iterator_types.data) != 0:
            return

        # ── Guard: body must be exactly <one_op> + <yield> ──────────────
        body = op.body.block
        if len(body.ops) != 2:
            return

        inner_op = body.first_op  # the single computation op
        # body.last_op is the linalg.yield (unused but implied)

        # ── Step 1: Extract scalars from each 0-d tensor input ──────────
        # For every tensor<T> input, insert a `tensor.extract %input[]`
        # to obtain the bare scalar value.
        scalar_inputs: list = []
        for inp in op.inputs:
            # 1. Get the element type of the input tensor (e.g., i1 or i64)
            # inp.type is typically a TensorType or RankedTensorType
            res_type = inp.type.element_type
            extract = tensor.ExtractOp(inp, [], res_type)
            rewriter.insert_op_before_matched_op(extract)
            scalar_inputs.append(extract.result)

        # ── Step 2: Clone the inner op with scalar operands ─────────────
        # The inner op originally uses block arguments as operands.
        # We remap them to the freshly extracted scalars.
        mapping: dict = {}
        for block_arg, scalar in zip(body.args, scalar_inputs):
            mapping[block_arg] = scalar

        cloned = inner_op.clone()
        for i, operand in enumerate(cloned.operands):
            if operand in mapping:
                cloned.operands[i] = mapping[operand]
        rewriter.insert_op_before_matched_op(cloned)

        # ── Step 3: Wrap the scalar result back into a 0-d tensor ───────
        from_elements = tensor.FromElementsOp(
            operands=[cloned.results[0]], 
            result_types=[op.res[0].type]
        )
        #from_elements = tensor.FromElementsOp.get(
        #    cloned.results[0], op.res[0].type
        #)
        rewriter.replace_matched_op(from_elements)
