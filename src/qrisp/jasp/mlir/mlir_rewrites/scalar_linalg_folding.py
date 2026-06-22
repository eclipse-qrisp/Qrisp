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
Optimization pass for unwrapping 0-dimensional linalg.generic operations
in MLIR/xDSL.

General Context:
    The JAX-to-MLIR lowering frequently produces verbose operation chains to
    evaluate simple scalar and boolean conditions. This pass is part of a suite
    of rewrite patterns designed to collapse unnecessarily verbose chains.

Specific Problem:
    Simple scalar operations are sometimes unnecessarily wrapped in 0-dimensional 
    `linalg.generic` ops. These ops carry significant structural overhead - such 
    as regions, block arguments, and affine maps - for what is semantically just 
    scalar instructions.

Solution:
    A rewrite pattern that eliminates the `linalg.generic` shell. It extracts
    the scalar operands using `tensor.extract`, clones the generic's inner block
    operations sequentially, and repacks the final result with `tensor.from_elements`.
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
from xdsl.rewriter import InsertPoint


def scalar_linalg_folding(xdsl_ctx: Context, xdsl_module: builtin.ModuleOp) -> None:
    """
    Applies custom rewrite patterns to fold 0-dimensional `linalg.generic` operations into
    scalar arithmetic wrapped in `tensor.extract` / `tensor.from_elements`.

    Parameters
    ----------
    xdsl_ctx: Context
        The xDSL context in which the module resides.
    xdsl_module: builtin.ModuleOp
        The xDSL module to be rewritten. The transformation is applied
        greedily and recursively over the whole module.
    """
    patterns = [FoldScalarLinalgGeneric()]

    # Apply patterns using a greedy rewriter over the entire module.
    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier(patterns),
        apply_recursively=True,
        walk_reverse=False,
    )

    walker.rewrite_module(xdsl_module)


# ====================================================================== #


class FoldScalarLinalgGeneric(RewritePattern):
    """Fold 0-dimensional `linalg.generic` operations into
    scalar arithmetic wrapped in `tensor.extract` / `tensor.from_elements`.

    This pattern extracts the scalar operands, sequentially clones all
    intermediate operations within the generic block, and repacks the
    final yielded scalar into a 0-dimensional tensor.

    0-dimensional `linalg.generic` ops operate on scalar tensors (tensor<T>
    with no dimensions). They carry significant overhead — region, block
    arguments, affine maps — for what is semantically scalar
    instructions. This pattern eliminates that overhead.

    Preconditions (all must hold for the pattern to fire):
        - The generic has zero iterator types (i.e., all operands are 0-d).
        - The generic produces exactly one output tensor.

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
        # Guard: only 0-d generics
        if len(op.iterator_types.data) != 0:
            return

        # Guard: single output only
        if len(op.outputs) != 1:
            return

        body = op.body.block
        yield_op = body.last_op  # The last op is always the yield

        # Step 1: Extract scalars from ALL tensor operands (ins + outs)
        all_tensors = list(op.inputs) + list(op.outputs)
        mapping: dict = {}

        for block_arg, tensor_val in zip(body.args, all_tensors):
            if len(list(block_arg.uses)) == 0:
                continue  # Skip extraction

            extract = tensor.ExtractOp(tensor_val, [], tensor_val.type.element_type)
            rewriter.insert_op(extract, InsertPoint.before(op))
            mapping[block_arg] = extract.results[0]

        # Step 2: Clone the body sequentially
        # Convert BlockOps to a standard Python list so we can slice it
        ops_list = list(body.ops)

        # Iterate over all ops EXCEPT the yield
        for op_in_body in ops_list[:-1]:
            cloned = op_in_body.clone()

            # Safely remap operands based on our running dictionary
            new_operands = [
                mapping.get(operand, operand) for operand in cloned.operands
            ]
            cloned.operands = tuple(new_operands)  # Reassign safely

            rewriter.insert_op(cloned, InsertPoint.before(op))

            # Map the original op's results to the cloned op's results
            for orig_res, cloned_res in zip(op_in_body.results, cloned.results):
                mapping[orig_res] = cloned_res

        # Step 3: Determine what the yield returns
        yield_val = yield_op.operands[0]

        # If it's a block arg or an intermediate result, it's in `mapping`.
        # If it's a constant captured from outside, fallback to itself.
        scalar_result = mapping.get(yield_val, yield_val)

        # Step 4: Wrap back into 0-d tensor
        from_elements = tensor.FromElementsOp(
            operands=[scalar_result],
            result_types=[op.results[0].type],  # Note: commonly .results over .res
        )
        rewriter.replace_matched_op(from_elements)
