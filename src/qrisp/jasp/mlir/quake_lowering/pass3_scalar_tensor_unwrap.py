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
PASS 3 – Tensor unwrapping and scalar return rewriting.
=========================================================

This pass applies a suite of greedy rewrite patterns to clean up rank-0 
tensor round-trips and unwrap scalar return types.

Architecture:
1. UnwrapFuncAndReturn: Rewrites func signatures and inserts tensor.extract
   at the return boundary to bridge the type gap.
2. FoldExtractOfDenseConstant: extract(constant dense<V>) -> constant V
3. FoldExtractOfFromElements: extract(from_elements V) -> V
4. ForwardExtractOfScalar: extract(scalar V) -> V (cleans up stale extracts)
5. EraseDeadTensorConstant: Cleans up orphaned dense constants.
6. EraseDeadFromElements: Cleans up orphaned from_elements.

Compatible with **xDSL ≥ 0.55**.
"""

from xdsl.context import Context
from xdsl.dialects import arith, func as func_dialect, tensor
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    FloatAttr,
    FunctionType,
    IntegerAttr,
    IntegerType,
    Float16Type,
    Float32Type,
    Float64Type,
    TensorType,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


# ===================================================================
# Public entry point
# ===================================================================


def unwrap_scalar_tensors(module: ModuleOp) -> None:
    """Applies the TensorUnwrapPass to the given module."""
    TensorUnwrapPass().apply(Context(), module)


# ------------------------------------------------------------------ #
# Helper
# ------------------------------------------------------------------ #
def _dense_to_scalar_attr(dense_attr, scalar_type):
    """Extract a scalar attribute from a rank-0 DenseIntOrFPElementsAttr."""
    if not isinstance(dense_attr, DenseIntOrFPElementsAttr):
        return None

    try:
        elements = list(dense_attr.iter_values())
    except (AttributeError, TypeError):
        return None

    if len(elements) != 1:
        return None

    raw = elements[0]

    if isinstance(scalar_type, IntegerType):
        return IntegerAttr(int(raw), scalar_type)
    if isinstance(scalar_type, (Float16Type, Float32Type, Float64Type)):
        return FloatAttr(float(raw), scalar_type)

    return None


# ------------------------------------------------------------------ #
# Pattern 1: Unwrap func signatures and return boundaries
# ------------------------------------------------------------------ #
class UnwrapFuncAndReturn(RewritePattern):
    """Updates func.func signatures and func.return operands from
    rank-0 tensor to scalar by injecting a tensor.extract at the boundary.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func_dialect.ReturnOp, rewriter: PatternRewriter) -> None:
        func_op = op.parent_op()
        if not isinstance(func_op, func_dialect.FuncOp):
            return

        changed = False
        new_operands = []
        new_return_types = []

        for operand in op.operands:
            t = operand.type
            if isinstance(t, TensorType) and not t.get_shape():
                # Isolate the boundary by inserting an extract
                extract_op = tensor.ExtractOp(operand, [], t.element_type)
                rewriter.insert_op(extract_op, InsertPoint.before(op))

                new_operands.append(extract_op.result)
                new_return_types.append(t.element_type)
                changed = True
            else:
                new_operands.append(operand)
                new_return_types.append(t)

        if not changed:
            return

        # 1. Replace the return op
        new_return = func_dialect.ReturnOp(*new_operands)
        rewriter.replace_matched_op(new_return)

        # 2. Update the function signature
        old_ftype = func_op.properties["function_type"]
        func_op.properties["function_type"] = FunctionType.from_lists(list(old_ftype.inputs), new_return_types)


class UnwrapFuncArgs(RewritePattern):
    """Updates func.func signatures from tensor<T> to T for arguments,
    injecting a tensor.from_elements at the start of the block.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func_dialect.FuncOp, rewriter: PatternRewriter) -> None:
        if not op.body.blocks:
            return

        # Safely get function_type across different xDSL versions
        ftype = op.function_type if hasattr(op, "function_type") else op.properties.get("function_type")
        if not ftype:
            return

        changed = False
        new_inputs = []
        args_to_wrap = []

        for i, t in enumerate(ftype.inputs):
            if isinstance(t, TensorType) and not t.get_shape():
                new_inputs.append(t.element_type)
                args_to_wrap.append((i, t, t.element_type))
                changed = True
            else:
                new_inputs.append(t)

        if not changed:
            return

        # 1. Update the function signature
        new_ftype = FunctionType.from_lists(new_inputs, list(ftype.outputs))
        if hasattr(op, "function_type"):
            op.function_type = new_ftype
        else:
            op.properties["function_type"] = new_ftype

        # 2. Update block arguments and insert boundary ops
        block = op.body.blocks[0]

        # Process in reverse to avoid index shifting issues
        for i, old_type, new_type in reversed(args_to_wrap):
            old_arg = block.args[i]

            # 1. Insert a new argument of the scalar type at the same index
            new_arg = block.insert_arg(new_type, i)

            # 2. Create the from_elements op to bridge the gap
            #    (Takes the NEW scalar arg, outputs the OLD tensor type)
            from_elem = tensor.FromElementsOp.create(operands=[new_arg], result_types=[old_type])

            # 3. Insert it at the top of the block
            if block.first_op is not None:
                block.insert_op_before(from_elem, block.first_op)
            else:
                block.add_op(from_elem)

            # 4. Replace all uses of the OLD tensor argument with our new from_elements result
            old_arg.replace_all_uses_with(from_elem.results[0])

            # 5. Erase the old argument (which is now completely unused) safely
            block.erase_arg(old_arg)


class UnwrapCall(RewritePattern):
    """Updates func.call operations to pass and expect scalar types
    instead of rank-0 tensors, injecting extracts/from_elements at boundaries.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func_dialect.CallOp, rewriter: PatternRewriter) -> None:
        changed = False

        new_operands = []
        new_result_types = []

        ops_to_insert_before = []
        ops_to_insert_after = []

        # 1. Handle Operands (Arguments passed to the call)
        for operand in op.operands:
            t = operand.type
            if isinstance(t, TensorType) and not t.get_shape():
                # Inject a tensor.extract before the call
                extract = tensor.ExtractOp(operand, [], t.element_type)
                ops_to_insert_before.append(extract)
                new_operands.append(extract.result)
                changed = True
            else:
                new_operands.append(operand)

        # 2. Handle Results (Returns from the call)
        for res in op.results:
            t = res.type
            if isinstance(t, TensorType) and not t.get_shape():
                new_result_types.append(t.element_type)
                changed = True
            else:
                new_result_types.append(t)

        if not changed:
            return

        # 3. Create the new call expecting scalars
        new_call = func_dialect.CallOp(op.callee, new_operands, new_result_types)

        # 4. Bridge the return boundary
        replacements = []
        for old_res, new_res in zip(op.results, new_call.results):
            if old_res.type != new_res.type:
                from_elem = tensor.FromElementsOp.create(operands=[new_res], result_types=[old_res.type])
                ops_to_insert_after.append(from_elem)
                replacements.append(from_elem.results[0])
            else:
                replacements.append(new_res)

        # 5. Insert operations and replace
        for ext_op in ops_to_insert_before:
            rewriter.insert_op(ext_op, InsertPoint.before(op))

        rewriter.replace_matched_op([new_call] + ops_to_insert_after, replacements)


# ------------------------------------------------------------------ #
# Pattern 2: tensor.extract of a dense constant → scalar constant
# ------------------------------------------------------------------ #
class FoldExtractOfDenseConstant(RewritePattern):
    """Replace a rank-0 tensor.extract of an arith.constant dense<V>
    with a scalar arith.constant V.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tensor.ExtractOp, rewriter: PatternRewriter) -> None:
        if len(op.indices) != 0:
            return

        src = op.tensor
        if not isinstance(src.type, TensorType) or src.type.get_shape():
            return

        defn = src.owner
        if not isinstance(defn, arith.ConstantOp):
            return

        scalar_attr = _dense_to_scalar_attr(defn.value, src.type.element_type)
        if scalar_attr is None:
            return

        new_const = arith.ConstantOp(scalar_attr)
        rewriter.replace_matched_op(new_const)


# ------------------------------------------------------------------ #
# Pattern 3: tensor.extract of a from_elements → forward scalar
# ------------------------------------------------------------------ #
class FoldExtractOfFromElements(RewritePattern):
    """Eliminate a tensor.extract by forwarding the scalar input
    directly from the producing tensor.from_elements op.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tensor.ExtractOp, rewriter: PatternRewriter) -> None:
        if len(op.indices) != 0:
            return

        defn = op.tensor.owner
        if isinstance(defn, tensor.FromElementsOp) and len(defn.operands) == 1:
            rewriter.replace_matched_op([], [defn.operands[0]])


# ------------------------------------------------------------------ #
# Pattern 4: tensor.extract of an already-scalar value → forward
# ------------------------------------------------------------------ #
class FoldExtractOfScalar(RewritePattern):
    """Eliminate a tensor.extract whose source is already a scalar."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tensor.ExtractOp, rewriter: PatternRewriter) -> None:
        if len(op.indices) != 0:
            return
        if not isinstance(op.tensor.type, TensorType):
            rewriter.replace_matched_op([], [op.tensor])


# ------------------------------------------------------------------ #
# Patterns 5 & 6: Erase dead ops
# ------------------------------------------------------------------ #
class EraseDeadTensorConstant(RewritePattern):
    """Erase arith.constant dense<...> : tensor<T> ops with no uses."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ConstantOp, rewriter: PatternRewriter) -> None:
        if isinstance(op.result.type, TensorType) and not any(op.result.uses):
            rewriter.erase_op(op)


class EraseDeadFromElements(RewritePattern):
    """Erase tensor.from_elements ops with no uses."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tensor.FromElementsOp, rewriter: PatternRewriter) -> None:
        if not any(op.result.uses):
            rewriter.erase_op(op)


# ------------------------------------------------------------------ #
# Pass
# ------------------------------------------------------------------ #
class TensorUnwrapPass(ModulePass):
    name = "tensor-unwrap"

    def apply(self, ctx: Context, op: ModuleOp) -> None:

        # Phase A: Call unwrapping (single pass, no recursion)
        PatternRewriteWalker(
            GreedyRewritePatternApplier([UnwrapCall()]),
            apply_recursively=False,
        ).rewrite_module(op)

        # Phase B: Func signature unwrapping + extract folding (recursive).
        # Erase-dead patterns are intentionally excluded here.
        # Mixing erase patterns with apply_recursively=True causes xDSL's
        # worklist to retain references to already-erased ops, leading to
        # "Operation insertion point must have a parent block" crashes.
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    UnwrapFuncAndReturn(),
                    UnwrapFuncArgs(),
                    FoldExtractOfDenseConstant(),
                    FoldExtractOfFromElements(),
                    FoldExtractOfScalar(),
                ]
            ),
            apply_recursively=True,
        ).rewrite_module(op)

        # Phase C: Erase dead tensor ops — separate non-recursive pass to avoid
        # the worklist retaining references to ops that were erased in Phase B.
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    EraseDeadTensorConstant(),
                    EraseDeadFromElements(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
