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

from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)
from xdsl.context import Context
from xdsl.passes import ModulePass
from xdsl.dialects import linalg, tensor, arith
from xdsl.dialects.builtin import ModuleOp, IntegerAttr, IntegerType, i1


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


class FoldCmpiExtui(RewritePattern):
    """Fold `arith.CmpiOp <pred>, (arith.extui %x : i1 to iN), 0` into a
    direct boolean operation on `%x`.

    When a boolean (`i1`) value is zero-extended to a wider integer and then
    compared against zero, the comparison is redundant — the result can be
    expressed directly in terms of the original `i1`:

        cmpi ne, (extui %x : i1 to iN), 0   →   %x
        cmpi eq, (extui %x : i1 to iN), 0   →   xori %x, true   (NOT %x)

    Proof:
        extui maps: false→0, true→1.
        - (ne, _, 0): 0≠0=false, 1≠0=true   → identity
        - (eq, _, 0): 0==0=true,  1==0=false → negation

    This pattern is critical for cleaning up lowering artifacts where
    higher-level boolean conditions (e.g., `if measurement == 0`) are
    lowered through integer extension and comparison.

    Preconditions:
        - LHS is defined by `arith.extui` from `i1`.
        - RHS is an `arith.ConstantOp` with value 0.
        - Predicate is either `eq` (0) or `ne` (1).
    """

    # arith.CmpiOpPredicate encoding (MLIR standard):
    _PRED_EQ: int = 0
    _PRED_NE: int = 1

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.CmpiOp, rewriter: PatternRewriter
    ) -> None:
        # ── Guard: LHS must be extui from i1 ───────────────────────────
        extui = op.lhs.owner
        if not isinstance(extui, arith.ExtUIOp):
            return
        if extui.input.type != i1:
            return

        # ── Guard: RHS must be constant 0 ──────────────────────────────
        rhs_op = op.rhs.owner
        if not isinstance(rhs_op, arith.ConstantOp):
            return
        rhs_val = rhs_op.value
        if not isinstance(rhs_val, IntegerAttr) or rhs_val.value.data != 0:
            return

        # ── The original boolean value before extension ─────────────────
        bool_val = extui.input
        predicate = op.predicate.value.data

        if predicate == self._PRED_NE:
            # cmpi ne, (extui %x), 0  ≡  %x
            # Simply forward the original i1 value.
            rewriter.replace_matched_op([], [bool_val])

        elif predicate == self._PRED_EQ:
            # cmpi eq, (extui %x), 0  ≡  NOT %x
            # Implement NOT as xori with true (1).
            true_const = arith.ConstantOp(IntegerAttr(1, i1))
            xor_op = arith.XOrI(bool_val, true_const)
            rewriter.insert_op_before_matched_op(true_const)
            rewriter.replace_matched_op(xor_op)

        else:
            # Other predicates (slt, uge, ...) are not handled.
            return

# ====================================================================== #

class FoldExtractFromElements(RewritePattern):
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

class FoldBoolCmpWithConst(RewritePattern):
    """
    Vereinfache boolsche Vergleiche gegen Konstante:

      cmpi eq, %x, true  -> %x
      cmpi ne, %x, false -> %x
      cmpi eq, %x, false -> not %x
      cmpi ne, %x, true  -> not %x
    """

    _PRED_EQ = 0
    _PRED_NE = 1

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.CmpiOp, rewriter: PatternRewriter):
        # Nur i1-Vergleiche
        if op.lhs.type != i1 or op.rhs.type != i1:
            return

        lhs, rhs = op.lhs, op.rhs
        pred = op.predicate.value.data

        def as_bool_const(v):
            c = v.owner
            if not isinstance(c, arith.ConstantOp):
                return None
            attr = c.value
            if not isinstance(attr, IntegerAttr):
                return None
            if attr.type != i1:
                return None
            return attr.value.data  # 0 oder 1

        lhs_c = as_bool_const(lhs)
        rhs_c = as_bool_const(rhs)

        if lhs_c is None and rhs_c is None:
            return

        # konstante Seite / variable Seite bestimmen
        if lhs_c is not None:
            const_val, var = lhs_c, rhs
        else:
            const_val, var = rhs_c, lhs

        # Fälle
        if pred == self._PRED_EQ:
            if const_val == 1:
                # x == true  → x
                rewriter.replace_matched_op([], [var])
            else:
                # x == false → not x
                true_c = arith.ConstantOp.from_int_and_width(1, 1)
                xor = arith.XOrIOp(true_c.result, var)
                rewriter.insert_op_before_matched_op(true_c)
                rewriter.replace_matched_op(xor)
        elif pred == self._PRED_NE:
            if const_val == 0:
                # x != false → x
                rewriter.replace_matched_op([], [var])
            else:
                # x != true → not x
                true_c = arith.ConstantOp.from_int_and_width(1, 1)
                xor = arith.XOrIOp(true_c.result, var)
                rewriter.insert_op_before_matched_op(true_c)
                rewriter.replace_matched_op(xor)
        else:
            return


class FoldCmpiExtui(RewritePattern):
    """
    cmpi ne, (extui %x : i1 to iN), 0  →  %x

    Nach FoldExtractFromElements ist dein Pattern:

      %15 = arith.extui %11 : i1 to i32
      %18 = arith.constant 0 : i32
      %19 = arith.cmpi ne, %15, %18 : i32

    genau dieser Fall.
    """

    _PRED_NE = 1

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.CmpiOp, rewriter: PatternRewriter):
        pred = op.predicate.value.data
        if pred != self._PRED_NE:
            return

        lhs, rhs = op.lhs, op.rhs

        # RHS muss konstante 0 sein
        rhs_op = rhs.owner
        if not isinstance(rhs_op, arith.ConstantOp):
            return
        rhs_val = rhs_op.value
        if not isinstance(rhs_val, IntegerAttr) or rhs_val.value.data != 0:
            return

        # LHS muss extui von i1 sein
        ext = lhs.owner
        if not isinstance(ext, arith.ExtUIOp):
            return
        if ext.input.type != i1:
            return

        # cmpi ne, (extui %x), 0 → %x
        rewriter.replace_matched_op([], [ext.input])


class FoldRedundantBoolChain(ModulePass):
    """
    Module-Pass: alle drei Patterns greedy anwenden.

    Wichtig: danach Canonicalize + DCE laufen lassen.
    """

    name = "fold-redundant-bool-chain"

    def apply(self, ctx: Context, module: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([
                FoldExtractFromElements(),
                FoldBoolCmpWithConst(),
                FoldCmpiExtui(),
            ]),
            apply_recursively=True,
        ).rewrite_module(module)


class FoldRedundantMeasurementCondition(ModulePass):
    """Module-level pass that eliminates redundant scalar `linalg.generic`
    chains commonly produced by the JAX→MLIR quantum lowering pipeline.

    Applies two rewrite patterns greedily until fixpoint:
        1. **FoldScalarLinalgGeneric** — converts 0-d generics to scalar ops
        2. **FoldCmpiExtui** — simplifies integer comparisons on extended bools

    For best results, schedule this pass **before** standard canonicalize
    and DCE passes:

        pipeline = [
            FoldRedundantMeasurementCondition,   # this pass
            Canonicalize,                         # tensor.extract(from_elements) folding
            CommonSubexpressionElimination,
            DeadCodeElimination,                  # remove unused ops
        ]

    After the full pipeline, a chain like:

        %8 (tensor<i1>)  →  extui i64  →  cmpi eq 0  →  extui i32
                          →  extract    →  cmpi ne 0  →  scf.if

    collapses to:

        %cond = tensor.extract %8[] : tensor<i1>
        scf.if %cond → ...   (branches swapped)
    """

    name = "fold-redundant-measurement-condition"

    def apply(self, ctx: Context, module: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([
                FoldScalarLinalgGeneric(),
                FoldCmpiExtui(),
            ]),
            apply_recursively=True,
        ).rewrite_module(module)