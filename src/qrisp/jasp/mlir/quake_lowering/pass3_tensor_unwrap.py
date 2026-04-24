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
PASS 3 – Tensor unwrapping and scalar return rewriting.
=========================================================

After earlier lowering passes, the IR may still contain rank-0 tensor
round-trips introduced by the JAX / StableHLO → arith lowering:

.. code-block:: mlir

    %t = arith.constant dense<42> : tensor<i64>
    %s = tensor.extract %t[] : tensor<i64>
    // ... %s used as a scalar i64 ...

Additionally, ``func.func`` return types may still be declared as
``tensor<T>`` even though the value is a trivial rank-0 constant.

This pass applies four greedy rewrite patterns to clean this up:

Pattern 1 – ``FoldExtractOfDenseConstant``
    Matches a ``tensor.extract`` whose source is an ``arith.constant
    dense<V> : tensor<T>`` with rank-0 shape.  Replaces the extract with
    a scalar ``arith.constant V : T``.

Pattern 2 – ``FoldExtractOfScalar``
    Matches a ``tensor.extract %v[]`` where ``%v`` already has a scalar
    type (not a ``TensorType``).  Forwards ``%v`` directly, eliminating
    the redundant extract.

Pattern 3 – ``EraseDeadTensorConstant``
    Erases any ``arith.constant dense<...> : tensor<T>`` that has no
    remaining uses (typically left behind after Pattern 1 folds all
    extracts).

Pattern 4 – ``UnwrapScalarReturn``
    Matches a ``func.return`` whose operand is an ``arith.constant
    dense<V> : tensor<T>`` (rank-0).  Replaces the operand with a scalar
    constant and rewrites the enclosing ``func.func`` signature from
    ``-> tensor<T>`` to ``-> T``.

All patterns are applied via a ``PatternRewriteWalker`` with
``apply_recursively=True`` so that fold chains are resolved
automatically.

Compatible with **xDSL ≥ 0.55**.
"""

try:
    from xdsl.context import Context
except ImportError:
    from xdsl.ir import Context

from xdsl.dialects import arith, func as func_dialect, tensor
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    FloatAttr,
    FunctionType,
    IntegerAttr,
    IntegerType,
    Float16Type,
    Float32Type,
    Float64Type,
    ModuleOp,
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


# ------------------------------------------------------------------ #
# Helper
# ------------------------------------------------------------------ #
def _dense_to_scalar_attr(dense_attr, scalar_type):
    """Extract a scalar attribute from a rank-0 ``DenseIntOrFPElementsAttr``.

    Parameters
    ----------
    dense_attr : DenseIntOrFPElementsAttr
        The dense attribute to unwrap.  Must contain exactly one element.
    scalar_type : Attribute
        The target scalar type (e.g. ``i64``, ``f64``).

    Returns
    -------
    IntegerAttr | FloatAttr | None
        The corresponding scalar attribute, or ``None`` if the conversion
        is not supported.
    """

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
# Pattern 1: tensor.extract of a dense constant → scalar constant
# ------------------------------------------------------------------ #
class FoldExtractOfDenseConstant(RewritePattern):
    """Replace a rank-0 ``tensor.extract`` of an ``arith.constant dense<V>``
    with a scalar ``arith.constant V``.

    .. code-block:: mlir

        // Before
        %t = arith.constant dense<42> : tensor<i64>
        %s = tensor.extract %t[] : tensor<i64>

        // After
        %s = arith.constant 42 : i64
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: tensor.ExtractOp, rewriter: PatternRewriter
    ) -> None:
        if len(op.indices) != 0:
            return

        src = op.tensor
        src_type = src.type

        if not isinstance(src_type, TensorType):
            return
        if src_type.get_shape():
            return

        defn = src.owner
        if not isinstance(defn, arith.ConstantOp):
            return

        scalar_type = src_type.element_type
        scalar_attr = _dense_to_scalar_attr(defn.value, scalar_type)
        if scalar_attr is None:
            return

        new_const = arith.ConstantOp(scalar_attr)
        rewriter.replace_matched_op(new_const)


# ------------------------------------------------------------------ #
# Pattern 2: tensor.extract of an already-scalar value → forward
# ------------------------------------------------------------------ #
class FoldExtractOfScalar(RewritePattern):
    """Eliminate a ``tensor.extract`` whose source is already a scalar.

    This can occur after earlier passes have rewritten a producer's
    result type from ``tensor<T>`` to ``T`` but left a stale extract
    in place.

    .. code-block:: mlir

        // Before  (source already has type i64, not tensor<i64>)
        %s = tensor.extract %v[] : tensor<i64>

        // After
        // (uses of %s replaced by %v; extract erased)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: tensor.ExtractOp, rewriter: PatternRewriter
    ) -> None:
        if len(op.indices) != 0:
            return
        src = op.tensor
        if isinstance(src.type, TensorType):
            return
        rewriter.replace_matched_op([], [src])


# ------------------------------------------------------------------ #
# Pattern 3: erase dead tensor constants
# ------------------------------------------------------------------ #
class EraseDeadTensorConstant(RewritePattern):
    """Erase ``arith.constant dense<...> : tensor<T>`` ops with no uses.

    These are typically left behind after Pattern 1 has folded all
    ``tensor.extract`` consumers into scalar constants.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.ConstantOp, rewriter: PatternRewriter
    ) -> None:
        if not isinstance(op.result.type, TensorType):
            return
        if any(True for _ in op.result.uses):
            return
        rewriter.erase_matched_op()

# ------------------------------------------------------------------ #
# Pattern 4 – unwrap scalar func.return operands + update signature
# ------------------------------------------------------------------ #

class UnwrapScalarReturn(RewritePattern):
    """Rewrite ``func.return %tensor : tensor<T>`` (rank-0) into a scalar
    return and update the enclosing ``func.func`` signature.

    Only applies when:

    * The return op has exactly one operand.
    * The enclosing op is a ``func.func``.

    Handles two source patterns:

    1. ``arith.constant dense<V> : tensor<T>`` → ``arith.constant V : T``
    2. ``tensor.from_elements %s : tensor<T>`` → forward ``%s``

    .. code-block:: mlir

        // Before (case 1)
        %t = arith.constant dense<0> : tensor<i64>
        func.return %t : tensor<i64>

        // Before (case 2)
        %s = quake.discriminate ... : i1
        %t = tensor.from_elements %s : tensor<i1>
        func.return %t : tensor<i1>

        // After (both cases)
        func.return %scalar : T

    .. warning::

        This changes the function's external ABI.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: func_dialect.ReturnOp, rewriter: PatternRewriter
    ) -> None:
        if len(op.operands) != 1:
            return

        ret_val = op.operands[0]
        ret_type = ret_val.type

        if not isinstance(ret_type, TensorType):
            return
        if ret_type.get_shape():
            return

        scalar_type = ret_type.element_type
        defn = ret_val.owner
        new_scalar = None

        # Case 1: arith.constant dense<V> : tensor<T>
        if isinstance(defn, arith.ConstantOp):
            scalar_attr = _dense_to_scalar_attr(defn.value, scalar_type)
            if scalar_attr is None:
                return
            new_scalar = arith.ConstantOp(scalar_attr)
            rewriter.insert_op_before_matched_op(new_scalar)
            scalar_val = new_scalar.result

        # Case 2: tensor.from_elements %s : tensor<T>
        elif isinstance(defn, tensor.FromElementsOp):
            if len(defn.operands) != 1:
                return
            scalar_val = defn.operands[0]

        else:
            return

        # Capture parent before replacement
        func_op = op.parent_op()

        # Replace return
        new_return = func_dialect.ReturnOp(scalar_val)
        rewriter.replace_matched_op(new_return)

        # Update function signature
        if isinstance(func_op, func_dialect.FuncOp):
            old_ftype = func_op.properties["function_type"]
            new_outputs = [
                scalar_type
                if isinstance(t, TensorType) and not t.get_shape()
                else t
                for t in old_ftype.outputs
            ]
            new_ftype = FunctionType.from_lists(
                list(old_ftype.inputs),
                new_outputs,
            )
            func_op.properties["function_type"] = new_ftype

# ------------------------------------------------------------------ #
# Pattern 5 – erase dead tensor.from_elements ops
# ------------------------------------------------------------------ #

class EraseDeadFromElements(RewritePattern):
    """Erase ``tensor.from_elements`` ops that have no remaining uses.

    After Pattern 4 (``UnwrapScalarReturn``) rewrites a ``func.return``
    to consume the scalar operand directly, the ``tensor.from_elements``
    that originally wrapped the scalar into a rank-0 tensor may become
    dead.  This pattern removes such ops to keep the IR clean.

    .. code-block:: mlir

        // Before
        %t = tensor.from_elements %s : tensor<T>
        // no remaining uses of %t

        // After
        // (the tensor.from_elements op is erased)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: tensor.FromElementsOp, rewriter: PatternRewriter
    ) -> None:
        if any(True for _ in op.result.uses):
            return
        rewriter.erase_matched_op()

# ------------------------------------------------------------------ #
# Pass
# ------------------------------------------------------------------ #
class TensorUnwrapPass(ModulePass):
    """xDSL ``ModulePass`` that applies all four tensor-unwrapping
    patterns in a single greedy walk.

    Usage from Python::

        from pass4_tensor_unwrap import unwrap_tensors
        unwrap_tensors(module)

    Or as a registered pass::

        ctx.register_pass(TensorUnwrapPass)
    """

    name = "tensor-unwrap"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    FoldExtractOfDenseConstant(),
                    FoldExtractOfScalar(),
                    EraseDeadTensorConstant(),
                    UnwrapScalarReturn(),
                    EraseDeadFromElements(),
                ]
            ),
            apply_recursively=True,
        ).rewrite_module(op)


def unwrap_tensors(module: ModuleOp) -> None:
    TensorUnwrapPass().apply(Context(), module)
