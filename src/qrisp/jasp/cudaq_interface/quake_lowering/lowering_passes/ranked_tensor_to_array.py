# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Lower ranked tensors to CUDA-Q classical-control arrays."""

# Ranked Tensor → CC Array Lowering
# ===================================
#
# Lowers rank-1 tensors (tensor<NxT>) and their element accesses to CC memory
# operations (cc.alloca, cc.compute_ptr, cc.load), matching native CUDA-Q
# output. A tensor is an immutable value, a CC array is a pointer to storage,
# so the lowering is a type conversion:
#
#     tensor<NxT>  →  !cc.ptr<!cc.array<T x N>>
#
# Approach
# --------
# 1. Lower tensor operations. Dense tensor constants become a cc.alloca filled
#    by element stores, and element accesses become cc.compute_ptr + cc.load.
#    After this stage the only tensor-typed values left are the ones that cross
#    a boundary: function arguments and results, call results, and the
#    arguments and results of structured control flow.
# 2. Convert the remaining tensor types. One TypeConversionPattern rewrites
#    operation result types and the block arguments of nested regions, and a
#    second pattern rewrites function signatures. Operand types need no handling
#    of their own, because an operand's type is the type of the result that
#    defines it.
# 3. Lower tensor operations again. Element accesses on values that only became
#    CC array pointers in stage 2 - a function's array argument, say - can be
#    lowered once their definition has been converted.
#
# Handles both static and dynamic array indexing. Anything this pass cannot
# express is left in place rather than lowered incorrectly; the pipeline's
# closing verifier reports what remains.

from xdsl.dialects import arith, tensor
from xdsl.dialects import func as func_dialect
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    FloatAttr,
    FunctionType,
    IndexType,
    IntegerAttr,
    ModuleOp,
    TensorType,
    i64,
)
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
    op_type_rewrite_pattern,
)

from qrisp.jasp.cudaq_interface.quake_lowering.dialects.cc_dialect import (
    CcAllocaOp,
    CcArrayType,
    CcCastOp,
    CcComputePtrOp,
    CcIfOp,
    CcLoadOp,
    CcLoopOp,
    CcPtrType,
    CcStoreOp,
)
from qrisp.jasp.cudaq_interface.quake_lowering.lowering_passes.safeguard_no_ranked_tensor_linalg import (
    CudaqUnsupportedArrayOperationError,
)

# MLIR's sentinel for "dynamic dimension/offset"
_MLIR_DYNAMIC = -9223372036854775808


# ===================================================================
# Public entry point
# ===================================================================


def _lower_ranked_tensors(module: ModuleOp) -> None:
    """In-place pass: lower rank-1 tensor constants and accesses to CC arrays.

    Runs the three stages described at the top of this module: lower tensor
    operations, convert the tensor types that cross a boundary, then lower the
    operations that became lowerable as a result.
    """
    _lower_tensor_operations(module)
    _convert_tensor_types(module)
    _lower_tensor_operations(module)


# ===================================================================
# Stage 1 and 3: operation lowering
# ===================================================================


def _lower_tensor_operations(module: ModuleOp) -> None:
    """Materialize tensor constants and lower element accesses to CC memory ops."""
    PatternRewriteWalker(
        GreedyRewritePatternApplier(
            [
                MaterializeDenseArrayConstant(),
                LowerTensorExtract(),
                LowerSlicedTensorExtract(),
            ]
        ),
        apply_recursively=False,
    ).rewrite_module(module)

    # Erasing is a separate walk: an operation only becomes dead once its users
    # have been rewritten above, and mixing erasure into the walk that creates
    # the replacements leaves erased operations on xDSL's worklist.
    PatternRewriteWalker(EraseDeadTensorOp(), apply_recursively=False).rewrite_module(module)


class MaterializeDenseArrayConstant(RewritePattern):
    """Replace a dense rank-1 tensor constant with a cc.alloca and element stores."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ConstantOp, rewriter: PatternRewriter) -> None:
        """Materialize a tensor literal into CC array storage."""
        tensor_type = op.result.type
        if not _is_rank_1_tensor(tensor_type) or not any(op.result.uses):
            return

        values = _dense_values(op)
        if values is None:
            return

        element_type = tensor_type.element_type
        alloca = CcAllocaOp(CcArrayType(element_type, tensor_type.get_shape()[0]))

        new_ops: list[Operation] = [alloca]
        for index, value in enumerate(values):
            scalar = _scalar_constant(value, element_type)
            pointer = _element_pointer_op(alloca.result, index, element_type)
            new_ops += [scalar, pointer, CcStoreOp(scalar.result, pointer.results[0])]

        rewriter.replace_matched_op(new_ops, [alloca.result])


class LowerTensorExtract(RewritePattern):
    """Rewrite tensor.extract on a CC array pointer to cc.compute_ptr + cc.load."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tensor.ExtractOp, rewriter: PatternRewriter) -> None:
        """Lower a direct element access."""
        indices = list(op.indices)
        if not indices or not _is_array_pointer(op.tensor.type):
            return

        new_ops, loaded = _emit_element_load(op.tensor, indices[0])
        rewriter.replace_matched_op(new_ops, [loaded])


class LowerSlicedTensorExtract(RewritePattern):
    """Rewrite extract(collapse_shape(extract_slice(array))) to a single element load.

    JAX lowers a runtime index into an array to a unit-size slice followed by a
    reshape to a scalar tensor. The slice offset is the index being read, so the
    whole chain collapses to one cc.compute_ptr + cc.load on the CC array.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tensor.ExtractOp, rewriter: PatternRewriter) -> None:
        """Lower an element access that reaches the array through a slice."""
        collapse_op = op.tensor.owner
        if not isinstance(collapse_op, tensor.CollapseShapeOp):
            return

        slice_op = collapse_op.operands[0].owner
        if not isinstance(slice_op, tensor.ExtractSliceOp):
            return

        if not _is_array_pointer(slice_op.source.type):
            return

        new_ops, loaded = _emit_element_load(slice_op.source, _slice_index(slice_op))
        rewriter.replace_matched_op(new_ops, [loaded])


class EraseDeadTensorOp(RewritePattern):
    """Erase tensor operations whose results fell out of use during lowering."""

    _ERASABLE = (arith.ConstantOp, tensor.CollapseShapeOp, tensor.ExtractSliceOp)

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        """Erase an unused tensor constant, slice, or reshape."""
        if not isinstance(op, self._ERASABLE):
            return
        if not all(_is_rank_1_tensor(result.type) for result in op.results):
            return
        if any(any(result.uses) for result in op.results):
            return

        rewriter.erase_op(op)


# ===================================================================
# Stage 2: type conversion
# ===================================================================


def _convert_tensor_types(module: ModuleOp) -> None:
    """Convert rank-1 tensor types at every boundary that carries a value."""
    PatternRewriteWalker(TensorToArrayPointer(), apply_recursively=False).rewrite_module(module)
    PatternRewriteWalker(ConvertFuncSignature(), apply_recursively=False).rewrite_module(module)


class TensorToArrayPointer(TypeConversionPattern):
    """Convert rank-1 tensor types to CC array pointers.

    ``recursive`` makes the conversion reach into structured attributes, which is
    what rewrites a ``func.func`` signature: its function type holds the input and
    output types as parameters, so both directions are converted together.

    ``ops`` restricts the conversion to the operations that carry a value across a
    boundary. ``arith.constant`` is deliberately absent, because its dense value
    attribute carries a tensor type that has to keep describing the literal rather
    than the storage it is materialized into. Stage 1 has already replaced every
    dense constant that can be materialized, and the pipeline's closing verifier
    reports any that could not be.
    """

    recursive = True
    ops = (func_dialect.FuncOp, func_dialect.CallOp, CcLoopOp, CcIfOp)

    @attr_type_rewrite_pattern
    def convert_type(self, typ: TensorType) -> Attribute | None:
        """Map a rank-1 tensor type onto the CC array pointer that replaces it."""
        if not _is_rank_1_tensor(typ):
            return None
        return _array_pointer_type(typ)


class ConvertFuncSignature(RewritePattern):
    """Convert rank-1 tensor types in a function signature to CC array pointers.

    This is not folded into TensorToArrayPointer because xDSL's recursive type
    conversion does not descend into a FunctionType's parameters (checked against
    xdsl 0.59), so a function's declared argument and result types are the one
    boundary it leaves untouched. Both patterns map types through
    :func:`_converted_type`, so they stay in agreement.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func_dialect.FuncOp, rewriter: PatternRewriter) -> None:
        """Rewrite a function's declared argument and result types."""
        function_type = op.function_type
        inputs = [_converted_type(t) for t in function_type.inputs]
        outputs = [_converted_type(t) for t in function_type.outputs]

        if inputs == list(function_type.inputs) and outputs == list(function_type.outputs):
            return

        op.function_type = FunctionType.from_lists(inputs, outputs)


# ===================================================================
# Types
# ===================================================================


def _is_rank_1_tensor(t: Attribute) -> bool:
    """Return True if *t* is a ranked tensor type with exactly one dimension."""
    return isinstance(t, TensorType) and len(t.get_shape()) == 1


def _is_array_pointer(t: Attribute) -> bool:
    """Return True if *t* is a pointer to a CC array."""
    return isinstance(t, CcPtrType) and isinstance(t.element_type, CcArrayType)


def _converted_type(t: Attribute) -> Attribute:
    """Return the CC array pointer replacing *t*, or *t* itself if it is not a rank-1 tensor."""
    return _array_pointer_type(t) if _is_rank_1_tensor(t) else t


def _array_pointer_type(tensor_type: TensorType) -> CcPtrType:
    """Return the CC array pointer type that replaces *tensor_type*."""
    return CcPtrType(CcArrayType(tensor_type.element_type, tensor_type.get_shape()[0]))


def _pointee_element_type(pointer: SSAValue) -> Attribute:
    """Return the element type of the CC array *pointer* refers to."""
    return pointer.type.element_type.element_type


# ===================================================================
# Emission
# ===================================================================


def _element_pointer_op(base: SSAValue, index: int, element_type: Attribute) -> Operation:
    """Return the operation addressing element *index* of the array *base*.

    Element 0 is addressed with cc.cast and the remaining elements with
    cc.compute_ptr, mirroring what native CUDA-Q emits when it fills an array.
    """
    if index == 0:
        return CcCastOp(base, CcPtrType(element_type))
    return CcComputePtrOp(base, index, element_type)


def _emit_element_load(pointer: SSAValue, index: int | SSAValue) -> tuple[list[Operation], SSAValue]:
    """Return the operations loading element *index* of *pointer*, and the loaded value."""
    element_type = _pointee_element_type(pointer)
    new_ops: list[Operation] = []

    if isinstance(index, SSAValue) and isinstance(index.type, IndexType):
        index_cast = arith.IndexCastOp(index, i64)
        new_ops.append(index_cast)
        index = index_cast.result

    element_pointer = CcComputePtrOp(pointer, index, element_type)
    load = CcLoadOp(element_pointer.result)
    new_ops += [element_pointer, load]

    return new_ops, load.result


def _scalar_constant(value, element_type: Attribute) -> arith.ConstantOp:
    """Return a scalar arith.constant holding *value* at *element_type*."""
    if isinstance(element_type, (Float16Type, Float32Type, Float64Type)):
        return arith.ConstantOp(FloatAttr(float(value), element_type))
    return arith.ConstantOp(IntegerAttr(int(value), element_type))


def _dense_values(const_op: arith.ConstantOp) -> list | None:
    """Return the literal elements of a dense constant, or None if it has none."""
    if not isinstance(const_op.value, DenseIntOrFPElementsAttr):
        return None
    try:
        return list(const_op.value.iter_values())
    except (AttributeError, TypeError):
        return None


# ===================================================================
# Slices
# ===================================================================


def _slice_index(slice_op: tensor.ExtractSliceOp) -> int | SSAValue:
    """Return the element index a unit slice selects, static or dynamic."""
    _require_unit_slice(slice_op)

    static_offsets = list(slice_op.static_offsets.get_values())
    if static_offsets and static_offsets[0] != _MLIR_DYNAMIC:
        return int(static_offsets[0])

    offsets = list(slice_op.offsets)
    if not offsets:
        raise CudaqUnsupportedArrayOperationError(
            "This @cudaq_kernel function reads an array slice whose offset is "
            "neither a constant nor a runtime value, which CUDA-Q cannot compile."
        )
    return offsets[0]


def _require_unit_slice(slice_op: tensor.ExtractSliceOp) -> None:
    """Reject any slice that reads more than a single element.

    A CC array access loads one element, so only a one-dimensional slice of size
    one and stride one can be expressed. Wider slices are rejected rather than
    silently lowered to a read of their first element.
    """
    sizes = _static_values(slice_op.static_sizes)
    strides = _static_values(slice_op.static_strides)

    if sizes != [1] or strides != [1]:
        raise CudaqUnsupportedArrayOperationError(
            "This @cudaq_kernel function reads a range of a classical array "
            f"(sizes {sizes}, strides {strides}), which CUDA-Q cannot compile.\n\n"
            "Read individual array elements instead of slicing the array."
        )


def _static_values(dense_array) -> list[int]:
    """Return the integers held by a dense array property."""
    return [int(value) for value in dense_array.get_values()]
