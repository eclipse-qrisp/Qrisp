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
Ranked Tensor → CC Array Lowering
=================================

Lowers ranked tensor constants (``tensor<NxT>``) and their element accesses
to CC memory operations (``cc.alloca``, ``cc.compute_ptr``, ``cc.load``),
matching native CUDA-Q output.

Additionally rewrites function signatures and call sites so that rank-1
tensor arguments are passed as ``!cc.ptr<!cc.array<T x N>>`` pointers,
enabling dynamic indexing across function boundaries without copying.

Pipeline within this pass
-------------------------
1. **Collect signature rewrites** — determine which functions have rank-1
   tensor arguments that need conversion to CC array pointers.
2. **Rewrite function definitions** — change ``tensor<NxT>`` block args to
   ``!cc.ptr<!cc.array<T x N>>``.
3. **Process each function body** — materialize local tensor constants into
   CC arrays (populating an ``array_map``), rewrite tensor access patterns,
   and rewrite outgoing calls using the populated ``array_map`` so that
   already-materialized arrays are passed directly (no redundant copies).

Handles both static and dynamic array indexing.
"""

from xdsl.dialects import arith, func as func_dialect, tensor
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    FloatAttr,
    FunctionType,
    IndexType,
    IntegerAttr,
    IntegerType,
    TensorType,
    Float64Type,
    Float32Type,
    Float16Type,
    i64,
)
from xdsl.ir import Block, Region, SSAValue
from xdsl.rewriter import Rewriter

from qrisp.jasp.mlir.quake_lowering.cc_dialect_v3 import (
    CcAllocaOp,
    CcArrayType,
    CcCastOp,
    CcComputePtrOp,
    CcLoadOp,
    CcPtrType,
    CcStoreOp,
)


# MLIR's sentinel for "dynamic dimension/offset"
_MLIR_DYNAMIC = -9223372036854775808


# ===================================================================
# Public entry point
# ===================================================================


def lower_ranked_tensors(module) -> None:
    """In-place pass: lower ranked tensor constants + accesses to CC arrays.

    Pipeline:
    1. Collect which functions need signature changes (tensor<NxT> → ptr)
    2. Rewrite function definitions (change block arg types)
    3. Process each function body (materialize + rewrite accesses + rewrite calls)
    """
    # Step 1: Determine which functions need signature changes.
    func_rewrites = _collect_signature_rewrites(module)

    # Step 2: Rewrite function definitions (tensor args → ptr args).
    for op in list(module.body.blocks[0].ops):
        if op.name == "func.func" and op.sym_name.data in func_rewrites:
            _rewrite_func_def(op, func_rewrites[op.sym_name.data])

    # Step 3: Process each function body (materialize + rewrite + calls).
    for op in list(module.body.blocks[0].ops):
        if op.name == "func.func":
            _process_func(op, func_rewrites)


# ===================================================================
# Step 1: Collect signature rewrites
# ===================================================================


def _collect_signature_rewrites(module) -> dict:
    """Determine which functions have rank-1 tensor args needing ptr conversion."""
    func_rewrites = {}
    for op in list(module.body.blocks[0].ops):
        if op.name != "func.func":
            continue
        ftype = op.function_type
        rewrites = []
        for i, inp_type in enumerate(ftype.inputs):
            if isinstance(inp_type, TensorType) and len(inp_type.get_shape()) == 1:
                arr_type = CcArrayType(inp_type.element_type, inp_type.get_shape()[0])
                rewrites.append((i, inp_type, arr_type, CcPtrType(arr_type)))
        if rewrites:
            func_rewrites[op.sym_name.data] = rewrites
    return func_rewrites


# ===================================================================
# Step 2: Rewrite function definitions
# ===================================================================


def _rewrite_func_def(func_op, rewrites: list) -> None:
    """Change tensor args to ptr args in a function definition."""
    ftype = func_op.function_type
    new_inputs = list(ftype.inputs)
    block = func_op.body.blocks[0]

    for arg_idx, _, _, ptr_type in rewrites:
        new_inputs[arg_idx] = ptr_type
        Rewriter.replace_value_with_new_type(block.args[arg_idx], ptr_type)

    func_op.function_type = FunctionType.from_lists(new_inputs, list(ftype.outputs))


# ===================================================================
# Step 3: Process function bodies
# ===================================================================


def _process_func(func_op, func_rewrites: dict) -> None:
    """Process a function: materialize arrays, rewrite accesses, rewrite calls."""
    func_array_map: dict[SSAValue, tuple[SSAValue, any, int]] = {}
    _process_block_recursive(func_op.body, func_array_map, func_rewrites)


def _process_block_recursive(region: Region, array_map: dict, func_rewrites: dict) -> None:
    """Recursively process all blocks."""
    for block in region.blocks:
        _process_block(block, array_map)
        _rewrite_calls_in_block(block, func_rewrites, array_map)
        for op in list(block.ops):
            for nested_region in op.regions:
                _process_block_recursive(nested_region, array_map, func_rewrites)


def _process_block(block: Block, array_map: dict) -> None:
    """Materialize arrays, rewrite accesses."""
    # Phase 0: Register ptr-typed block arguments.
    for arg in block.args:
        if isinstance(arg.type, CcPtrType) and isinstance(arg.type.element_type, CcArrayType):
            inner = arg.type.element_type
            array_map[arg] = (arg, inner.element_type, inner.size.value.data)

    # Phase 1: Materialize ranked tensor constants.
    for op in list(block.ops):
        if isinstance(op, arith.ConstantOp) and _is_ranked_1_tensor(op.result.type):
            _materialize_array(op, block, array_map)

    # Phase 2: Rewrite tensor access patterns.
    for op in list(block.ops):
        if op.name == "tensor.extract":
            _rewrite_tensor_extract(op, block, array_map)
        elif op.name == "tensor.extract_slice":
            _rewrite_extract_slice_chain(op, block, array_map)

    # Phase 3: Erase dead tensor constants.
    for op in list(block.ops):
        if (isinstance(op, arith.ConstantOp)
                and _is_ranked_1_tensor(op.result.type)
                and not any(op.result.uses)):
            Rewriter.erase_op(op, safe_erase=False)


# ===================================================================
# Call rewriting (uses array_map — no redundant copies)
# ===================================================================


def _rewrite_calls_in_block(block: Block, func_rewrites: dict, array_map: dict) -> None:
    """Rewrite func.call ops, passing existing CC arrays from array_map."""
    for op in list(block.ops):
        if not isinstance(op, func_dialect.CallOp):
            continue
        callee = op.callee.string_value()
        if callee not in func_rewrites:
            continue

        new_operands = list(op.operands)
        for arg_idx, old_type, arr_type, _ in func_rewrites[callee]:
            tensor_val = op.operands[arg_idx]
            if tensor_val in array_map:
                new_operands[arg_idx] = array_map[tensor_val][0]
            else:
                new_operands[arg_idx] = _materialize_tensor_value(
                    tensor_val, old_type, arr_type, block, op
                )

        new_call = func_dialect.CallOp(op.callee, new_operands, list(op.result_types))
        block.insert_ops_before([new_call], op)
        for old_res, new_res in zip(op.results, new_call.results):
            old_res.replace_by(new_res)
        Rewriter.erase_op(op, safe_erase=False)


# ===================================================================
# Helpers
# ===================================================================


def _is_ranked_1_tensor(t) -> bool:
    return isinstance(t, TensorType) and len(t.get_shape()) == 1


def _cast_index_to_i64(val: SSAValue, block: Block, insert_before) -> SSAValue:
    if isinstance(val.type, IndexType):
        cast = arith.IndexCastOp(val, i64)
        block.insert_ops_before([cast], insert_before)
        return cast.result
    return val


def _emit_load_from_array(arr_ptr, index, elem_type, block, insert_before):
    """Emit cc.compute_ptr + cc.load."""
    if isinstance(index, SSAValue):
        index = _cast_index_to_i64(index, block, insert_before)
    ptr = CcComputePtrOp(arr_ptr, index, elem_type)
    load = CcLoadOp(ptr.result)
    block.insert_ops_before([ptr, load], insert_before)
    return load.result


def _make_scalar_const(value, elem_type):
    if isinstance(elem_type, (Float64Type, Float32Type, Float16Type)):
        return arith.ConstantOp(FloatAttr(float(value), elem_type))
    return arith.ConstantOp(IntegerAttr(int(value), elem_type))


# ===================================================================
# Array materialization
# ===================================================================


def _materialize_array(const_op, block: Block, array_map: dict) -> None:
    """Replace a ranked tensor constant with cc.alloca + stores."""
    values = _get_dense_values(const_op)
    if values is None:
        return

    result_type = const_op.result.type
    size = result_type.get_shape()[0]
    elem_type = result_type.element_type
    arr_type = CcArrayType(elem_type, size)

    alloca = CcAllocaOp(arr_type)
    block.insert_ops_before([alloca], const_op)

    for i, val in enumerate(values):
        scalar_const = _make_scalar_const(val, elem_type)
        block.insert_ops_before([scalar_const], const_op)
        if i == 0:
            cast = CcCastOp(alloca.result, CcPtrType(elem_type))
            block.insert_ops_before([cast], const_op)
            store = CcStoreOp(scalar_const.result, cast.result)
        else:
            ptr = CcComputePtrOp(alloca.result, i, elem_type)
            block.insert_ops_before([ptr], const_op)
            store = CcStoreOp(scalar_const.result, ptr.result)
        block.insert_ops_before([store], const_op)

    array_map[const_op.result] = (alloca.result, elem_type, size)


def _materialize_tensor_value(
    tensor_val: SSAValue,
    tensor_type: TensorType,
    arr_type: CcArrayType,
    block: Block,
    insert_before,
) -> SSAValue:
    """Materialize a non-constant tensor into a CC array (fallback)."""
    size = tensor_type.get_shape()[0]
    elem_type = tensor_type.element_type

    alloca = CcAllocaOp(arr_type)
    block.insert_ops_before([alloca], insert_before)

    for i in range(size):
        idx = arith.ConstantOp(IntegerAttr(i, i64))
        extract = tensor.ExtractOp(tensor_val, [idx.result], elem_type)
        block.insert_ops_before([idx, extract], insert_before)
        if i == 0:
            cast = CcCastOp(alloca.result, CcPtrType(elem_type))
            block.insert_ops_before([cast], insert_before)
            store = CcStoreOp(extract.result, cast.result)
        else:
            ptr = CcComputePtrOp(alloca.result, i, elem_type)
            block.insert_ops_before([ptr], insert_before)
            store = CcStoreOp(extract.result, ptr.result)
        block.insert_ops_before([store], insert_before)

    return alloca.result


def _get_dense_values(const_op):
    if not isinstance(const_op.value, DenseIntOrFPElementsAttr):
        return None
    try:
        return list(const_op.value.iter_values())
    except (AttributeError, TypeError):
        return None


# ===================================================================
# Access pattern rewriting
# ===================================================================


def _rewrite_tensor_extract(extract_op, block: Block, array_map: dict) -> None:
    """Rewrite ranked tensor.extract → cc.compute_ptr + cc.load."""
    source = extract_op.tensor
    indices = list(extract_op.indices)
    if not indices or source not in array_map:
        return
    arr_ptr, elem_type, _ = array_map[source]
    loaded = _emit_load_from_array(arr_ptr, indices[0], elem_type, block, extract_op)
    extract_op.result.replace_by(loaded)
    Rewriter.erase_op(extract_op, safe_erase=False)


def _rewrite_extract_slice_chain(slice_op, block: Block, array_map: dict) -> None:
    """Rewrite extract_slice → collapse_shape → extract[] chain."""
    source = slice_op.operands[0]
    if source not in array_map:
        return
    arr_ptr, elem_type, _ = array_map[source]

    offset = _get_static_offset(slice_op)
    collapse_op = _find_single_user(slice_op.results[0], "tensor.collapse_shape")
    if collapse_op is None:
        return
    extract_op = _find_single_user(collapse_op.results[0], "tensor.extract")
    if extract_op is None:
        return

    if offset is not None:
        loaded = _emit_load_from_array(arr_ptr, offset, elem_type, block, extract_op)
    else:
        dynamic_offsets = list(slice_op.offsets)
        if not dynamic_offsets:
            return
        loaded = _emit_load_from_array(arr_ptr, dynamic_offsets[0], elem_type, block, extract_op)

    extract_op.result.replace_by(loaded)
    Rewriter.erase_op(extract_op, safe_erase=False)
    if not any(collapse_op.results[0].uses):
        Rewriter.erase_op(collapse_op, safe_erase=False)
    if not any(slice_op.results[0].uses):
        Rewriter.erase_op(slice_op, safe_erase=False)


def _get_static_offset(slice_op) -> int | None:
    """Extract the static offset from a tensor.extract_slice op.

    Returns None if the offset is dynamic (sentinel value i64::MIN).
    """
    values = list(slice_op.static_offsets.get_values())
    if values and values[0] != _MLIR_DYNAMIC:
        return int(values[0])
    return None


def _find_single_user(value: SSAValue, op_name: str):
    """Find a single user of a value with the given operation name.

    Returns the operation if found, otherwise None.
    """
    for use in value.uses:
        if use.operation.name == op_name:
            return use.operation
    return None
