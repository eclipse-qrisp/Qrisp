# pass4_array_to_stdvec.py
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
PASS 4 – CC Array Pointer → CC StdVec Lowering (Entrypoint + Propagation)
===========================================================================

Rewrites ``!cc.ptr<!cc.array<T x N>>`` function parameters for CUDA-Q
runtime compatibility.

**Entrypoint function** (marked with ``cudaq.entrypoint = "true"``):
  - Parameter type → ``!cc.stdvec<T>``
  - Inserts ``cc.stdvec_data`` to extract ``!cc.ptr<!cc.array<T x ?>>``
  - All uses of the parameter get the dynamic ptr

**Internal helper functions**:
  - Only params that receive a dynamic-ptr value from the entrypoint
    (traced through call sites) are converted to ``!cc.ptr<!cc.array<T x ?>>``.
  - Params that receive locally-allocated arrays (``cc.alloca``) are left
    unchanged.

This ensures type consistency at ALL call sites without breaking functions
that receive concrete locally-allocated arrays.
"""

from xdsl.dialects import func as func_dialect
from xdsl.dialects.builtin import FunctionType, StringAttr
from xdsl.ir import Block, Region, SSAValue
from xdsl.rewriter import Rewriter

from qrisp.jasp.mlir.quake_lowering.cc_dialect import (
    CcArrayType,
    CcLoopOp,
    CcPtrType,
    CcStdVecDataOp,
    CcStdVecType,
)

# ===================================================================
# Public entry point
# ===================================================================


def lower_array_params_to_stdvec(module) -> None:
    """In-place pass: rewrite !cc.ptr<!cc.array<T x N>> params.

    1. Process entrypoint: param → !cc.stdvec<T> + cc.stdvec_data
    2. Trace which dynamic ptrs flow into internal function calls
    3. Update only those internal function params that receive dynamic ptrs
    """
    # Find the entrypoint and all functions
    entrypoint = None
    all_funcs = {}
    for op in list(module.body.blocks[0].ops):
        if op.name == "func.func":
            name = _get_func_name(op)
            all_funcs[name] = op
            if _is_entrypoint(op):
                entrypoint = op

    if entrypoint is None:
        return

    # Step 1: Process the entrypoint (stdvec + stdvec_data)
    dynamic_ptrs = _process_entrypoint(entrypoint)

    # Step 2: Trace dynamic ptrs through call sites and propagate to callees
    if dynamic_ptrs:
        _propagate_dynamic_types(entrypoint, all_funcs, dynamic_ptrs)


def _get_func_name(func_op) -> str:
    """Get the function name from a func.func op."""
    if hasattr(func_op, "sym_name"):
        name = func_op.sym_name
        if hasattr(name, "data"):
            return name.data
        return str(name)
    return ""


def _is_entrypoint(func_op) -> bool:
    """Check if a function has the cudaq.entrypoint attribute."""
    # func_op.attributes is a dict-like mapping in xDSL
    try:
        ep_attr = func_op.attributes.get("cudaq.entrypoint")
    except (TypeError, AttributeError):
        ep_attr = None

    if ep_attr is None:
        # Fallback: check if this is the "main" function
        return _get_func_name(func_op) == "main"

    if isinstance(ep_attr, StringAttr):
        return ep_attr.data == "true"
    return bool(ep_attr)


def _get_func_inputs(func_op) -> list:
    """Get function input types as a Python list (handles ArrayAttr)."""
    ftype = func_op.function_type
    inputs = ftype.inputs
    if hasattr(inputs, "data"):
        return list(inputs.data)
    return list(inputs)


def _get_func_outputs(func_op) -> list:
    """Get function output types as a Python list (handles ArrayAttr)."""
    ftype = func_op.function_type
    outputs = ftype.outputs
    if hasattr(outputs, "data"):
        return list(outputs.data)
    return list(outputs)


# ===================================================================
# Entrypoint processing (stdvec + stdvec_data)
# ===================================================================


def _process_entrypoint(func_op) -> set:
    """Process the entrypoint function: array ptr → stdvec + stdvec_data.

    Returns a set of SSAValues that are dynamic ptrs (stdvec_data results).
    """
    if not func_op.body.blocks:
        return set()

    block = func_op.body.blocks[0]
    inputs = _get_func_inputs(func_op)

    # Identify which args are !cc.ptr<!cc.array<T x N>> with static size
    rewrites = []  # list of (arg_idx, elem_type, static_size)
    for i, inp_type in enumerate(inputs):
        if _is_static_array_ptr(inp_type):
            arr_type = inp_type.element_type
            elem_type = arr_type.element_type
            size = arr_type.size.value.data
            rewrites.append((i, elem_type, size))

    if not rewrites:
        return set()

    # Build new function signature
    new_inputs = list(inputs)
    dynamic_ptrs = set()

    # Process each array param
    first_op = block.first_op
    for arg_idx, elem_type, size in rewrites:
        # 1. Create the stdvec type: !cc.stdvec<T>
        stdvec_type = CcStdVecType(elem_type)

        # 2. Create the dynamic ptr type for stdvec_data result:
        #    !cc.ptr<!cc.array<T x ?>>
        dyn_arr_type = CcArrayType(elem_type, -1)
        dyn_ptr_type = CcPtrType(dyn_arr_type)

        # 3. The old static ptr type (for matching in loops)
        old_static_arr_type = CcArrayType(elem_type, size)
        old_static_ptr_type = CcPtrType(old_static_arr_type)

        # 4. Update the function signature entry
        new_inputs[arg_idx] = stdvec_type

        # 5. Get the block arg (still has ptr type at this point)
        arg = block.args[arg_idx]

        # 6. Insert the stdvec_data op (takes arg as operand)
        stdvec_data_op = CcStdVecDataOp(arg, dyn_ptr_type)
        if first_op is not None:
            block.insert_ops_before([stdvec_data_op], first_op)
        else:
            block.add_op(stdvec_data_op)

        # 7. Replace all uses of arg EXCEPT in stdvec_data_op with the
        #    stdvec_data result (the dynamic ptr)
        _replace_uses_except(arg, stdvec_data_op.result, stdvec_data_op)

        # 8. Update cc.loop ops that now carry the dyn_ptr as a loop-carried
        #    value — their block args and result types still have the old
        #    static ptr type and need to be updated to dyn_ptr_type.
        _update_loop_types_recursive(func_op.body, old_static_ptr_type, dyn_ptr_type)

        # 9. Mutate the arg's type to stdvec in-place.
        _set_block_arg_type(arg, stdvec_type)

        # 10. Track the dynamic ptr
        dynamic_ptrs.add(stdvec_data_op.result)

    # Update the function type
    outputs = _get_func_outputs(func_op)
    func_op.function_type = FunctionType.from_lists(new_inputs, outputs)

    return dynamic_ptrs


# ===================================================================
# Propagation: trace dynamic ptrs through calls to internal functions
# ===================================================================


def _propagate_dynamic_types(func_op, all_funcs: dict, dynamic_ptrs: set) -> None:
    """Trace dynamic ptrs through call sites and update callee signatures.

    Walks the function body looking for func.call ops. For each call, checks
    which operands are dynamic ptrs. For those operands, updates the callee's
    corresponding parameter type to !cc.ptr<!cc.array<T x ?>>.

    Then recursively propagates into the callee (the callee's updated params
    become dynamic ptrs that may flow into further calls).
    """
    if not func_op.body.blocks:
        return

    # Also consider loop results as dynamic ptrs if they carry a dynamic ptr type
    _collect_dynamic_ptrs_from_loops(func_op.body, dynamic_ptrs)

    # Find all call sites in this function
    for block in func_op.body.blocks:
        for op in list(block.ops):
            if isinstance(op, func_dialect.CallOp):
                callee_name = op.callee.string_value()
                callee_op = all_funcs.get(callee_name)
                if callee_op is None:
                    continue

                # Check which operands are dynamic ptrs
                callee_dynamic_ptrs = set()
                for i, operand in enumerate(op.operands):
                    if operand in dynamic_ptrs or _is_dynamic_array_ptr(operand.type):
                        # This operand is a dynamic ptr — update callee's param
                        _update_callee_param(callee_op, i)
                        # The callee's block arg at this index is now also a dynamic ptr
                        if callee_op.body.blocks:
                            callee_block = callee_op.body.blocks[0]
                            if i < len(callee_block.args):
                                callee_dynamic_ptrs.add(callee_block.args[i])

                # Recursively propagate into the callee
                if callee_dynamic_ptrs:
                    _propagate_dynamic_types(callee_op, all_funcs, callee_dynamic_ptrs)


def _collect_dynamic_ptrs_from_loops(region: Region, dynamic_ptrs: set) -> None:
    """Collect loop results that carry dynamic ptr types into the dynamic_ptrs set."""
    for block in region.blocks:
        for op in list(block.ops):
            if isinstance(op, CcLoopOp):
                for res in op.res:
                    if _is_dynamic_array_ptr(res.type):
                        dynamic_ptrs.add(res)
                for nested_region in op.regions:
                    _collect_dynamic_ptrs_from_loops(nested_region, dynamic_ptrs)
            else:
                for nested_region in op.regions:
                    _collect_dynamic_ptrs_from_loops(nested_region, dynamic_ptrs)


def _update_callee_param(callee_op, param_idx: int) -> None:
    """Update a single parameter of a callee to dynamic ptr type.

    Only updates if the param is currently a static array ptr.
    """
    if not callee_op.body.blocks:
        return

    inputs = _get_func_inputs(callee_op)
    if param_idx >= len(inputs):
        return

    current_type = inputs[param_idx]
    if not _is_static_array_ptr(current_type):
        return  # Already dynamic or not an array ptr — skip

    # Determine the dynamic ptr type
    arr_type = current_type.element_type
    elem_type = arr_type.element_type
    old_size = arr_type.size.value.data
    dyn_arr_type = CcArrayType(elem_type, -1)
    dyn_ptr_type = CcPtrType(dyn_arr_type)

    old_static_arr_type = CcArrayType(elem_type, old_size)
    old_static_ptr_type = CcPtrType(old_static_arr_type)

    # Update function signature
    new_inputs = list(inputs)
    new_inputs[param_idx] = dyn_ptr_type
    outputs = _get_func_outputs(callee_op)
    callee_op.function_type = FunctionType.from_lists(new_inputs, outputs)

    # Update block arg type
    block = callee_op.body.blocks[0]
    if param_idx < len(block.args):
        _set_block_arg_type(block.args[param_idx], dyn_ptr_type)

    # Update any loops that carry this type
    _update_loop_types_recursive(callee_op.body, old_static_ptr_type, dyn_ptr_type)


# ===================================================================
# Type predicates
# ===================================================================


def _is_static_array_ptr(t) -> bool:
    """Check if type is !cc.ptr<!cc.array<T x N>> with N > 0."""
    return (
        isinstance(t, CcPtrType)
        and isinstance(t.element_type, CcArrayType)
        and t.element_type.size.value.data > 0
    )


def _is_dynamic_array_ptr(t) -> bool:
    """Check if type is !cc.ptr<!cc.array<T x ?>> (size == -1)."""
    return (
        isinstance(t, CcPtrType)
        and isinstance(t.element_type, CcArrayType)
        and t.element_type.size.value.data < 0
    )


# ===================================================================
# Shared helpers
# ===================================================================


def _update_loop_types_recursive(region: Region, old_type, new_type) -> None:
    """Recursively walk all cc.loop ops and update block arg / result types.

    Any block arg or result that has `old_type` is changed to `new_type`.
    """
    for block in region.blocks:
        for op in list(block.ops):
            if isinstance(op, CcLoopOp):
                # Update result types
                for res in op.res:
                    if _types_match(res.type, old_type):
                        _set_ssa_value_type(res, new_type)

                # Update block args in all regions
                for loop_region in op.regions:
                    for loop_block in loop_region.blocks:
                        for arg in loop_block.args:
                            if _types_match(arg.type, old_type):
                                _set_block_arg_type(arg, new_type)

                # Recurse into nested regions
                for loop_region in op.regions:
                    _update_loop_types_recursive(loop_region, old_type, new_type)
            else:
                for nested_region in op.regions:
                    _update_loop_types_recursive(nested_region, old_type, new_type)


def _types_match(t1, t2) -> bool:
    """Check if two types are structurally equivalent."""
    if type(t1) != type(t2):
        return False
    if isinstance(t1, CcPtrType) and isinstance(t2, CcPtrType):
        return _types_match(t1.element_type, t2.element_type)
    if isinstance(t1, CcArrayType) and isinstance(t2, CcArrayType):
        return (
            t1.element_type == t2.element_type
            and t1.size.value.data == t2.size.value.data
        )
    return t1 == t2


def _set_block_arg_type(arg, new_type) -> None:
    """Mutate a BlockArgument's type in-place."""
    if hasattr(arg, "_type"):
        arg._type = new_type
    elif hasattr(arg, "typ"):
        arg.typ = new_type
    else:
        for attr_name in ["_type", "typ", "_typ"]:
            if hasattr(arg, attr_name):
                try:
                    setattr(arg, attr_name, new_type)
                    return
                except (AttributeError, TypeError):
                    continue
        raise AttributeError(
            f"Cannot set type on BlockArgument. "
            f"Available attributes: {[a for a in dir(arg) if not a.startswith('__')]}"
        )


def _set_ssa_value_type(val: SSAValue, new_type) -> None:
    """Mutate an SSAValue's (OpResult's) type in-place."""
    if hasattr(val, "_type"):
        val._type = new_type
    elif hasattr(val, "typ"):
        val.typ = new_type
    else:
        for attr_name in ["_type", "typ", "_typ"]:
            if hasattr(val, attr_name):
                try:
                    setattr(val, attr_name, new_type)
                    return
                except (AttributeError, TypeError):
                    continue
        raise AttributeError(
            f"Cannot set type on SSAValue. "
            f"Available attributes: {[a for a in dir(val) if not a.startswith('__')]}"
        )


def _replace_uses_except(old_val: SSAValue, new_val: SSAValue, except_op) -> None:
    """Replace all uses of old_val with new_val, except in except_op."""
    for use in list(old_val.uses):
        if use.operation is not except_op:
            use.operation.operands[use.index] = new_val
