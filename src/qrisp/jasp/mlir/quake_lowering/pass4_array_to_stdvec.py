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
PASS 4 – CC Array Pointer → CC StdVec Lowering (Entrypoint Only)
=================================================================

Rewrites ``!cc.ptr<!cc.array<T x N>>`` function parameters to
``!cc.stdvec<T>`` for CUDA-Q runtime compatibility.

**Only the entrypoint function** (marked with ``cudaq.entrypoint = "true"``)
is rewritten, because only the entrypoint receives arguments from the Python
runtime via CUDA-Q's marshaling layer (which uses ``!cc.stdvec<T>``).

Internal helper functions called from the entrypoint receive their array
arguments as raw ``!cc.ptr<!cc.array<T x N>>`` pointers directly and do
NOT need stdvec conversion.

This pass:
1. Changes the entrypoint's parameter type to ``!cc.stdvec<T>``.
2. Inserts a ``cc.stdvec_data`` op at the top of the entrypoint body
   to extract a raw ``!cc.ptr<!cc.array<T x ?>>`` pointer.
3. Replaces all uses of the original parameter with the extracted pointer.
4. Updates type annotations in any cc.loop ops that carry the pointer
   as a loop-carried value.

This pass operates on the xDSL IR level (not string-based), ensuring
correctness across all uses including inside nested regions (cc.loop,
cc.if, etc.).
"""

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
    """In-place pass: rewrite !cc.ptr<!cc.array<T x N>> params to !cc.stdvec<T>.

    Only processes the entrypoint function (cudaq.entrypoint = "true").
    Internal helper functions are left unchanged.
    """
    for op in list(module.body.blocks[0].ops):
        if op.name == "func.func" and _is_entrypoint(op):
            _process_func(op)


def _is_entrypoint(func_op) -> bool:
    """Check if a function has the cudaq.entrypoint attribute."""
    ep_attr = func_op.attributes.get("cudaq.entrypoint")
    if ep_attr is None:
        return False
    if isinstance(ep_attr, StringAttr):
        return ep_attr.data == "true"
    return bool(ep_attr)


def _process_func(func_op) -> None:
    """Process the entrypoint function: find array ptr params and rewrite them.

    Strategy for each array ptr param:
    1. Insert cc.stdvec_data op that takes the param as input
    2. Replace all OTHER uses of the param with the stdvec_data result
    3. Update cc.loop block args and result types that carried the old ptr type
    4. Mutate the param's type to !cc.stdvec<T> in-place
    """
    if not func_op.body.blocks:
        return

    ftype = func_op.function_type
    block = func_op.body.blocks[0]

    # Identify which args are !cc.ptr<!cc.array<T x N>> with static size
    rewrites = []  # list of (arg_idx, elem_type, static_size)
    for i, inp_type in enumerate(ftype.inputs):
        if (isinstance(inp_type, CcPtrType)
                and isinstance(inp_type.element_type, CcArrayType)):
            arr_type = inp_type.element_type
            elem_type = arr_type.element_type
            size = arr_type.size.value.data
            if size > 0:  # Skip already-dynamic arrays (size == -1)
                rewrites.append((i, elem_type, size))

    if not rewrites:
        return

    # Build new function signature
    new_inputs = list(ftype.inputs)

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

    # Update the function type
    func_op.function_type = FunctionType.from_lists(new_inputs, list(ftype.outputs))


def _update_loop_types_recursive(region: Region, old_type, new_type) -> None:
    """Recursively walk all cc.loop ops and update block arg / result types.

    Any block arg or result that has `old_type` is changed to `new_type`.
    This ensures type annotations in cc.condition, cc.continue, and the
    loop header are consistent with the actual operand types.
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
    """Check if two types are structurally equivalent.

    Compares CcPtrType<CcArrayType<elem, size>> structures.
    """
    if type(t1) != type(t2):
        return False
    if isinstance(t1, CcPtrType) and isinstance(t2, CcPtrType):
        return _types_match(t1.element_type, t2.element_type)
    if isinstance(t1, CcArrayType) and isinstance(t2, CcArrayType):
        return (t1.element_type == t2.element_type and
                t1.size.value.data == t2.size.value.data)
    return t1 == t2


def _set_block_arg_type(arg, new_type) -> None:
    """Mutate a BlockArgument's type in-place.

    xDSL's BlockArgument.type is a read-only property. We access the
    underlying storage attribute to change it without creating a new
    SSA value (which would mangle the name).
    """
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
    """Replace all uses of old_val with new_val, except in except_op.

    This iterates over all uses and directly patches the operand reference.
    """
    for use in list(old_val.uses):
        if use.operation is not except_op:
            use.operation.operands[use.index] = new_val
