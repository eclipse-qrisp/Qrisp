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
PASS 6: CUDA-Q module preparation.

Transforms a Quake+CC xDSL module (output of passes 1–5) into the structure
that CUDA-Q's Module.parse expects:

- Strips module sym_name (anonymous module)
- Renames @main → @__nvqpp__mlirgen__<uuid>
- Removes old-style attrs, strips visibility
- Adds cudaq-entrypoint / cudaq-kernel unit attrs
- Packs multiple return values into !cc.struct
- Synthesizes .run variant (cc.log_output + void return)
- Synthesizes .run.entry
- Injects module-level attributes (llvm.data_layout, quake.mangled_name_map, etc.)

xDSL stores inherent attributes (sym_name, function_type, sym_visibility)
in op.properties (a dict). Discardable attributes (cudaq-kernel, no_this, etc.)
live in op.attributes.
"""

from typing import Literal

from xdsl.dialects import func
from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FunctionType,
    ModuleOp,
    StringAttr,
    UnitAttr,
)
from xdsl.ir import Attribute, Block, Region, SSAValue

from qrisp.jasp.mlir.quake_lowering.cc_dialect import (
    CcLogOutputOp,
    CcInsertValueOp,
    CcStructType,
    CcUndefOp,
)


# ===========================================================================
# Internal helpers
# ===========================================================================


def _find_func_by_name(module: ModuleOp, name: str):
    """Find a func.func with the given symbol name."""
    for op in module.body.block.ops:
        if isinstance(op, func.FuncOp) and op.sym_name.data == name:
            return op
    return None


def _find_entry_return(func_op: func.FuncOp):
    """Find the func.ReturnOp op in the entry block (last op)."""
    entry_block = func_op.body.blocks[0]
    ops_list = list(entry_block.ops)
    if ops_list:
        last_op = ops_list[-1]
        if isinstance(last_op, func.ReturnOp):
            return last_op
    # Fallback: walk all blocks
    for block in func_op.body.blocks:
        for op in block.ops:
            if isinstance(op, func.ReturnOp):
                return op
    return None


# ===========================================================================
# Pass: Strip module name
# ===========================================================================


def _pass_strip_module_name(module: ModuleOp) -> None:
    """Remove the module's sym_name so it prints as `module attributes {...}`."""
    if "sym_name" in module.properties:
        del module.properties["sym_name"]


# ===========================================================================
# Pass: Rename @main
# ===========================================================================


def _pass_rename_main(module: ModuleOp, new_name: str):
    """Rename @main → @new_name. Returns the FuncOp."""
    main_func = _find_func_by_name(module, "main")
    if main_func is None:
        raise ValueError("Could not find @main function in module.")
    main_func.properties["sym_name"] = StringAttr(new_name)
    return main_func


# ===========================================================================
# Pass: Add cudaq function attributes
# ===========================================================================


def _pass_add_func_attrs(
    func_op: func.FuncOp,
    *,
    entrypoint: bool = False,
    kernel: bool = False,
    no_this: bool = False,
    run_types=None,
) -> None:
    """Set CUDA-Q-specific function attributes.

    Also strips old-style attrs and removes visibility.
    """
    # Remove old-style string attrs from passes 1–5
    for key in ("cudaq.kernel", "cudaq.entrypoint"):
        if key in func_op.attributes:
            del func_op.attributes[key]

    # Remove visibility so it doesn't print `public`
    if "sym_visibility" in func_op.properties:
        del func_op.properties["sym_visibility"]

    # Set unit attrs
    if entrypoint:
        func_op.attributes["cudaq-entrypoint"] = UnitAttr()
    if kernel:
        func_op.attributes["cudaq-kernel"] = UnitAttr()
    if no_this:
        func_op.attributes["no_this"] = UnitAttr()
    if run_types is not None:
        func_op.attributes["quake.cudaq_run"] = ArrayAttr(run_types)


# ===========================================================================
# Pass: Pack multiple return values into !cc.struct
# ===========================================================================


def _pass_pack_multi_return(func_op: func.FuncOp):
    """If func returns >1 value, pack into a cc.struct.

    Returns the struct type if packing occurred, else None.
    """
    return_op = _find_entry_return(func_op)
    if return_op is None:
        return None

    operands = list(return_op.operands)
    if len(operands) <= 1:
        return None

    field_types = [v.type for v in operands]
    struct_type = CcStructType("tuple", field_types)
    block = return_op.parent_block()

    # Build: %s = cc.undef; %s1 = cc.insert_value %s[0], %v0; ...
    undef = CcUndefOp(struct_type)
    block.insert_op_before(undef, return_op)

    current = undef.result
    for i, val in enumerate(operands):
        insert = CcInsertValueOp(current, i, val)
        block.insert_op_before(insert, return_op)
        current = insert.result

    # Replace multi-return with single-value return
    new_return = func.ReturnOp(current)
    block.insert_op_before(new_return, return_op)
    block.erase_op(return_op)

    # Update function type
    input_types = list(func_op.function_type.inputs.data)
    func_op.properties["function_type"] = FunctionType.from_lists(input_types, [struct_type])
    return struct_type


# ===========================================================================
# Pass: Synthesize .run variant
# ===========================================================================


def _pass_synthesize_run(module: ModuleOp, source_func: func.FuncOp, run_func_name: str):
    """Create the .run function: clone source, replace return with log_output + void return."""
    source_output_types = list(source_func.function_type.outputs.data)
    input_types = list(source_func.function_type.inputs.data)

    run_func = source_func.clone()
    run_func.properties["sym_name"] = StringAttr(run_func_name)

    # Replace return with cc.log_output + void return
    return_op = _find_entry_return(run_func)
    if return_op is not None:
        block = return_op.parent_block()
        for val in list(return_op.operands):
            block.insert_op_before(CcLogOutputOp(val), return_op)
        block.insert_op_before(func.ReturnOp(), return_op)
        block.erase_op(return_op)

    # Set void return type
    run_func.properties["function_type"] = FunctionType.from_lists(input_types, [])

    # Set attributes
    _pass_add_func_attrs(
        run_func,
        entrypoint=True,
        kernel=True,
        no_this=True,
        run_types=source_output_types if source_output_types else None,
    )

    module.body.block.add_op(run_func)
    return run_func


# ===========================================================================
# Pass: Synthesize .run.entry stub
# ===========================================================================


def _pass_synthesize_run_entry(module: ModuleOp, source_func: func.FuncOp, run_entry_name: str):
    """Create the .run.entry stub: same params, empty body, void return."""
    input_types = list(source_func.function_type.inputs.data)

    entry_block = Block(arg_types=input_types)
    entry_block.add_op(func.ReturnOp())

    entry_func = func.FuncOp(
        run_entry_name,
        FunctionType.from_lists(input_types, []),
        Region([entry_block]),
    )
    entry_func.attributes["no_this"] = UnitAttr()
    if "sym_visibility" in entry_func.properties:
        del entry_func.properties["sym_visibility"]

    module.body.block.add_op(entry_func)
    return entry_func


# ===========================================================================
# Pass: Sample mode — strip returns
# ===========================================================================


def _pass_strip_returns(func_op: func.FuncOp) -> None:
    """Replace func.ReturnOp %vals with void func.ReturnOp. Update signature."""
    return_op = _find_entry_return(func_op)
    if return_op is not None and list(return_op.operands):
        block = return_op.parent_block()
        block.insert_op_before(func.ReturnOp(), return_op)
        block.erase_op(return_op)

    input_types = list(func_op.function_type.inputs.data)
    func_op.properties["function_type"] = FunctionType.from_lists(input_types, [])


# ===========================================================================
# Pass: Inject module-level attributes
# ===========================================================================


def _pass_inject_module_attrs(
    module: ModuleOp,
    *,
    func_name: str,
    entry_point: str,
    uniq_name: str,
    data_layout: str,
    target_triple,
    run_func_name=None,
    run_entry_name=None,
) -> None:
    """Set module-level attributes required by CUDA-Q."""
    module.attributes["cc.python_uniqued"] = StringAttr(uniq_name)
    module.attributes["llvm.data_layout"] = StringAttr(data_layout)
    if target_triple:
        module.attributes["llvm.target_triple"] = StringAttr(target_triple)

    name_map = {func_name: StringAttr(entry_point)}
    if run_func_name and run_entry_name:
        name_map[run_func_name] = StringAttr(run_entry_name)
    module.attributes["quake.mangled_name_map"] = DictionaryAttr(name_map)


# ===========================================================================
# Orchestrator
# ===========================================================================


def prepare_module_for_cudaq(
    module: ModuleOp,
    *,
    func_name: str,
    entry_point: str,
    uniq_name: str,
    data_layout: str,
    target_triple,
    execution_mode: Literal["run", "sample"] = "run",
) -> None:
    """Apply all CUDA-Q preparation passes to the module in-place.

    Parameters
    ----------
    module : ModuleOp
        xDSL module containing a @main function.
    func_name : str
        Target function name (e.g. "__nvqpp__mlirgen__<uuid>").
    entry_point : str
        CUDA-Q entry point name.
    uniq_name : str
        Unique kernel name for CUDA-Q registration.
    data_layout : str
        LLVM data layout string.
    target_triple : str | None
        LLVM target triple string.
    execution_mode : "run" | "sample"
        Whether to prepare for cudaq.run or cudaq.sample.
    """
    _pass_strip_module_name(module)
    main_func = _pass_rename_main(module, func_name)

    if execution_mode == "sample":
        _pass_add_func_attrs(main_func, entrypoint=True, kernel=True)
        _pass_strip_returns(main_func)
        _pass_inject_module_attrs(
            module,
            func_name=func_name,
            entry_point=entry_point,
            uniq_name=uniq_name,
            data_layout=data_layout,
            target_triple=target_triple,
        )

    elif execution_mode == "run":
        _pass_pack_multi_return(main_func)
        _pass_add_func_attrs(main_func, entrypoint=True, kernel=True)

        run_func_name = func_name + ".run"
        run_entry_name = func_name + ".run.entry"

        _pass_synthesize_run(module, main_func, run_func_name)
        _pass_synthesize_run_entry(module, main_func, run_entry_name)
        _pass_inject_module_attrs(
            module,
            func_name=func_name,
            entry_point=entry_point,
            uniq_name=uniq_name,
            data_layout=data_layout,
            target_triple=target_triple,
            run_func_name=run_func_name,
            run_entry_name=run_entry_name,
        )

    else:
        raise ValueError(f"Unknown execution_mode: {execution_mode!r}")
