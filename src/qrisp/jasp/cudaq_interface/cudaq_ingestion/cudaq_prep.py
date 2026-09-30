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

"""Prepare lowered xDSL modules for execution by CUDA-Q."""

# CUDA-Q module preparation.
# ==========================
#
# Transforms a Quake+CC xDSL module (output of _jaspr_to_quake_mlir) into the structure
# that CUDA-Q's Module.parse expects:
#
# - Strips module sym_name (anonymous module)
# - Renames @main -> @__nvqpp__mlirgen__<uuid>
# - Packs multiple return values into !cc.struct
# - Synthesizes .run variant (quake.log_output + void return)
# - Synthesizes .run.entry
# - Injects module-level attributes (quake.mangled_name_map, etc.)
# - Rejects Quake-valued returns for the .run variant

from dataclasses import dataclass
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
from xdsl.ir import Attribute, Block, Region

from qrisp.jasp.cudaq_interface.quake_lowering.dialects.cc_dialect import (
    CcInsertValueOp,
    CcStructType,
    CcUndefOp,
)
from qrisp.jasp.cudaq_interface.quake_lowering.dialects.quake_dialect import (
    QuakeLogOutputOp,
    QuakeMeasureType,
    QuakeRefType,
    QuakeVeqType,
)

# ===========================================================================
# Internal helpers
# ===========================================================================


@dataclass(frozen=True)
class _CudaqPreparationConfig:
    """Configuration shared by the CUDA-Q module preparation steps."""

    func_name: str
    entry_point: str
    unique_name: str
    execution_mode: Literal["run", "sample"] = "run"


def _find_func_by_name(module: ModuleOp, name: str):
    """Find a func.func with the given symbol name."""
    for op in module.body.block.ops:
        if isinstance(op, func.FuncOp) and op.sym_name.data == name:
            return op
    return None


def _find_entry_return(func_op: func.FuncOp):
    """Return the func.ReturnOp terminating the entry block, if there is one.

    Every block must end in a terminator, and the lowering only ever emits
    structured control flow (``cc.loop``/``cc.scope``, which carry their own
    terminators), so a ``func.return`` can only appear as the last op of the
    function's entry block.
    """
    ops_list = list(func_op.body.blocks[0].ops)
    if ops_list and isinstance(ops_list[-1], func.ReturnOp):
        return ops_list[-1]
    return None


def _set_result_types(func_op: func.FuncOp, result_types: list[Attribute]) -> None:
    """Rewrite the function's signature to return *result_types*, keeping its inputs."""
    input_types = list(func_op.function_type.inputs.data)
    func_op.properties["function_type"] = FunctionType.from_lists(input_types, result_types)


def _rewrite_return_as_log_output(func_op: func.FuncOp) -> None:
    """Make the function void-returning, logging each value it used to return.

    ``quake.log_output`` is the mechanism CUDA-Q uses to capture and aggregate
    per-shot results, so this is how a returned value survives ``cudaq.run``.
    """
    return_op = _find_entry_return(func_op)
    if return_op is not None:
        block = return_op.parent_block()
        for val in list(return_op.operands):
            block.insert_op_before(QuakeLogOutputOp(val), return_op)
        block.insert_op_before(func.ReturnOp(), return_op)
        block.erase_op(return_op)

    _set_result_types(func_op, [])


def _contains_quake_type(attribute: Attribute) -> bool:
    """Return whether an attribute is a Quake type, including packed fields."""
    if isinstance(attribute, (QuakeMeasureType, QuakeRefType, QuakeVeqType)):
        return True
    if isinstance(attribute, CcStructType):
        return any(_contains_quake_type(field_type) for field_type in attribute.field_types.data)
    return False


def _validate_run_return_types(func_op: func.FuncOp) -> None:
    """Reject Quake-valued returns, which CUDA-Q cannot expose through ``run``."""
    for result_type in func_op.function_type.outputs.data:
        if _contains_quake_type(result_type):
            raise ValueError(
                "Kernels used with CUDA-Q run mode must return only classical values; a quantum value was returned."
            )


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
# Pass: Pack multiple return values into !cc.struct
# ===========================================================================


def _pass_pack_multi_return(func_op: func.FuncOp) -> None:
    """If func returns >1 value, pack them into a single cc.struct."""
    return_op = _find_entry_return(func_op)
    if return_op is None:
        return

    operands = list(return_op.operands)
    if len(operands) <= 1:
        return

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

    _set_result_types(func_op, [struct_type])


# ===========================================================================
# Pass: Synthesize .run variant
# ===========================================================================


def _pass_synthesize_run(module: ModuleOp, source_func: func.FuncOp, run_func_name: str) -> None:
    """Create the .run function: clone source, replace return with log_output + void return."""
    source_output_types = list(source_func.function_type.outputs.data)
    _validate_run_return_types(source_func)

    run_func = source_func.clone()
    run_func.properties["sym_name"] = StringAttr(run_func_name)

    _rewrite_return_as_log_output(run_func)

    # cudaq-entrypoint / cudaq-kernel and the stripped visibility come along
    # with the clone; only these two are specific to the .run variant.
    run_func.attributes["no_this"] = UnitAttr()
    if source_output_types:
        run_func.attributes["quake.cudaq_run"] = ArrayAttr(source_output_types)

    module.body.block.add_op(run_func)


# ===========================================================================
# Pass: Synthesize .run.entry stub
# ===========================================================================


def _pass_synthesize_run_entry(module: ModuleOp, source_func: func.FuncOp, run_entry_name: str) -> None:
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

    module.body.block.add_op(entry_func)


# ===========================================================================
# Pass: Inject module-level attributes
# ===========================================================================


def _pass_inject_module_attrs(
    module: ModuleOp,
    config: _CudaqPreparationConfig,
    run_func_name=None,
    run_entry_name=None,
) -> None:
    """Set the module-level attributes CUDA-Q registers the kernel under.

    ``quake.mangled_name_map`` pairs each entry point with the symbol CUDA-Q
    launches it through, including the ``.run`` variant when one was
    synthesized.
    """
    module.attributes["quake.python_uniqued"] = StringAttr(config.unique_name)

    name_map = {config.func_name: StringAttr(config.entry_point)}
    if run_func_name and run_entry_name:
        name_map[run_func_name] = StringAttr(run_entry_name)
    module.attributes["quake.mangled_name_map"] = DictionaryAttr(name_map)


# ===========================================================================
# Orchestrator
# ===========================================================================


def _prepare_module_for_cudaq(
    module: ModuleOp,
    config: _CudaqPreparationConfig,
) -> None:
    """Apply all CUDA-Q preparation passes to the module in-place.

    Parameters
    ----------
    module : ModuleOp
        xDSL module containing a @main function.
    config : _CudaqPreparationConfig
        Kernel names and execution mode for the preparation.

    """
    _pass_strip_module_name(module)
    main_func = _pass_rename_main(module, config.func_name)

    if config.execution_mode == "sample":
        _pass_inject_module_attrs(module, config)

    elif config.execution_mode == "run":
        _pass_pack_multi_return(main_func)

        run_func_name = config.func_name + ".run"
        run_entry_name = config.func_name + ".run.entry"

        _pass_synthesize_run(module, main_func, run_func_name)
        _pass_synthesize_run_entry(module, main_func, run_entry_name)
        _pass_inject_module_attrs(module, config, run_func_name, run_entry_name)

    else:
        raise ValueError(f"Unknown execution_mode: {config.execution_mode!r}. Supported: 'run', 'sample'.")
