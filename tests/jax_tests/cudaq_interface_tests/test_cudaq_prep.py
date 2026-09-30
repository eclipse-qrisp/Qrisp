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

"""Tests for the CUDA-Q module preparation stage (cudaq_prep)."""

import warnings

import pytest
from xdsl.dialects import func as xfunc

from qrisp import QuantumFloat, QuantumVariable, cx, h, measure, x
from qrisp.jasp import make_jaspr, qache
from qrisp.jasp.cudaq_interface.cudaq_ingestion.cudaq_prep import (
    _CudaqPreparationConfig,
    _prepare_module_for_cudaq,
)
from qrisp.jasp.cudaq_interface.quake_lowering.jaspr_to_quake import _jaspr_to_quake_mlir
from qrisp.jasp.interpreter_tools import jaspr_to_static_register_jaspr

FUNC_NAME = "__nvqpp__mlirgen__probe"
ENTRY_POINT = "probe.entry"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare(circuit_fn, execution_mode="run", register_size=None):
    """Lower *circuit_fn* and run the CUDA-Q preparation passes over it.

    ``register_size`` mirrors the @cudaq_kernel argument of the same name and
    defaults to None, i.e. no static-register conversion. That matters here:
    converting collapses qached subroutines into classical tracer helpers, so
    a harness that always converted would never see a quantum callee.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jaspr = make_jaspr(circuit_fn)()
        if register_size is not None:
            jaspr = jaspr_to_static_register_jaspr(jaspr, register_size)
        module = _jaspr_to_quake_mlir(jaspr, execution_mode=execution_mode)
        _prepare_module_for_cudaq(
            module,
            _CudaqPreparationConfig(
                func_name=FUNC_NAME,
                entry_point=ENTRY_POINT,
                unique_name="probe",
                execution_mode=execution_mode,
            ),
        )
    return module


def _funcs(module):
    """Map symbol name -> FuncOp for every function in the module."""
    return {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, xfunc.FuncOp)}


@qache
def _subroutine(qv):
    """A qached helper, lowered to a sibling func.func called via func.call."""
    h(qv[0])
    cx(qv[0], qv[1])
    return 42


def _with_callee():
    qv = QuantumFloat(3)
    value = _subroutine(qv)
    return measure(qv) + value


def _simple():
    qv = QuantumVariable(2)
    h(qv[0])
    cx(qv[0], qv[1])
    return measure(qv)


# ---------------------------------------------------------------------------
# Callees are left untouched
# ---------------------------------------------------------------------------


def test_callee_keeps_private_visibility():
    """Visibility is only stripped from entry points, matching CUDA-Q's own output."""
    funcs = _funcs(_prepare(_with_callee))

    callees = {name: f for name, f in funcs.items() if not name.startswith(FUNC_NAME)}
    assert callees, f"no callee to test; module held only {sorted(funcs)}"

    for name, func_op in callees.items():
        visibility = func_op.properties.get("sym_visibility")
        assert visibility is not None, f"{name} lost its visibility"
        assert visibility.data == "private", f"{name} is {visibility.data}, expected private"


# ---------------------------------------------------------------------------
# Structural synthesis
# ---------------------------------------------------------------------------


def test_module_name_is_stripped():
    """CUDA-Q parses an anonymous module, so @jasp_module must not survive."""
    module = _prepare(_simple)

    assert "sym_name" not in module.properties
    assert str(module).lstrip().startswith("builtin.module attributes")


def test_multiple_return_values_are_packed_into_a_struct():
    """CUDA-Q entry points return at most one value, so a tuple becomes a cc.struct."""

    def two_results():
        qv1 = QuantumVariable(2)
        qv2 = QuantumVariable(2)
        h(qv1[0])
        x(qv2[0])
        return measure(qv1), measure(qv2)

    main_func = _funcs(_prepare(two_results))[FUNC_NAME]

    outputs = list(main_func.function_type.outputs.data)
    assert len(outputs) == 1, f"expected a single packed result, got {len(outputs)}"
    assert "cc.struct" in str(outputs[0])
    assert "cc.undef" in str(main_func)
    assert "cc.insert_value" in str(main_func)


def test_run_mode_synthesizes_run_variants():
    """Run mode adds .run (void, logs outputs) and an empty .run.entry stub."""
    funcs = _funcs(_prepare(_simple, "run"))

    run_func = funcs[FUNC_NAME + ".run"]
    run_entry = funcs[FUNC_NAME + ".run.entry"]

    assert len(run_func.function_type.outputs.data) == 0
    assert "no_this" in run_func.attributes
    assert "quake.cudaq_run" in run_func.attributes
    assert "quake.log_output" in str(run_func)

    assert len(run_entry.function_type.outputs.data) == 0
    assert "no_this" in run_entry.attributes
    assert "sym_visibility" not in run_entry.properties


def test_run_mode_rejects_quake_return():
    """Run mode must reject a kernel returning a quantum register."""

    def returns_qubits():
        qv = QuantumVariable(3)
        x(qv[0])
        return qv

    with pytest.raises(ValueError, match="must return only classical values"):
        _prepare(returns_qubits, "run")


def test_run_mode_rejects_quake_type_nested_in_struct():
    """Run mode must reject Quake values inside packed return types."""

    def returns_qubits_and_classical_value():
        qv = QuantumVariable(3)
        x(qv[0])
        return qv, measure(qv[0])

    with pytest.raises(ValueError, match="must return only classical values"):
        _prepare(returns_qubits_and_classical_value, "run")


def test_sample_mode_strips_returns_and_adds_no_run_variant():
    """Sample mode voids the entry point and synthesizes no .run functions."""
    funcs = _funcs(_prepare(_simple, "sample"))

    assert len(funcs[FUNC_NAME].function_type.outputs.data) == 0
    assert FUNC_NAME + ".run" not in funcs
    assert FUNC_NAME + ".run.entry" not in funcs


def test_unique_name_uses_cudaqs_own_attribute():
    """The unique kernel name must go under the name CUDA-Q reads it from.

    CUDA-Q spells this quake.python_uniqued (cudaq.kernel.utils
    cudaq__unique_attr_name); anything else is an attribute it never looks at.
    """
    module = _prepare(_simple)

    assert "quake.python_uniqued" in module.attributes
    assert module.attributes["quake.python_uniqued"].data == "probe"


def test_mangled_name_map_covers_exactly_the_entry_points():
    """The name map lists @main and .run in run mode, and only @main in sample mode."""
    run_map = _prepare(_simple, "run").attributes["quake.mangled_name_map"].data
    assert set(run_map) == {FUNC_NAME, FUNC_NAME + ".run"}

    sample_map = _prepare(_simple, "sample").attributes["quake.mangled_name_map"].data
    assert set(sample_map) == {FUNC_NAME}


def test_unknown_execution_mode_raises():
    """An unrecognized execution_mode is rejected rather than silently ignored."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module = _jaspr_to_quake_mlir(make_jaspr(_simple)())

    with pytest.raises(ValueError, match="Unknown execution_mode"):
        _prepare_module_for_cudaq(
            module,
            _CudaqPreparationConfig(
                func_name=FUNC_NAME,
                entry_point=ENTRY_POINT,
                unique_name="probe",
                execution_mode="bogus",
            ),
        )
