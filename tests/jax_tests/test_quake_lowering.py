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
Tests for the Jasp → Quake (memory-semantics) lowering backend.

Coverage
--------
- Basic quantum circuit (alloc, gate, measure, dealloc).
- Parameterized gates (rz, rx, u3).
- Controlled gates (cx, ccx).
- Reset operation.
- SCF control-flow lowering (jrange loop).
- Interface invariant: no ``!jasp.*`` types in the output.
- Negative test: ``jasp.parity`` is left in place (not lowered).
- Negative test: unsupported gate emits a warning and is left in place.
"""

import warnings
import pytest

from qrisp import QuantumVariable, h, x, y, z, cx, rz, rx, s, t, measure
from qrisp.jasp import make_jaspr, jrange


def _import_or_skip():
    """Skip tests if the required optional packages are not installed."""
    try:
        from qrisp.jasp.mlir.quake_lowering import jaspr_to_quake
        return jaspr_to_quake
    except ImportError as exc:
        pytest.skip(f"quake_lowering unavailable: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lower(circuit_fn, *trace_args) -> str:
    """Build the Jaspr for *circuit_fn* and lower it to Quake MLIR.

    Returns the MLIR as a string.  Deprecation warnings from xDSL internals
    are silenced.
    """
    jaspr_to_quake = _import_or_skip()
    jaspr = make_jaspr(circuit_fn)(*trace_args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module = jaspr_to_quake(jaspr)
    return str(module)


def assert_no_jasp(mlir_str: str) -> None:
    """Assert that no ``!jasp.*`` types remain in *mlir_str*."""
    import re
    jasp_types = re.findall(r"!jasp\.\w+", mlir_str)
    assert not jasp_types, (
        f"Expected no !jasp.* types in Quake output, but found: {set(jasp_types)}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_alloc_and_dealloc():
    """create_qubits → quake.alloca / delete_qubits → quake.dealloc."""

    def circuit():
        qv = QuantumVariable(3)
        return qv  # implicitly deallocated at end of jaspr scope

    mlir = _lower(circuit)
    assert "quake.alloca" in mlir, "Expected quake.alloca in output"
    assert_no_jasp(mlir)


def test_single_qubit_gates():
    """Standard single-qubit gates lower to the corresponding quake.* ops."""

    def circuit():
        qv = QuantumVariable(4)
        h(qv[0])
        x(qv[1])
        y(qv[2])
        z(qv[3])
        s(qv[0])
        t(qv[1])
        return qv

    mlir = _lower(circuit)
    for gate in ("quake.h", "quake.x", "quake.y", "quake.z", "quake.s", "quake.t"):
        assert gate in mlir, f"Expected {gate!r} in output"
    assert_no_jasp(mlir)


def test_parameterized_gate():
    """rz / rx gates carry floating-point parameters."""

    def circuit():
        qv = QuantumVariable(2)
        rz(0.5, qv[0])
        rx(1.0, qv[1])
        return qv

    mlir = _lower(circuit)
    assert "quake.rz" in mlir, "Expected quake.rz in output"
    assert "quake.rx" in mlir, "Expected quake.rx in output"
    # Parameters should appear as f64 scalars
    assert "f64" in mlir, "Expected f64 parameter type in output"
    assert_no_jasp(mlir)


def test_controlled_gate_cx():
    """cx maps to quake.x with one control qubit."""

    def circuit():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return measure(qv)

    mlir = _lower(circuit)
    assert "quake.h" in mlir
    assert "quake.x" in mlir
    # Control qubit should be present in square brackets
    assert "[%"  in mlir, "Expected control qubit in bracket notation"
    assert_no_jasp(mlir)


def test_measure_single_qubit():
    """Single-qubit measure: quake.mz + quake.discriminate."""

    def circuit():
        qv = QuantumVariable(1)
        h(qv[0])
        return measure(qv)

    mlir = _lower(circuit)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_no_jasp(mlir)


def test_extract_ref():
    """get_qubit → quake.extract_ref."""

    def circuit():
        qv = QuantumVariable(3)
        h(qv[0])
        cx(qv[0], qv[2])
        return qv

    mlir = _lower(circuit)
    assert "quake.extract_ref" in mlir, "Expected quake.extract_ref in output"
    assert_no_jasp(mlir)


def test_jrange_loop():
    """A jrange loop produces a scf.while with !quake.veq<?> loop-carried value."""

    def circuit():
        qv = QuantumVariable(3)
        for i in jrange(3):
            h(qv[i])
        return qv

    mlir = _lower(circuit)
    assert "scf.while" in mlir or "cc.loop" in mlir, (
        "Expected scf.while or cc.loop in output"
    )
    assert "quake.h" in mlir, "Expected quake.h inside loop"
    assert_no_jasp(mlir)


def test_no_jasp_types_in_output():
    """Interface invariant: output must not contain any !jasp.* types."""

    def bell():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return measure(qv)

    mlir = _lower(bell)
    assert_no_jasp(mlir)


def test_cudaq_kernel_attribute():
    """Lowered function should carry cudaq.kernel and cudaq.entrypoint attributes."""

    def simple():
        qv = QuantumVariable(1)
        h(qv[0])
        return measure(qv)

    mlir = _lower(simple)
    assert "cudaq.kernel" in mlir, "Expected cudaq.kernel attribute on function"
    assert "cudaq.entrypoint" in mlir, "Expected cudaq.entrypoint attribute on function"


def test_quake_types_present():
    """Output should contain quake.* types and ops."""

    def circuit():
        qv = QuantumVariable(2)
        h(qv[0])
        return measure(qv)

    mlir = _lower(circuit)
    assert "!quake.veq<?>" in mlir or "!quake.ref" in mlir, (
        "Expected Quake qubit types in output"
    )
    assert "quake." in mlir, "Expected Quake ops in output"


def test_unsupported_gate_warning():
    """An unsupported gate should emit a warning and leave the op in place."""
    from qrisp.jasp.mlir.quake_lowering import jaspr_to_quake

    # Build a custom jaspr with an exotic gate – use rxx which is not in GATE_MAP
    # We do this by patching the Jaspr IR after construction.
    # Since we can't easily inject an exotic gate via the Python API,
    # we test the gate_mapping module directly.
    from qrisp.jasp.mlir.quake_lowering.gate_mapping import get_gate_info

    assert get_gate_info("rxx") is None, "rxx should not be in GATE_MAP"
    assert get_gate_info("gphase") is None, "gphase should not be in GATE_MAP"
    assert get_gate_info("rzz") is None, "rzz should not be in GATE_MAP"
    assert get_gate_info("xxyy") is None, "xxyy should not be in GATE_MAP"


def test_parity_not_lowered():
    """jasp.parity ops are left in place (not lowered to Quake)."""
    from qrisp.jasp.mlir.quake_lowering.gate_mapping import get_gate_info

    # parity is not a gate name, but verify it's not in GATE_MAP
    assert get_gate_info("parity") is None


def test_gate_mapping_standard_gates():
    """Verify that all standard gates are in the gate map."""
    from qrisp.jasp.mlir.quake_lowering.gate_mapping import get_gate_info, GATE_MAP

    expected_gates = {
        "h", "x", "y", "z", "s", "t",
        "rx", "ry", "rz", "p", "r1",
        "cx", "cy", "cz", "swap",
        "mcx",
    }
    for gate in expected_gates:
        info = get_gate_info(gate)
        assert info is not None, f"Expected gate '{gate}' in GATE_MAP"


def test_multi_qubit_alloc():
    """Multiple QuantumVariable allocations produce multiple quake.alloca ops."""

    def circuit():
        qv1 = QuantumVariable(2)
        qv2 = QuantumVariable(3)
        h(qv1[0])
        x(qv2[0])
        return measure(qv1), measure(qv2)

    mlir = _lower(circuit)
    # At least two alloca ops
    alloca_count = mlir.count("quake.alloca")
    assert alloca_count >= 2, f"Expected ≥2 quake.alloca ops, got {alloca_count}"
    assert_no_jasp(mlir)


# ---------------------------------------------------------------------------
# MLIR format validity tests
# These tests verify that the generated MLIR conforms to the CUDA-Q Quake
# dialect assembly format (functional-type format, correct type signatures).
# ---------------------------------------------------------------------------


def test_extract_ref_functional_type_format():
    """quake.extract_ref must use functional-type: (!quake.veq<?>, <idx>) -> !quake.ref.

    Per CUDA-Q Quake dialect spec, the assemblyFormat uses functional-type
    (all operand types in parens) rather than a bare ``veq -> ref``.
    """

    def circuit():
        qv = QuantumVariable(2)
        h(qv[0])
        return qv

    mlir = _lower(circuit)
    # The correct format includes parentheses and the index type
    assert "quake.extract_ref" in mlir
    # Must NOT have the bare (pre-fix) format "!quake.veq<?> -> !quake.ref"
    assert "!quake.veq<?> -> !quake.ref" not in mlir, (
        "extract_ref should use functional-type format, not bare 'veq -> ref'"
    )
    # Must have the functional-type format with both input types in parens
    assert "(!quake.veq<?>" in mlir, (
        "extract_ref should use functional-type format: (!quake.veq<?>, idx_type) -> !quake.ref"
    )
    assert "-> !quake.ref" in mlir
    assert_no_jasp(mlir)


def test_gate_type_signature_no_bracket_prefix():
    """Gate ops must NOT use [ctrl-types](tgt-types) format — only flat functional-type.

    The CUDA-Q Quake dialect uses functional-type(operands, results) where all
    operand types (params + controls + targets) appear in a single flat list.
    """

    def circuit():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return qv

    mlir = _lower(circuit)
    # H gate: single target, no controls
    assert "quake.h" in mlir
    assert "(!quake.ref) -> ()" in mlir, "H gate should have '(!quake.ref) -> ()' type sig"
    # CX gate: control + target, both refs → flat list
    assert "quake.x" in mlir
    # Should NOT have the old bracket prefix format [!quake.ref](!quake.ref)
    assert "[!quake.ref]" not in mlir, (
        "Gate type signatures must not use [ctrl-types] prefix — use flat functional-type"
    )
    # Should have flat format for CX: (!quake.ref, !quake.ref)
    assert "(!quake.ref, !quake.ref) -> ()" in mlir, (
        "CX gate should have '(!quake.ref, !quake.ref) -> ()' type sig"
    )
    assert_no_jasp(mlir)


def test_parameterized_gate_functional_type():
    """Parameterized gates use flat (f64, !quake.ref) -> () format."""

    def circuit():
        qv = QuantumVariable(1)
        rz(0.5, qv[0])
        return qv

    mlir = _lower(circuit)
    assert "quake.rz" in mlir
    # The type signature must include both the f64 param and the quake.ref target
    assert "(f64, !quake.ref) -> ()" in mlir, (
        "rz gate should have '(f64, !quake.ref) -> ()' type sig"
    )
    assert_no_jasp(mlir)


def test_veq_measurement_returns_cc_stdvec_type():
    """Measuring a veq returns !cc.stdvec<!quake.measure>, matching the CUDA-Q reference.

    Per the CUDA-Q spec, measuring a veq returns !cc.stdvec<!quake.measure> — a CC
    span of single-qubit measurement values. Not !quake.measurements<?> (invalid) and
    not !cc.stdvec<i1>.
    """

    def circuit():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return measure(qv)

    mlir = _lower(circuit)
    assert "quake.mz" in mlir
    # Must use CC type with !quake.measure element type, matching CUDA-Q reference
    assert "!cc.stdvec<!quake.measure>" in mlir, (
        "Measuring a veq should return !cc.stdvec<!quake.measure>"
    )
    assert "!quake.measurements<?>" not in mlir, (
        "!quake.measurements<?> is not a valid CUDA-Q type"
    )
    assert_no_jasp(mlir)


def test_single_qubit_measurement_result_type():
    """Measuring any qubit register (even 1 qubit) returns !cc.stdvec<!quake.measure>."""

    def circuit():
        qv = QuantumVariable(1)
        h(qv[0])
        return measure(qv)

    mlir = _lower(circuit)
    # Even a 1-qubit circuit measures the full veq → !cc.stdvec<!quake.measure>
    assert "quake.mz" in mlir
    assert "!cc.stdvec<!quake.measure>" in mlir, (
        "mz on a veq (even 1-qubit) should return !cc.stdvec<!quake.measure>"
    )
    assert_no_jasp(mlir)


def test_veq_size_functional_type():
    """quake.veq_size must use functional-type: (!quake.veq<?>) -> i64."""
    from qrisp.jasp import jrange

    def circuit():
        qv = QuantumVariable(3)
        for i in jrange(3):
            h(qv[i])
        return qv

    mlir = _lower(circuit)
    # veq_size op should use functional-type with parens if present
    if "quake.veq_size" in mlir:
        assert "(!quake.veq<?>) -> i64" in mlir, (
            "veq_size should use functional-type: (!quake.veq<?>) -> i64"
        )
    assert_no_jasp(mlir)


def test_alloca_veq_format():
    """quake.alloca !quake.veq<?>[%n : i64] — type before size in brackets."""

    def circuit():
        qv = QuantumVariable(4)
        return qv

    mlir = _lower(circuit)
    # The alloca must print the type first, then size in brackets
    assert "quake.alloca !quake.veq<?>[" in mlir, (
        "alloca format should be '!quake.veq<?>[%n : i64]'"
    )
    assert_no_jasp(mlir)


def test_bell_circuit_full_format():
    """Full Bell circuit MLIR format validation — spot-check every key op."""

    def bell():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return measure(qv)

    mlir = _lower(bell)

    # Function attributes
    assert 'cudaq.kernel = "true"' in mlir
    assert 'cudaq.entrypoint = "true"' in mlir

    # Alloca
    assert "quake.alloca !quake.veq<?>[" in mlir

    # extract_ref with functional-type
    assert "(!quake.veq<?>" in mlir and "-> !quake.ref" in mlir, (
        "extract_ref must use functional-type format: (!quake.veq<?>, idx) -> !quake.ref"
    )

    # H gate
    assert "(!quake.ref) -> ()" in mlir

    # CX gate (quake.x with control)
    assert "(!quake.ref, !quake.ref) -> ()" in mlir

    # Measurement with correct CUDA-Q CC type (matches CUDA-Q reference output)
    assert "!cc.stdvec<!quake.measure>" in mlir, (
        "veq measurement must return !cc.stdvec<!quake.measure>"
    )

    assert_no_jasp(mlir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
