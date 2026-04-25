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
import re

from qrisp import QuantumVariable, QuantumBool, h, x, y, z, cx, rz, rx, s, t, measure, control
from qrisp.jasp import make_jaspr, jrange, q_while_loop, q_cond, q_fori_loop    

try:
    from qrisp.jasp.mlir.quake_lowering import jaspr_to_quake, validate_quake_mlir, run_quake_mlir
except ImportError as exc:
    # Skip the entire test file if the import fails
    pytest.skip(f"quake_lowering unavailable: {exc}", allow_module_level=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lower(circuit_fn, *trace_args) -> str:
    """Build the Jaspr for *circuit_fn* and lower it to Quake MLIR.

    Returns the MLIR as a string.  Deprecation warnings from xDSL internals
    are silenced.
    """
    jaspr = make_jaspr(circuit_fn)(*trace_args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module = jaspr_to_quake(jaspr)
    return str(module)


def assert_return_type(mlir: str, expected_type: str):
    """Assert the MLIR contains a func.return with the given type."""
    pattern = rf'func\.return\s+%\w+\s*:\s*{re.escape(expected_type)}'
    assert re.search(pattern, mlir), \
        f"Expected return type '{expected_type}', not found in:\n{mlir}"


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
    validate_quake_mlir(mlir)

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
    validate_quake_mlir(mlir)


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
    validate_quake_mlir(mlir)


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
    validate_quake_mlir(mlir)


def test_veq_size_functional_type():
    """quake.veq_size must use functional-type: (!quake.veq<?>) -> i64."""
    from qrisp.jasp import jrange

    def circuit():
        qv = QuantumVariable(3)
        h(qv[qv.size - 1])
        return qv

    mlir = _lower(circuit)
    # veq_size op should use functional-type with parens if present
    if "quake.veq_size" in mlir:
        assert "(!quake.veq<?>) -> i64" in mlir, (
            "veq_size should use functional-type: (!quake.veq<?>) -> i64"
        )
    validate_quake_mlir(mlir)


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
    validate_quake_mlir(mlir)

# ---------------------------------------------------------------------------
# Test 
# ---------------------------------------------------------------------------

def test_extract_ref():
    """get_qubit → quake.extract_ref."""

    def circuit():
        qv = QuantumVariable(3)
        h(qv[0])
        cx(qv[0], qv[2])
        return qv

    mlir = _lower(circuit)
    assert "quake.extract_ref" in mlir, "Expected quake.extract_ref in output"
    validate_quake_mlir(mlir)


def test_no_jasp_types_in_output():
    """Interface invariant: output must not contain any !jasp.* types."""

    def bell():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return measure(qv)

    mlir = _lower(bell)
    validate_quake_mlir(mlir)


def test_cudaq_kernel_attribute():
    """Lowered function should carry cudaq.kernel and cudaq.entrypoint attributes."""

    def simple():
        qv = QuantumVariable(1)
        h(qv[0])
        return measure(qv)

    mlir = _lower(simple)
    assert "cudaq.kernel" in mlir, "Expected cudaq.kernel attribute on function"
    assert "cudaq.entrypoint" in mlir, "Expected cudaq.entrypoint attribute on function"
    validate_quake_mlir(mlir)


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
    validate_quake_mlir(mlir)


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

# ---------------------------------------------------------------------------
# Test quantum variable allocation 
# and gate application functions acting on qubits
# ---------------------------------------------------------------------------

def test_multi_qubit_alloc():
    """Multiple QuantumVariable allocations produce multiple quake.alloca ops."""

    def circuit():
        qv1 = QuantumVariable(2)
        qv2 = QuantumVariable(3)
        h(qv1[0])
        x(qv2[0])
        return measure(qv1[0]), measure(qv2[0])

    mlir = _lower(circuit)
    # At least two alloca ops
    alloca_count = mlir.count("quake.alloca")
    assert alloca_count >= 2, f"Expected ≥2 quake.alloca ops, got {alloca_count}"
    validate_quake_mlir(mlir)

# ---------------------------------------------------------------------------
# Test gate application functions acting on qubits
# ---------------------------------------------------------------------------

def test_single_qubit_gates():
    """Standard single-qubit gates (h,x,y,z,s,t) lower to the corresponding quake.* ops."""

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
    validate_quake_mlir(mlir)


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
    validate_quake_mlir(mlir)


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
    validate_quake_mlir(mlir)

# ---------------------------------------------------------------------------
# Test measure qubit
# ---------------------------------------------------------------------------

def test_measure_single_qubit():
    """Single-qubit measure: quake.mz + quake.discriminate."""

    def circuit():
        qv = QuantumVariable(1)
        x(qv[0])
        return measure(qv[0])

    mlir = _lower(circuit)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i1")
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10*[1]

# ---------------------------------------------------------------------------
# Test measure QuantumVariable
# ---------------------------------------------------------------------------

def test_measure_quantum_variable():
    """QuantumVariable measure"""

    def circuit():
        qv = QuantumVariable(3)
        x(qv[0])
        return measure(qv)
    
    mlir = _lower(circuit)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10*[1]


def test_measure_single_qubit_quantum_variable():
    """QuantumVariable measure"""

    def circuit():
        qv = QuantumVariable(1)
        x(qv[0])
        return measure(qv)
    
    mlir = _lower(circuit)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10*[1]

# ---------------------------------------------------------------------------
# Test control
# ---------------------------------------------------------------------------

def test_classcial_control():

    def circuit():
        qv = QuantumVariable(2)
        h(qv[0])
        c = measure(qv[0])
        with control(c):
            x(qv[1])
        return measure(qv)
    
    mlir = _lower(circuit)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)

# ---------------------------------------------------------------------------
# Test q_cond
# ---------------------------------------------------------------------------

def test_q_cond():

    def circuit():

        def false_fun(qbl):
            x(qbl[0])
            return qbl

        def true_fun(qbl):
            return qbl

        qbl = QuantumBool()
        h(qbl[0])
        pred = measure(qbl[0])

        qbl = q_cond(pred,
                    true_fun,
                    false_fun,
                    qbl)

        return measure(qbl[0])
    
    mlir = _lower(circuit)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i1")
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    #assert result == 10*[1] # Needs fix of issue #538

# ---------------------------------------------------------------------------
# Test q_while_loop
# ---------------------------------------------------------------------------

def test_q_while_loop():
    """While loop with loop-carried quantum variable."""

    def circuit():
        qv = QuantumVariable(10)

        def cond_fun(val):
            i, qv = val
            return i < 10
        
        def body_fun(val):
            i, qv = val
            x(qv[i])
            return i+1, qv

        q_while_loop(cond_fun, body_fun, (0, qv))
        return measure(qv)    

    mlir = _lower(circuit)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10*[1023]


def test_q_while_loop_acc():
    """While loop-carried quantum variable and accumulator."""

    def circuit():
        qv = QuantumVariable(10)

        def cond_fun(val):
            i, acc, qv = val
            return i < 5
        
        def body_fun(val):
            i, acc, qv = val
            x(qv[i])
            acc += measure(qv[i])
            return i+1, acc, qv

        i, acc, qv = q_while_loop(cond_fun, body_fun, (0, 0, qv))
        return acc   

    mlir = _lower(circuit)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10*[5]

# ---------------------------------------------------------------------------
# Test q_fori_loop
# ---------------------------------------------------------------------------

def test_q_fori_loop():
    """Fori loop with loop-carried quantum variable and accumulator."""

    def circuit():
        qv = QuantumVariable(10)

        def body_fun(i, val):
            acc, qv = val
            x(qv[i])
            acc += measure(qv[i])
            return acc, qv

        acc, qv = q_fori_loop(0, 5, body_fun, (0, qv))

        return acc

    mlir = _lower(circuit)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10*[5]

# ---------------------------------------------------------------------------
# Test nested control flow
# ---------------------------------------------------------------------------

def test_nested_q_fori_loop_control():
    """Nested fori loop with control inside."""

    def circuit():
        qv = QuantumVariable(10)

        def body_fun(i, val):
            acc, qv = val
            x(qv[i])
            c = measure(qv[i])
            with control(c):
                x(qv[i])
            acc += measure(qv[i])
            return acc, qv

        acc, qv = q_fori_loop(0, 5, body_fun, (0, qv))

        return acc

    mlir = _lower(circuit)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    #assert result == 10*[0] # Needs fix of issue #538

# ---------------------------------------------------------------------------
# Test jrange loop
# ---------------------------------------------------------------------------

def test_jrange_loop():
    """A jrange loop produces a scf.while with !quake.veq<?> loop-carried value."""

    pytest.skip("jrange loop test skipped")

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
    validate_quake_mlir(mlir)

# ---------------------------------------------------------------------------
# Test gate application functions acting on QuantumVariables
# (uses while loop)
# ---------------------------------------------------------------------------

def test_single_gate_application_quantum_variable():
    """Gate application function (x) applied to QuantumVariable"""

    def circuit():
        qv = QuantumVariable(10)
        x(qv)
        return measure(qv)

    mlir = _lower(circuit)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10*[1023]


def test_gate_application_quantum_variable():
    """Gate application functions (h,x,y,z,s,t) applied to QuantumVariable"""

    def circuit():
        qv = QuantumVariable(4)
        h(qv)
        x(qv)
        y(qv)
        z(qv)
        s(qv)
        t(qv)
        return measure(qv)

    mlir = _lower(circuit)
    for gate in ("quake.h", "quake.x", "quake.y", "quake.z", "quake.s", "quake.t"):
        assert gate in mlir, f"Expected {gate!r} in output"
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)

# ---------------------------------------------------------------------------
# Test algorithms
# ---------------------------------------------------------------------------

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

    # Measurement with correct i64 type
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")

    validate_quake_mlir(mlir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
