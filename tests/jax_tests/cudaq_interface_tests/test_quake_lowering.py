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

from collections import Counter
import warnings
import jax
import jax.numpy as jnp
import numpy as np
import operator
import pytest
import re

from qrisp import (
    QuantumVariable,
    QuantumBool,
    QuantumFloat,
    h,
    mcx,
    x,
    y,
    z,
    cp,
    cx,
    cy,
    cz,
    gphase,
    p,
    rx,
    ry,
    rz,
    rxx,
    rz,
    rzz,
    s,
    swap,
    sx,
    t,
    xxyy,
    measure,
    control,
    invert,
    conjugate,
)
from qrisp.alg_primitives import amplitude_amplification, q_switch, QPE, QFT
from qrisp.block_encodings import BlockEncoding
from qrisp.jasp import (
    make_jaspr,
    jrange,
    q_while_loop,
    q_cond,
    q_fori_loop,
    qache,
    terminal_sampling,
)
from qrisp.operators import X, Y, Z

try:
    from qrisp.jasp.mlir.quake_lowering import jaspr_to_quake_mlir, validate_quake_mlir
    from qrisp.jasp.cudaq_interface import run_quake_mlir
except ImportError as exc:
    # Skip the entire test file if the import fails
    pytest.skip(f"cudaq unavailable: {exc}", allow_module_level=True)

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
        mlir_str = jaspr_to_quake_mlir(jaspr)
    return mlir_str


def assert_return_type(mlir: str, expected_type: str):
    """Assert the MLIR contains a func.return with the given type."""
    pattern = rf"func\.return\s+%\w+\s*:\s*{re.escape(expected_type)}"
    assert re.search(pattern, mlir), f"Expected return type '{expected_type}', not found in:\n{mlir}"


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
    assert "(!quake.ref, !quake.ref) -> ()" in mlir, "CX gate should have '(!quake.ref, !quake.ref) -> ()' type sig"
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
    assert "(f64, !quake.ref) -> ()" in mlir, "rz gate should have '(f64, !quake.ref) -> ()' type sig"
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
        assert "(!quake.veq<?>) -> i64" in mlir, "veq_size should use functional-type: (!quake.veq<?>) -> i64"
    validate_quake_mlir(mlir)


def test_alloca_veq_format():
    """quake.alloca !quake.veq<?>[%n : i64] — type before size in brackets."""

    def circuit():
        qv = QuantumVariable(4)
        return qv

    mlir = _lower(circuit)
    # The alloca must print the type first, then size in brackets
    assert "quake.alloca !quake.veq<?>[" in mlir, "alloca format should be '!quake.veq<?>[%n : i64]'"
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
    assert "!quake.veq<?>" in mlir or "!quake.ref" in mlir, "Expected Quake qubit types in output"
    assert "quake." in mlir, "Expected Quake ops in output"
    validate_quake_mlir(mlir)


def test_parity_not_lowered():
    """jasp.parity ops are left in place (not lowered to Quake)."""
    from qrisp.jasp.mlir.quake_lowering.jasp_to_quake.gate_mapping import get_gate_info

    # parity is not a gate name, but verify it's not in GATE_MAP
    assert get_gate_info("parity") is None


def test_gate_mapping_standard_gates():
    """Verify that all standard gates are in the gate map."""
    from qrisp.jasp.mlir.quake_lowering.jasp_to_quake.gate_mapping import get_gate_info, GATE_MAP

    expected_gates = {
        "h",
        "x",
        "y",
        "z",
        "s",
        "t",
        "rx",
        "ry",
        "rz",
        "p",
        "r1",
        "cx",
        "cy",
        "cz",
        "swap",
        "u3",
    }
    for gate in expected_gates:
        info = get_gate_info(gate)
        assert info is not None, f"Expected gate '{gate}' in GATE_MAP"


def test_func_call_lowering():
    """
    Test that a function call to a separate @qache function is correctly lowered.
    No unsupported jasp types should be present in the output.
    """

    @qache
    def test():
        qv = QuantumVariable(2)
        return qv

    def main():
        qv = test()
        return qv

    mlir = _lower(main)
    assert "test" in mlir, "Expected call to 'test' function in output"
    assert "quake.alloca" in mlir, "Expected quake.alloca in output for test function"
    assert "quake.veq" in mlir, "Expected quake.veq type in output for test function"
    # No jasp types should be present in the output
    validate_quake_mlir(mlir)


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


def test_decomposed_gates_sx():
    """Decomposed gates (sx, sx_dg) emit the expected sequence of quake ops."""

    def circuit():
        qv = QuantumVariable(2)
        sx(qv[0])
        with invert():
            sx(qv[0])
        return qv

    mlir = _lower(circuit)
    # Check that the expected sequence of ops for sx/sx_dg is present
    assert "quake.h" in mlir, "Expected quake.h in output for sx decomposition"
    assert "quake.s" in mlir, "Expected quake.s in output for sx decomposition"
    assert "quake.s<adj>" in mlir, "Expected quake.s<adj> in output for sx_dg decomposition"
    validate_quake_mlir(mlir)


def test_decomposed_gates():
    """Decomposed gates (rxx, rzz, xxyy) emit the expected sequence of quake ops."""

    def circuit():
        qv = QuantumVariable(4)

        rxx(0.3, qv[0], qv[1])
        with invert():
            rxx(0.3, qv[0], qv[1])

        rzz(0.2, qv[1], qv[2])
        with invert():
            rzz(0.2, qv[1], qv[2])

        xxyy(0.5, 0.1, qv[0], qv[1])
        with invert():
            xxyy(0.5, 0.1, qv[0], qv[1])

        return qv

    mlir = _lower(circuit)
    validate_quake_mlir(mlir)


def test_parameterized_gate():
    """rz / rx gates carry floating-point parameters."""

    def circuit():
        qv = QuantumVariable(2)
        rz(0.5, qv[0])
        rx(1.0, qv[1])
        rx(1, qv[1])  # Test that integer literals are also accepted as parameters and correctly typed as f64
        return qv

    mlir = _lower(circuit)
    assert "quake.rz" in mlir, "Expected quake.rz in output"
    assert "quake.rx" in mlir, "Expected quake.rx in output"
    # Parameters should appear as f64 scalars
    assert "f64" in mlir, "Expected f64 parameter type in output"
    validate_quake_mlir(mlir)


def test_controlled_gates():
    """cx maps to quake.x with one control qubit."""

    def circuit():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        cy(qv[0], qv[1])
        cz(qv[0], qv[1])
        cp(0.5, qv[0], qv[1])
        return qv

    mlir = _lower(circuit)
    assert "quake.h" in mlir
    assert "quake.x" in mlir
    assert "quake.y" in mlir
    assert "quake.z" in mlir
    assert "quake.r1" in mlir
    # Control qubit should be present in square brackets
    assert "[%" in mlir, "Expected control qubit in bracket notation"
    validate_quake_mlir(mlir)


def test_swap_gate():
    """swap is decomposed and maps to quake.x gates with two qubit operands."""

    def circuit():
        qv = QuantumVariable(2)
        swap(qv[0], qv[1])
        return qv

    mlir = _lower(circuit)
    assert "quake.x" in mlir
    # Both qubits should be present as operands
    assert "(!quake.ref, !quake.ref) -> ()" in mlir, (
        "Expected quake.x to have '(!quake.ref, !quake.ref) -> ()' type sig"
    )
    validate_quake_mlir(mlir)


def test_cgphase_gate():
    """cgphase maps to quake.r1."""

    def main():

        qv = QuantumVariable(2)
        h(qv[0])
        with control(qv[0]):
            gphase(np.pi / 2, qv[1])
        h(qv[0])
        return measure(qv[0])  # P(0) = P(1) = 0.5

    mlir = _lower(main)
    assert "quake.r1" in mlir, "Expected quake.r1 in output"
    validate_quake_mlir(mlir)


def test_negative_indexing():

    def main():
        qv = QuantumVariable(3)
        x(qv[-1])
        return measure(qv[2])

    mlir = _lower(main)
    assert "quake.x" in mlir, "Expected quake.x in output"
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [1], f"Expected qubit 2 to be flipped by x(qv[-1]), got {result}"


def test_math():
    """Test that classical math operations are correctly lowered to arith/math ops and can be used in the quantum program."""

    def main():
        a = QuantumFloat(1)
        h(a[0])
        b = measure(a)
        c = 2.0**b
        d = jnp.log(c)
        return d

    mlir = _lower(main)
    assert "math.powf" in mlir, "Expected math.powf for exponentiation"
    assert "math.log" in mlir, "Expected math.log for logarithm"
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)


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
    assert result == 10 * [1]


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
    assert result == 10 * [1]


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
    assert result == 10 * [1]


# ---------------------------------------------------------------------------
# Test invert and cojugate
# (with qubit-wise gate application functions acting on qubits)
# ---------------------------------------------------------------------------


def test_invert():

    def circuit():

        def inner(qv):
            rx(0.5, qv[0])
            z(qv[0])
            h(qv[0])

        qv = QuantumVariable(1)

        inner(qv)

        with invert():
            inner(qv)

        return measure(qv)

    mlir = _lower(circuit)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [0]


def test_conjugate():

    def circuit():
        qv = QuantumVariable(1)
        with conjugate(h)(qv[0]):
            z(qv[0])
        return measure(qv[0])

    mlir = _lower(circuit)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [1]


# ---------------------------------------------------------------------------
# Test control
# ---------------------------------------------------------------------------


def test_classcial_control():
    """Control on a measurement result."""

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


def test_quantum_control():
    """Control on a qubit value (not measurement result)."""

    def circuit():
        qv = QuantumVariable(2)
        x(qv[0])
        with control(qv[0]):
            x(qv[1])
        return measure(qv[1])

    mlir = _lower(circuit)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i1")
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [1]


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

        qbl = q_cond(pred, true_fun, false_fun, qbl)

        return measure(qbl[0])

    mlir = _lower(circuit)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i1")
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [1]


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
            return i + 1, qv

        q_while_loop(cond_fun, body_fun, (0, qv))
        return measure(qv)

    mlir = _lower(circuit)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [1023]


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
            return i + 1, acc, qv

        i, acc, qv = q_while_loop(cond_fun, body_fun, (0, 0, qv))
        return acc

    mlir = _lower(circuit)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [5]


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
    assert result == 10 * [5]


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
    assert result == 10 * [0]


# ---------------------------------------------------------------------------
# Test classical control flow
# ---------------------------------------------------------------------------


def test_jax_fori_loop():
    """Test that a JAX fori_loop is correctly lowered to MLIR."""

    def circuit():
        result = jax.lax.fori_loop(0, 10, lambda i, x: x + i, 0)
        return result

    mlir = _lower(circuit)
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [45], f"Expected sum of 0..9 to be 45, got {result}"


# ---------------------------------------------------------------------------
# Test jrange loop
# ---------------------------------------------------------------------------


def test_jrange_loop():
    """Test that a jrange loop is correctly lowered to MLIR."""

    def circuit():
        qv = QuantumVariable(3)
        for i in jrange(3):
            x(qv[i])
        return measure(qv)

    mlir = _lower(circuit)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [7], f"Expected all qubits measured as 1 (7), got {result}"


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
    assert result == 10 * [1023], f"Expected all qubits measured as 1 (1023), got {result}"


def test_gate_application_quantum_variable_slice():
    """Gate application function (x) applied to a slice of QuantumVariable"""

    def circuit():
        qv = QuantumVariable(10)
        x(qv[:5])
        return measure(qv)

    mlir = _lower(circuit)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [31], f"Expected lower 5 qubits measured as 1 (31), got {result}"

    """Gate application function (x) applied to a slice of QuantumVariable with negative upper bound"""

    def circuit():
        qv = QuantumVariable(10)
        x(qv[:-5])
        return measure(qv)

    mlir = _lower(circuit)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [31], f"Expected lower 5 qubits measured as 1 (31), got {result}"


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


def test_invert_quantum_variable():
    """Test that invert works when applied to a QuantumVariable.

    Previouly, the condition lowering for the while loop used in invert was incorrectly treating the sge (signed greater-than-or-equal) condition as a strict greater-than,
    causing the loop to miss the final iteration where the last qubit is flipped. This test verifies that all qubits are correctly flipped to 1,
    confirming that the loop boundary condition is now correctly implemented (hotfix in _scf_to_cc.py to convert sge to sgt).
    https://github.com/NVIDIA/cuda-quantum/issues/4401
    """

    def main():
        qv = QuantumVariable(3)
        with invert():
            x(qv)
        return measure(qv)

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [7], f"Expected all qubits flipped to 1 (7), got {result}"


# ---------------------------------------------------------------------------
# Test mcx
# ---------------------------------------------------------------------------

methods = ["balauca", "khattar"]


@pytest.mark.parametrize("method", methods)
def test_mcx(method):
    """Multi-controlled X gate (mcx) with variable number of controls."""

    def circuit():
        """Test mcx with 2 controls and 1 target."""
        qv = QuantumVariable(3)
        x(qv[0])
        x(qv[1])
        mcx(qv[:2], qv[2], method=method)
        return measure(qv)

    mlir = _lower(circuit)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [7], f"Expected target qubit flipped to 1 when both controls are 1 (7), got {result}"

    def circuit():
        """Test mcx with 9 controls and 1 target."""
        qv = QuantumVariable(10)
        x(qv[:9])
        mcx(qv[:9], qv[9], method=method)
        return measure(qv)

    mlir = _lower(circuit)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [1023], f"Expected target qubit flipped to 1 when all 9 controls are 1 (1023), got {result}"


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


def test_amplitude_amplification():
    """Test amplitude amplification algorithm with a simple oracle and state function."""

    def state_function(qb):
        ry(np.pi / 8, qb)

    def oracle_function(qb):
        z(qb)

    def main():
        qb = QuantumBool()
        state_function(qb)
        amplitude_amplification([qb], state_function, oracle_function, iter=3)
        return measure(qb[0])

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert np.mean(result) >= 0.8, f"Expected amplitude amplification to yield mostly 1s, got {result}"


def test_quantum_fourier_transform():
    """Test that the quantum Fourier transform produces the expected output state."""

    def main():
        qv = QuantumVariable(5)
        h(qv)
        QFT(qv)
        return measure(qv)

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [0]


def test_quantum_phase_estimation():
    """Test that the quantum phase estimation produces the expected output state."""

    def U(qv):
        x = 0.5
        y = 0.125

        p(x * 2 * np.pi, qv[0])
        p(y * 2 * np.pi, qv[1])

    def main():
        qv = QuantumVariable(2)
        h(qv)
        res = QPE(qv, U, precision=3)
        return measure(res)

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    for r in result:
        assert r in {0, 0.125, 0.5, 0.625}


def test_trotterization():
    """Test that a simple Trotterized Hamiltonian evolution produces valid results."""
    from qrisp.operators import X, Y, Z

    def main():
        qv = QuantumVariable(2)
        H = X(0) * X(1) + Y(0) * Y(1) + Z(0) * Z(1) + X(0) + X(1)
        U = H.trotterization()
        U(qv, t=1.0, steps=10)
        return measure(qv)

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    # We can't predict the exact result, but we can check that it's a valid measurement (0, 1, 2, or 3)
    assert all(0 <= r <= 3 for r in result), f"Expected valid measurement results (0-3), got {result}"


@pytest.mark.parametrize("method", ["tree", "sequential"])
def test_q_switch(method):
    """Test q_switch with multiple branches and a quantum float index."""

    def main():

        def f0(x):
            x += 1

        def f1(x):
            x += 2

        def f2(x):
            pass

        def f3(x):
            h(x[1])

        branches = [f0, f1, f2, f3]

        operand = QuantumFloat(4)
        operand[:] = 1
        index = QuantumFloat(2)
        h(index)

        q_switch(index, branches, operand, method=method)
        return measure(operand)

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)


# ---------------------------------------------------------------------------
# Test arrays
# ---------------------------------------------------------------------------


def test_array():
    """Test that we can create classical (traced) arrays and access them in the quantum program."""

    def main():
        """Static indexing."""

        arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a = QuantumVariable(5)
        rz(arr[0], a[0])

        return measure(a[0])

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [0], f"Expected a measurement result of 0, got {result}"

    def main():
        """Dynamic indexing."""

        arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a = QuantumVariable(5)
        b = QuantumVariable(1)
        ind = jnp.int32(measure(b[0]))
        rz(arr[ind], a[0])

        return measure(a[0])

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [0], f"Expected a measurement result of 0, got {result}"

    @qache
    def test(arr, qv):
        rz(arr[0], qv[0])
        return measure(qv[0])

    def main():
        """Test that we can pass traced arrays as arguments to a @qache function and use them in the quantum program."""
        arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        a = QuantumVariable(5)
        res = test(arr, a)
        return res

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [0], f"Expected a measurement result of 0, got {result}"


# ---------------------------------------------------------------------------
# Test dynamic classical array indexing
# ---------------------------------------------------------------------------


def test_array_dynamic_index_return():
    """Dynamic index into a classical array used as the return value."""

    def main():
        arr = jnp.array([1.57, 0.78, 0.39, 0.25, 0.12])
        qv = QuantumVariable(1)
        # qubit is |0⟩ so measure returns 0 → pick first element
        ind = jnp.int32(measure(qv[0]))
        return arr[ind]

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=5)
    assert all(abs(r - 1.57) < 1e-6 for r in result), f"Expected all shots to return 1.57, got {result}"


def test_array_dynamic_index_gate_angle():
    """Dynamic index into a classical array used as a gate rotation angle."""

    def main():
        arr = jnp.array([0.0, 3.14159265])
        qv = QuantumVariable(2)
        # qubit 0 is |0⟩ → measure returns 0 → arr[0] = 0.0 → rz(0) leaves |0⟩
        ind = jnp.int32(measure(qv[0]))
        rz(arr[ind], qv[1])
        return measure(qv[1])

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [0], f"Expected all shots to return 0, got {result}"


def test_array_dynamic_index_in_qache():
    """Dynamic index into an array inside a @qache-decorated subroutine."""

    @qache
    def apply_angle(arr, qv):
        ind = jnp.int32(measure(qv[0]))
        rz(arr[ind], qv[1])
        return measure(qv[1])

    def main():
        arr = jnp.array([0.0, 3.14159265])
        qv = QuantumVariable(2)
        return apply_angle(arr, qv)

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [0], f"Expected all shots to return 0, got {result}"


# ---------------------------------------------------------------------------
# Test QuantumFloat and arithmetic operations
# ---------------------------------------------------------------------------


ops = [operator.add, operator.sub]
rhs_type = ["classical", "quantum"]
instances = [
    # (size1, exp1, val1, size2, exp2, val2)
    pytest.param(3, 0, 2, 4, 0, 1, id="QuantumFloat case 1"),
    pytest.param(3, 0, 3, 4, 0, 2, id="QuantumFloat case 2"),
]


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("rhs_type", rhs_type)
@pytest.mark.parametrize("size1, exp1, val1, size2, exp2, val2", instances)
def test_quantum_float_arithmetic(op, rhs_type, size1, exp1, val1, size2, exp2, val2):
    """Arithmetic operations on QuantumFloat with classical or quantum RHS."""

    def main():
        a = QuantumFloat(size1, exponent=exp1)
        a[:] = val1

        if rhs_type == "classical":
            b = val2
        else:
            b = QuantumFloat(size2, exponent=exp2)
            b[:] = val2

        c = op(a, b)
        return measure(c)

    expected = op(val1, val2)

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [expected], (
        f"Expected quantum-{rhs_type} {op.__name__} of {val1} and {val2} to yield {expected}, got {result}"
    )


ops_inpl = [operator.iadd, operator.isub]
rhs_type = ["classical", "quantum"]
instances = [
    # (size1, exp1, val1, size2, exp2, val2)
    pytest.param(3, 0, 2, 3, 0, 1, id="QuantumFloat case 1"),
    pytest.param(3, 0, 3, 4, 0, 3, id="QuantumFloat case 2"),
]


@pytest.mark.parametrize("op", ops_inpl)
@pytest.mark.parametrize("rhs_type", rhs_type)
@pytest.mark.parametrize("size1, exp1, val1, size2, exp2, val2", instances)
def test_quantum_float_arithmetic_inpl(op, rhs_type, size1, exp1, val1, size2, exp2, val2):
    """In-place arithmetic operations on QuantumFloat."""

    def main():
        a = QuantumFloat(size1, exponent=exp1)
        a[:] = val1

        if rhs_type == "classical":
            b = val2
        else:
            b = QuantumFloat(size2, exponent=exp2)
            b[:] = val2

        op(a, b)
        return measure(a)

    expected = op(val1, val2)

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)
    assert result == 10 * [expected], (
        f"Expected quantum-{rhs_type} {op.__name__} of {val1} and {val2} to yield {expected}, got {result}"
    )


ops_comp = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
]
rhs_type = ["classical", "quantum"]
instances = [
    # (size1, exp1, val1, size2, exp2, val2)
    pytest.param(3, 0, 2, 3, 0, 1, id="QuantumFloat case 1"),
    # pytest.param(3, 0, 3, 4, 0, 3, id="QuantumFloat case 2"),
]


@pytest.mark.parametrize("op", ops_comp)
@pytest.mark.parametrize("rhs_type", rhs_type)
@pytest.mark.parametrize("size1, exp1, val1, size2, exp2, val2", instances)
def test_quantum_float_comparison(op, rhs_type, size1, exp1, val1, size2, exp2, val2):
    """Comparison operations on QuantumFloat with classical or quantum RHS."""

    if op in (operator.eq, operator.ne):
        pytest.skip("Equality and inequality comparisons on QuantumFloat are not supported.")

    def main():
        a = QuantumFloat(size1, exponent=exp1)
        a[:] = val1

        if rhs_type == "classical":
            b = val2
        else:
            b = QuantumFloat(size2, exponent=exp2)
            b[:] = val2

        c = op(a, b)
        return measure(c)

    expected = op(val1, val2)

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=1)
    assert result == 1 * [expected], (
        f"Expected quantum-{rhs_type} {op.__name__} of {val1} and {val2} to yield {expected}, got {result}"
    )


# ---------------------------------------------------------------------------
# Test BlockEncoding
# ---------------------------------------------------------------------------


def _post_selection(res_dict):
    # Post-selection on ancillas being in |0> state
    filtered_dict = {k[0]: p for k, p in res_dict.items() if all(x == 0 for x in k[1:])}
    success_prob = sum(filtered_dict.values())
    filtered_dict = {k: p / success_prob for k, p in filtered_dict.items()}
    return filtered_dict


@pytest.mark.parametrize("operator", [X(0), X(0) + Z(1), X(0) * X(1) + Z(0) * Z(1)])
def test_simple_block_encoding(operator):
    """Test that we can create a BlockEncoding from a simple Hamiltonian and apply it to a quantum variable."""

    BE = BlockEncoding.from_operator(operator)

    def main():

        operand = QuantumVariable(2)
        ancs = BE.apply(operand)
        return measure(operand)

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    result = run_quake_mlir(mlir, shots=10)


@pytest.mark.parametrize("operator", [X(0) + Z(1), X(0) * X(1) + Z(0) * Z(1)])
def test_simple_block_encoding_results(operator):
    """
    Test that applying a BlockEncoding of a simple Hamiltonian produces the expected
    measurement distribution on the operand qubits after post-selection on ancillas.
    """

    BE = BlockEncoding.from_operator(operator)

    def main():

        operand = QuantumVariable(2)
        ancs = BE.apply(operand)
        return measure(operand), measure(ancs[0])

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    results = run_quake_mlir(mlir, shots=500)
    counts = Counter(results)
    counts_dict = dict(counts)
    res_dict = {outcome: count / len(results) for outcome, count in counts.items()}
    filtered_dict = _post_selection(res_dict)

    # Now do the same thing with JASP's terminal_sampling to get the expected distribution for comparison
    @terminal_sampling
    def main():

        operand = QuantumVariable(2)
        ancs = BE.apply(operand)
        return operand, ancs[0]

    res_dict_jasp = main()
    filtered_dict_jasp = _post_selection(res_dict_jasp)

    all_keys = set(filtered_dict.keys()).union(set(filtered_dict_jasp.keys()))

    for key in all_keys:
        val1 = filtered_dict.get(key, 0.0)
        val2 = filtered_dict_jasp.get(key, 0.0)
        assert np.isclose(val1, val2, atol=1e-1)


@pytest.mark.parametrize("operator", [X(0) + Z(1), X(0) * X(1) + Z(0) * Z(1)])
def test_simple_block_encoding_poly(operator):
    """Test that we can create a polynomial from a BlockEncoding and apply it to a quantum variable."""

    BE = BlockEncoding.from_operator(operator)
    BE_poly = BE.poly(np.array([0.5, 0.5]))

    def main():

        operand = QuantumVariable(2)
        ancs = BE_poly.apply(operand)
        return measure(operand)

    mlir = _lower(main)
    validate_quake_mlir(mlir)
    results = run_quake_mlir(mlir, shots=10)


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
