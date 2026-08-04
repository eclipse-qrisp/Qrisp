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
Tests for the Jasp → Quake (memory-semantics) lowering backend.

Sections
--------
- Basic circuit structure: allocation, extraction, deallocation.
- Interface invariants: Quake types present, no ``!jasp.*`` types leak, kernel attributes.
- MLIR assembly-format validity (exact CUDA-Q Quake dialect printed syntax).
- Gate mapping and lowering: single/multi-qubit, parameterized, decomposed, controlled gates.
- Qubit indexing (constant/dynamic/negative indices).
- Classical math lowering (arith/math ops mixed with quantum control flow).
- Measurement.
- QuantumVariable-wide gate application (while-loop lowering).
- Function calls and kernel decorators (``@qache``, ``@quantum_kernel``).
- Negative tests: unsupported constructs raise ``NotImplementedError``.
"""

import warnings
import jax.numpy as jnp
import numpy as np
import pytest
import re

import cudaq

from qrisp import (
    QuantumVariable,
    QuantumFloat,
    h,
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
)
from qrisp.jasp import (
    make_jaspr,
    qache,
    quantum_kernel,
)
from qrisp.jasp.cudaq_interface.quake_lowering import jaspr_to_quake_mlir, validate_quake_mlir
from qrisp.jasp.cudaq_interface import cudaq_kernel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lower(circuit_fn, *trace_args):
    """Build the Jaspr for *circuit_fn* and lower it to Quake MLIR.

    Returns the xDSL module. Deprecation warnings from xDSL internals
    are silenced.
    """
    jaspr = make_jaspr(circuit_fn)(*trace_args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xdsl_module = jaspr_to_quake_mlir(jaspr)
    return xdsl_module


def assert_return_type(mlir: str, expected_type: str):
    """Assert the MLIR contains a func.return with the given type."""
    pattern = rf"func\.return\s+%\w+\s*:\s*{re.escape(expected_type)}"
    assert re.search(pattern, mlir), f"Expected return type '{expected_type}', not found in:\n{mlir}"


# ---------------------------------------------------------------------------
# Basic circuit structure: allocation, extraction, deallocation
# ---------------------------------------------------------------------------


def test_alloc_and_dealloc():
    """create_qubits → quake.alloca / delete_qubits → quake.dealloc."""

    def circuit():
        qv = QuantumVariable(3)
        return qv  # implicitly deallocated at end of jaspr scope

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    assert "quake.alloca" in mlir, "Expected quake.alloca in output"
    validate_quake_mlir(mlir)


def test_multi_qubit_alloc():
    """Multiple QuantumVariable allocations produce multiple quake.alloca ops."""

    def circuit():
        qv1 = QuantumVariable(2)
        qv2 = QuantumVariable(3)
        h(qv1[0])
        x(qv2[0])
        return measure(qv1[0]), measure(qv2[0])

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    # At least two alloca ops
    alloca_count = mlir.count("quake.alloca")
    assert alloca_count >= 2, f"Expected ≥2 quake.alloca ops, got {alloca_count}"
    validate_quake_mlir(mlir)


def test_extract_ref():
    """get_qubit → quake.extract_ref."""

    def circuit():
        qv = QuantumVariable(3)
        h(qv[0])
        cx(qv[0], qv[2])
        return qv

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    assert "quake.extract_ref" in mlir, "Expected quake.extract_ref in output"
    validate_quake_mlir(mlir)


# ---------------------------------------------------------------------------
# Interface invariants: Quake types present, no Jasp types leak, kernel attributes
# ---------------------------------------------------------------------------


def test_quake_types_present():
    """Output should contain quake.* types and ops."""

    def circuit():
        qv = QuantumVariable(2)
        h(qv[0])
        return measure(qv)

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    assert "!quake.veq<?>" in mlir or "!quake.ref" in mlir, "Expected Quake qubit types in output"
    assert "quake." in mlir, "Expected Quake ops in output"
    validate_quake_mlir(mlir)


def test_no_jasp_types_in_output():
    """Interface invariant: output must not contain any !jasp.* types."""

    def bell():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return measure(qv)

    xdsl_module = _lower(bell)

    mlir = str(xdsl_module)
    validate_quake_mlir(mlir)


def test_cudaq_kernel_attribute():
    """Lowered function should carry cudaq.kernel and cudaq.entrypoint attributes."""

    def simple():
        qv = QuantumVariable(1)
        h(qv[0])
        return measure(qv)

    xdsl_module = _lower(simple)

    mlir = str(xdsl_module)
    assert "cudaq.kernel" in mlir, "Expected cudaq.kernel attribute on function"
    assert "cudaq.entrypoint" in mlir, "Expected cudaq.entrypoint attribute on function"
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

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
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

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
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

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    assert "quake.rz" in mlir
    # The type signature must include both the f64 param and the quake.ref target
    assert "(f64, !quake.ref) -> ()" in mlir, "rz gate should have '(f64, !quake.ref) -> ()' type sig"
    validate_quake_mlir(mlir)


# ---------------------------------------------------------------------------
# Test dynamic-sized veq
# ---------------------------------------------------------------------------


def test_dynamic_veq_size_functional_type():
    """For a traced (non-constant) size, quake.veq_size keeps the dynamic
    !quake.veq<?> type and uses functional-type: (!quake.veq<?>) -> i64."""

    # Size is a traced argument (not a constant), so the alloca stays dynamic
    # (pass3b only staticizes constant-sized allocas).
    def circuit(n):
        qv = QuantumVariable(n)
        h(qv[qv.size - 1])
        return qv

    xdsl_module = _lower(circuit, 3)

    mlir = str(xdsl_module)
    # veq_size op should use functional-type with parens if present
    if "quake.veq_size" in mlir:
        assert "(!quake.veq<?>) -> i64" in mlir, "veq_size should use functional-type: (!quake.veq<?>) -> i64"
    validate_quake_mlir(mlir)


def test_dynamic_alloca_veq_format():
    """For a traced (non-constant) size, quake.alloca keeps the dynamic format
    !quake.veq<?>[%n : i64] — type before size in brackets."""

    # Size is a traced argument (not a constant), so the alloca stays dynamic
    # (pass3b only staticizes constant-sized allocas).
    def circuit(n):
        qv = QuantumVariable(n)
        return qv

    xdsl_module = _lower(circuit, 4)

    mlir = str(xdsl_module)
    # The alloca must print the type first, then size in brackets
    assert "quake.alloca !quake.veq<?>[" in mlir, "alloca format should be '!quake.veq<?>[%n : i64]'"
    validate_quake_mlir(mlir)


# ---------------------------------------------------------------------------
# Test static-sized veq (pass3b: constant-sized quake.alloca staticization)
# ---------------------------------------------------------------------------


def test_alloca_staticized_for_constant_size():
    """A constant-sized QuantumVariable is allocated with a static !quake.veq<N>."""

    def circuit():
        qv = QuantumVariable(5)
        h(qv[0])
        return measure(qv)

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    assert "quake.alloca !quake.veq<5>" in mlir
    assert "veq<?>[" not in mlir, "no dynamic alloca expected for a constant size"
    validate_quake_mlir(mlir)


def test_static_veq_transparent_uses_stay_static():
    """extract_ref/veq_size/dealloc on a static veq keep the static type, no relax_size."""

    def circuit():
        qv = QuantumVariable(4)
        x(qv[0])
        result = measure(qv)
        return result

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    assert "quake.alloca !quake.veq<4>" in mlir
    assert "quake.extract_ref %0[" in mlir or "!quake.veq<4>, i64) -> !quake.ref" in mlir
    assert "quake.relax_size" not in mlir, "transparent consumers should not need a relax_size cast"
    validate_quake_mlir(mlir)


def test_static_veq_control_flow_gets_relax_size():
    """A constant-sized veq carried through q_fori_loop is staticized, with
    quake.relax_size bridging the loop's dynamically-typed block argument."""
    from qrisp.jasp import q_fori_loop

    def circuit():
        qv = QuantumVariable(5)

        def body_fun(i, val):
            acc, qv = val
            x(qv[i])
            return acc + 1, qv

        acc, qv = q_fori_loop(0, 5, body_fun, (0, qv))
        return measure(qv)

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    # The alloca itself is staticized...
    assert "quake.alloca !quake.veq<5>" in mlir
    # ...and a relax_size cast bridges it into the loop's dynamic block argument.
    assert "quake.relax_size" in mlir
    assert "veq<?>[" not in mlir, "no dynamic alloca expected for a constant size"
    validate_quake_mlir(mlir)

    kernel = cudaq_kernel(circuit)
    result = cudaq.run(kernel, shots_count=5)
    assert result == 5 * [31], f"Expected all 5 qubits flipped (0b11111 = 31), got {result}"


def test_bell_circuit_full_format():
    """Full Bell circuit MLIR format validation — spot-check every key op."""

    def bell():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return measure(qv)

    xdsl_module = _lower(bell)

    mlir = str(xdsl_module)

    # Function attributes
    assert 'cudaq.kernel = "true"' in mlir
    assert 'cudaq.entrypoint = "true"' in mlir

    # Alloca (constant size 2 -> statically-sized veq)
    assert "quake.alloca !quake.veq<2>" in mlir

    # extract_ref with functional-type
    assert "(!quake.veq<2>" in mlir and "-> !quake.ref" in mlir, (
        "extract_ref must use functional-type format: (!quake.veq<2>, idx) -> !quake.ref"
    )

    # H gate
    assert "(!quake.ref) -> ()" in mlir

    # CX gate (quake.x with control)
    assert "(!quake.ref, !quake.ref) -> ()" in mlir

    # Measurement with correct i64 type
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)


# ---------------------------------------------------------------------------
# Gate mapping and lowering: single/multi-qubit, parameterized,
# decomposed, controlled gates
# ---------------------------------------------------------------------------


def test_gate_mapping_standard_gates():
    """Verify that all standard gates are in the gate map."""
    from qrisp.jasp.cudaq_interface.quake_lowering.jasp_to_quake.gate_mapping import get_gate_info, GATE_MAP

    expected_gates = {
        "h",
        "x",
        "y",
        "z",
        "s",
        "s_dg",
        "sx",
        "sx_dg",
        "t",
        "t_dg",
        "rx",
        "ry",
        "rz",
        "p",
        "r1",
        "cx",
        "cy",
        "cz",
        "u3",
        "gphase",
        "cgphase",
    }
    for gate in expected_gates:
        info = get_gate_info(gate)
        assert info is not None, f"Expected gate '{gate}' in GATE_MAP"


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

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    for gate in ("quake.h", "quake.x", "quake.y", "quake.z", "quake.s", "quake.t"):
        assert gate in mlir, f"Expected {gate!r} in output"
    validate_quake_mlir(mlir)


def test_decomposed_gates_sx():
    """sx / sx_dg lower to quake.rx(±π/2), matching Qrisp's RX(±π/2) definition."""

    def circuit():
        qv = QuantumVariable(2)
        sx(qv[0])
        with invert():
            sx(qv[0])
        return qv

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    assert "quake.rx" in mlir, "Expected quake.rx in output for sx/sx_dg"
    assert "1.5707963267948966" in mlir, "Expected a pi/2 constant angle in output"
    assert "-1.5707963267948966" in mlir, "Expected a -pi/2 constant angle in output for sx_dg"
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

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    validate_quake_mlir(mlir)


def test_parameterized_gate():
    """rz / rx / ry / p gates carry floating-point parameters."""

    def circuit():
        qv = QuantumVariable(2)
        rz(0.5, qv[0])
        rx(1.0, qv[1])
        rx(1, qv[1])  # Test that integer literals are also accepted as parameters and correctly typed as f64
        ry(0.3, qv[0])
        p(0.7, qv[1])
        return qv

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    assert "quake.rz" in mlir, "Expected quake.rz in output"
    assert "quake.rx" in mlir, "Expected quake.rx in output"
    assert "quake.ry" in mlir, "Expected quake.ry in output"
    assert "quake.r1" in mlir, "Expected quake.r1 in output for p gate"
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

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
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

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
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

    xdsl_module = _lower(main)

    mlir = str(xdsl_module)
    assert "quake.r1" in mlir, "Expected quake.r1 in output"
    validate_quake_mlir(mlir)


# ---------------------------------------------------------------------------
# Qubit indexing (constant / dynamic / negative indices)
# ---------------------------------------------------------------------------


def test_negative_indexing():
    """Negative indexing (e.g. qv[-1]) is supported and lowers to quake.extract_ref with idx + size(veq)."""

    def main():
        qv = QuantumVariable(3)
        x(qv[-1])
        return measure(qv[2])

    xdsl_module = _lower(main)

    mlir = str(xdsl_module)
    assert "quake.x" in mlir, "Expected quake.x in output"
    validate_quake_mlir(mlir)

    kernel = cudaq_kernel(main)
    result = cudaq.run(kernel, shots_count=10)
    assert result == 10 * [1], f"Expected qubit 2 to be flipped by x(qv[-1]), got {result}"


def test_constant_nonneg_index_skips_normalization_scaffold():
    """A compile-time-constant, non-negative index (e.g. qv[0]) needs no
    sign check: no cmpi/select (and no veq_size lookup) should be emitted."""

    def main():
        qv = QuantumVariable(3)
        x(qv[0])
        return measure(qv[0])

    xdsl_module = _lower(main)

    mlir = str(xdsl_module)
    assert "arith.cmpi" not in mlir
    assert "arith.select" not in mlir
    assert "quake.veq_size" not in mlir
    validate_quake_mlir(mlir)

    kernel = cudaq_kernel(main)
    result = cudaq.run(kernel, shots_count=10)
    assert result == 10 * [1]


def test_constant_negative_index_skips_cmpi_select():
    """A compile-time-constant negative index (e.g. qv[-1]) has a
    statically-known sign: only the addi (idx + size) is needed, no
    cmpi/select."""

    def main():
        qv = QuantumVariable(3)
        x(qv[-1])
        return measure(qv[2])

    xdsl_module = _lower(main)

    mlir = str(xdsl_module)
    assert "arith.cmpi" not in mlir
    assert "arith.select" not in mlir
    assert "arith.addi" in mlir, "Expected idx + size(veq) addi for a negative constant index"
    validate_quake_mlir(mlir)

    kernel = cudaq_kernel(main)
    result = cudaq.run(kernel, shots_count=10)
    assert result == 10 * [1]


# ---------------------------------------------------------------------------
# Classical math lowering
# ---------------------------------------------------------------------------


def test_math():
    """Test that classical math operations are correctly lowered to arith/math ops and can be used in the quantum program."""

    def main():
        a = QuantumFloat(1)
        h(a[0])
        b = measure(a)
        c = 2.0**b
        d = jnp.log(c)
        return d

    xdsl_module = _lower(main)

    mlir = str(xdsl_module)
    assert "math.powf" in mlir, "Expected math.powf for exponentiation"
    assert "math.log" in mlir, "Expected math.log for logarithm"
    validate_quake_mlir(mlir)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def test_measure_single_qubit():
    """Single-qubit measure: quake.mz + quake.discriminate."""

    def circuit():
        qv = QuantumVariable(1)
        x(qv[0])
        return measure(qv[0])

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i1")
    validate_quake_mlir(mlir)

    kernel = cudaq_kernel(circuit)
    result = cudaq.run(kernel, shots_count=10)
    assert result == 10 * [1]


def test_measure_quantum_variable():
    """QuantumVariable measure"""

    def circuit():
        qv = QuantumVariable(3)
        x(qv[0])
        return measure(qv)

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)

    kernel = cudaq_kernel(circuit)
    result = cudaq.run(kernel, shots_count=10)
    assert result == 10 * [1]


def test_measure_single_qubit_quantum_variable():
    """QuantumVariable measure"""

    def circuit():
        qv = QuantumVariable(1)
        x(qv[0])
        return measure(qv)

    xdsl_module = _lower(circuit)
    mlir = str(xdsl_module)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)

    kernel = cudaq_kernel(circuit)
    result = cudaq.run(kernel, shots_count=10)
    assert result == 10 * [1]


# ---------------------------------------------------------------------------
# QuantumVariable-wide gate application (while-loop lowering)
# ---------------------------------------------------------------------------


def test_single_gate_application_quantum_variable():
    """Gate application function (x) applied to QuantumVariable"""

    def circuit():
        qv = QuantumVariable(10)
        x(qv)
        return measure(qv)

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)

    kernel = cudaq_kernel(circuit)
    result = cudaq.run(kernel, shots_count=10)
    assert result == 10 * [1023], f"Expected all qubits measured as 1 (1023), got {result}"


def test_gate_application_quantum_variable_slice():
    """Gate application function (x) applied to a slice of QuantumVariable"""

    def circuit():
        qv = QuantumVariable(10)
        x(qv[:5])
        return measure(qv)

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)

    kernel = cudaq_kernel(circuit)
    result = cudaq.run(kernel, shots_count=10)
    assert result == 10 * [31], f"Expected lower 5 qubits measured as 1 (31), got {result}"

    """Gate application function (x) applied to a slice of QuantumVariable with negative upper bound"""

    def circuit():
        qv = QuantumVariable(10)
        x(qv[:-5])
        return measure(qv)

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    assert "quake.mz" in mlir, "Expected quake.mz in output"
    assert_return_type(mlir, "i64")
    validate_quake_mlir(mlir)

    kernel = cudaq_kernel(circuit)
    result = cudaq.run(kernel, shots_count=10)
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

    xdsl_module = _lower(circuit)

    mlir = str(xdsl_module)
    for gate in ("quake.h", "quake.x", "quake.y", "quake.z", "quake.s", "quake.t"):
        assert gate in mlir, f"Expected {gate!r} in output"
    validate_quake_mlir(mlir)

    kernel = cudaq_kernel(circuit)
    result = cudaq.run(kernel, shots_count=10)


# ---------------------------------------------------------------------------
# Function calls and kernel decorators (@qache, @quantum_kernel)
# ---------------------------------------------------------------------------


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

    xdsl_module = _lower(main)

    mlir = str(xdsl_module)
    assert "test" in mlir, "Expected call to 'test' function in output"
    assert "quake.alloca" in mlir, "Expected quake.alloca in output for test function"
    assert "quake.veq" in mlir, "Expected quake.veq type in output for test function"
    # No jasp types should be present in the output
    validate_quake_mlir(mlir)


def test_quantum_kernel_lowering():
    """Test that a function decorated with @quantum_kernel is correctly lowered."""

    @quantum_kernel
    def test_kernel():
        qv = QuantumVariable(3)
        return measure(qv[0])

    def main():
        res = test_kernel()
        return res

    xdsl_module = _lower(main)

    mlir = str(xdsl_module)
    assert "cudaq.kernel" in mlir
    validate_quake_mlir(mlir)


# ---------------------------------------------------------------------------
# Negative tests: unsupported constructs raise NotImplementedError
# ---------------------------------------------------------------------------


def test_parity_raises_not_implemented_error():
    """jasp.parity has no Quake equivalent; lowering it must raise NotImplementedError."""
    from qrisp.jasp import parity

    def circuit():
        qv = QuantumVariable(2)
        h(qv[0])
        x(qv[1])
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        return parity(m0, m1)

    with pytest.raises(NotImplementedError, match="jasp.parity"):
        _lower(circuit)


def test_unsupported_gate_raises_not_implemented_error():
    """An unknown/unsupported gate raises NotImplementedError during lowering
    (rather than emitting a warning and leaving the op in place)."""
    from unittest.mock import patch

    def circuit():
        qv = QuantumVariable(1)
        h(qv[0])
        return qv

    with patch(
        "qrisp.jasp.cudaq_interface.quake_lowering.jasp_to_quake.pass1a_lower_jasp_to_quake.get_gate_info",
        return_value=None,
    ):
        with pytest.raises(NotImplementedError, match="Unsupported Jasp gate"):
            _lower(circuit)
