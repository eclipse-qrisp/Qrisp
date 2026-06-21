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

from __future__ import annotations

import numpy as np
import pytest
from qrisp import CircuitPass
from qrisp.circuit.quantum_circuit import QuantumCircuit


# ---------------------------------------------------------------------------
# Simple pass helpers
# ---------------------------------------------------------------------------


def identity_pass(qc: QuantumCircuit) -> QuantumCircuit:
    """Return the circuit unchanged."""
    return qc


def add_h_pass(qc: QuantumCircuit) -> QuantumCircuit:
    """Append an H gate on qubit 0."""
    qc.h(qc.qubits[0])
    return qc


def add_cx_pass(qc: QuantumCircuit) -> QuantumCircuit:
    """Append a CX gate on qubits 0 and 1."""
    qc.cx(qc.qubits[0], qc.qubits[1])
    return qc


def return_int_pass(qc: QuantumCircuit) -> int:
    """Return an int instead of a QuantumCircuit (invalid)."""
    return 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_qubit_qc() -> QuantumCircuit:
    """A basic entangled 2-qubit circuit: H(0), CX(0,1)."""
    qc = QuantumCircuit(2)
    qc.h(qc.qubits[0])
    qc.cx(qc.qubits[0], qc.qubits[1])
    return qc


@pytest.fixture
def single_qubit_qc() -> QuantumCircuit:
    """A single qubit in superposition."""
    qc = QuantumCircuit(1)
    qc.h(qc.qubits[0])
    return qc


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestCircuitPassConstruction:
    """Test CircuitPass construction as decorator and explicit wrapper."""

    def test_constructor_stores_func(self):
        cp = CircuitPass(identity_pass)
        assert cp._func is identity_pass

    def test_constructor_copies_name(self):
        cp = CircuitPass(identity_pass)
        assert cp.__name__ == "identity_pass"

    def test_constructor_copies_doc(self):
        cp = CircuitPass(identity_pass)
        assert cp.__doc__ == "Return the circuit unchanged."

    def test_constructor_sets_wrapped(self):
        cp = CircuitPass(identity_pass)
        assert cp.__wrapped__ is identity_pass

    def test_bare_decorator_syntax(self):
        @CircuitPass
        def my_pass(qc):
            """A decorated pass."""
            return qc

        assert isinstance(my_pass, CircuitPass)
        assert my_pass.__name__ == "my_pass"
        assert my_pass.__doc__ == "A decorated pass."
        assert my_pass.__wrapped__.__name__ == "my_pass"


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------


class TestCircuitPassCall:
    """Test invoking CircuitPass instances."""

    def test_call_identity_returns_same_qc(self, two_qubit_qc):
        cp = CircuitPass(identity_pass)
        result = cp(two_qubit_qc)
        assert result is two_qubit_qc

    def test_call_applies_wrapped_function(self, two_qubit_qc):
        cp = CircuitPass(add_h_pass)
        initial = len(two_qubit_qc.data)
        result = cp(two_qubit_qc)
        assert len(result.data) == initial + 1
        # The appended gate should be an H on qubit 0
        assert result.data[-1].op.name == "h"

    def test_call_passes_kwargs_is_equivalent_to_preconfiguration(self):
        """If a pass factory pre-configures kwargs at creation time,
        the resulting CircuitPass should work with a single positional arg."""

        def make_double_h_pass():
            @CircuitPass
            def _double_h_pass(qc):
                return qc

            return _double_h_pass

        cp = make_double_h_pass()
        qc = QuantumCircuit(1)
        result = cp(qc)
        assert result is qc

    # --- Input type guards ---

    def test_call_with_keyword_argument_raises_typeerror(self, two_qubit_qc):
        cp = CircuitPass(identity_pass)
        with pytest.raises(TypeError, match="expected exactly one positional"):
            cp(qc=two_qubit_qc)

    def test_call_with_zero_args_raises_typeerror(self):
        cp = CircuitPass(identity_pass)
        with pytest.raises(TypeError, match="expected exactly one positional"):
            cp()

    def test_call_with_two_args_raises_typeerror(self, two_qubit_qc):
        cp = CircuitPass(identity_pass)
        with pytest.raises(TypeError, match="expected exactly one positional"):
            cp(two_qubit_qc, two_qubit_qc)

    def test_call_with_non_qc_arg_raises_typeerror(self):
        cp = CircuitPass(identity_pass)
        with pytest.raises(TypeError, match="expected a QuantumCircuit"):
            cp(42)

    # --- Output type guards ---

    def test_call_raises_typeerror_when_wrapped_returns_non_qc(self):
        cp = CircuitPass(return_int_pass)
        qc = QuantumCircuit(1)
        with pytest.raises(TypeError, match="expected the wrapped function to return a QuantumCircuit"):
            cp(qc)


# ---------------------------------------------------------------------------
# compare_unitary
# ---------------------------------------------------------------------------


class TestCircuitPassCompareUnitary:
    """Test verification that a CircuitPass preserves the unitary."""

    # --- Basic correctness ---

    def test_identity_pass_preserves_unitary(self, two_qubit_qc):
        cp = CircuitPass(identity_pass)
        assert cp.compare_unitary(two_qubit_qc)

    def test_pass_adding_gate_breaks_unitary(self, two_qubit_qc):
        cp = CircuitPass(add_cx_pass)
        assert not cp.compare_unitary(two_qubit_qc)

    def test_precision_parameter(self, two_qubit_qc):
        """With sufficiently loose precision even a wrong unitary should match."""
        cp = CircuitPass(add_cx_pass)
        assert cp.compare_unitary(two_qubit_qc, precision=-5)

    def test_ignore_gphase_true_identity(self, two_qubit_qc):
        """ignore_gphase=True works correctly with an identity pass."""
        cp = CircuitPass(identity_pass)
        assert cp.compare_unitary(two_qubit_qc, ignore_gphase=True)

    def test_ignore_gphase_false_modifying_pass(self, two_qubit_qc):
        """A pass that changes the unitary should return False regardless
        of ignore_gphase, because the difference is not just a global phase."""
        cp = CircuitPass(add_cx_pass)
        assert not cp.compare_unitary(two_qubit_qc, ignore_gphase=False)
        assert not cp.compare_unitary(two_qubit_qc, ignore_gphase=True)

    def test_different_qubit_count_returns_false(self):
        """Transforming a 1-qubit circuit into a 2-qubit circuit."""

        def add_qubit_pass(qc: QuantumCircuit) -> QuantumCircuit:
            qc = qc.copy()
            qc.add_qubit()
            return qc

        cp = CircuitPass(add_qubit_pass)
        qc = QuantumCircuit(1)
        qc.h(qc.qubits[0])
        assert not cp.compare_unitary(qc)

    def test_empty_circuit_identity_pass(self):
        qc = QuantumCircuit(2)
        cp = CircuitPass(identity_pass)
        assert cp.compare_unitary(qc)

    # --- Input validation ---

    def test_non_qc_raises_typeerror(self):
        cp = CircuitPass(identity_pass)
        with pytest.raises(TypeError, match="Expected a QuantumCircuit"):
            cp.compare_unitary(42)

    def test_input_with_measurement_raises_valueerror(self):
        qc = QuantumCircuit(1)
        qc.add_clbit()
        qc.h(qc.qubits[0])
        qc.measure(qc.qubits[0], qc.clbits[0])

        cp = CircuitPass(identity_pass)
        with pytest.raises(ValueError, match="input circuit contains measurement"):
            cp.compare_unitary(qc)

    def test_pass_introducing_measurement_raises_valueerror(self):
        """A pass that injects a measurement into the circuit should raise."""

        def measure_pass(qc: QuantumCircuit) -> QuantumCircuit:
            qc = qc.copy()
            qc.add_clbit()
            qc.measure(qc.qubits[0], qc.clbits[0])
            return qc

        cp = CircuitPass(measure_pass)
        qc = QuantumCircuit(1)
        qc.h(qc.qubits[0])
        with pytest.raises(ValueError, match="transformed circuit contains measurement"):
            cp.compare_unitary(qc)

    # --- Integration: real passes ---

    def test_fuse_adjacents_preserves_unitary(self):
        """fuse_adjacents removes inverse pairs and should preserve the unitary."""
        from qrisp.circuit.pass_management.passes.fuse_adjacents import fuse_adjacents

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(0)
        qc.x(0)  # X·X = I — cancels
        qc.cx(0, 1)
        qc.cx(0, 1)  # CX·CX = I — cancels
        qc.h(1)

        cp = fuse_adjacents
        assert cp.compare_unitary(qc)

    def test_combine_single_qubit_gates_preserves_unitary(self):
        """combine_single_qubit_gates merges adjacent gates and should preserve the unitary."""
        from qrisp.circuit.pass_management.passes.combine_single_qubit_gates import combine_single_qubit_gates

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.z(0)
        qc.x(0)
        qc.cx(0, 1)
        qc.h(1)

        cp = combine_single_qubit_gates
        assert cp.compare_unitary(qc)

    def test_convert_to_cz_preserves_unitary(self):
        """convert_to_cz rewrites CX→CZ·H and should preserve the unitary."""
        from qrisp.circuit.pass_management.passes.convert_to_cz import convert_to_cz

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)

        cp = convert_to_cz()
        assert cp.compare_unitary(qc)

    def test_convert_to_cx_preserves_unitary(self):
        """convert_to_cx rewrites CZ→CX·H and should preserve the unitary."""
        from qrisp.circuit.pass_management.passes.convert_to_cx import convert_to_cx

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cz(0, 1)
        qc.h(1)

        cp = convert_to_cx()
        assert cp.compare_unitary(qc)

    # --- Mutation safety ---

    def test_original_circuit_not_mutated(self, two_qubit_qc):
        """compare_unitary must not modify the input circuit."""
        original_data_len = len(two_qubit_qc.data)
        cp = CircuitPass(identity_pass)
        cp.compare_unitary(two_qubit_qc)
        assert len(two_qubit_qc.data) == original_data_len


# ---------------------------------------------------------------------------
# compare_measurement
# ---------------------------------------------------------------------------


@pytest.fixture
def ghz_measured() -> QuantumCircuit:
    """A measured 3-qubit GHZ circuit: |000>+|111> with equal probability."""
    qc = QuantumCircuit(3)
    for _ in range(3):
        qc.add_clbit()
    qc.h(qc.qubits[0])
    qc.cx(qc.qubits[0], qc.qubits[1])
    qc.cx(qc.qubits[1], qc.qubits[2])
    qc.measure(qc.qubits, qc.clbits)
    return qc


@pytest.fixture
def equal_superposition_measured() -> QuantumCircuit:
    """A measured 2-qubit equal superposition: all outcomes p=0.25."""
    qc = QuantumCircuit(2)
    for _ in range(2):
        qc.add_clbit()
    qc.h(qc.qubits[0])
    qc.h(qc.qubits[1])
    qc.measure(qc.qubits, qc.clbits)
    return qc


class TestCircuitPassCompareMeasurement:
    """Test verification that a CircuitPass preserves measurement statistics."""

    # --- Basic correctness ---

    def test_identity_pass_preserves_distribution(self, ghz_measured):
        cp = CircuitPass(identity_pass)
        assert cp.compare_measurement(ghz_measured)

    def test_identity_pass_empty_circuit(self):
        qc = QuantumCircuit(2)
        cp = CircuitPass(identity_pass)
        assert cp.compare_measurement(qc)

    def test_pass_that_adds_x_flips_distribution(self, ghz_measured):
        """Adding an X gate before measurement changes the measurement outcomes."""

        def add_x_before_measure(qc: QuantumCircuit) -> QuantumCircuit:
            from qrisp.circuit import Instruction

            # Insert X on qubit 0 just before the measurement instructions
            new_qc = qc.clearcopy()
            for instr in qc.data:
                if instr.op.name == "measure":
                    new_qc.x(qc.qubits[0])
                new_qc.append(instr.op, instr.qubits, instr.clbits)
            return new_qc

        cp = CircuitPass(add_x_before_measure)
        assert not cp.compare_measurement(ghz_measured)

    def test_pass_that_adds_h_changes_distribution(self, ghz_measured):
        """Adding an H gate before measurement changes the measurement statistics."""

        def add_h_before_measure(qc: QuantumCircuit) -> QuantumCircuit:
            new_qc = qc.clearcopy()
            for instr in qc.data:
                if instr.op.name == "measure":
                    new_qc.h(qc.qubits[0])
                new_qc.append(instr.op, instr.qubits, instr.clbits)
            return new_qc

        cp = CircuitPass(add_h_before_measure)
        assert not cp.compare_measurement(ghz_measured)

    # --- Precision parameter ---

    def test_precision_loose_enough(self, equal_superposition_measured):
        """With precision=-1 (tolerance 10), even a wrong distribution 'passes'."""

        def add_x_pass(qc: QuantumCircuit) -> QuantumCircuit:
            qc = qc.copy()
            qc.x(qc.qubits[0])
            return qc

        cp = CircuitPass(add_x_pass)
        assert cp.compare_measurement(equal_superposition_measured, precision=-1)

    def test_precision_tight(self, equal_superposition_measured):
        """With very tight precision, identity still passes."""
        cp = CircuitPass(identity_pass)
        assert cp.compare_measurement(equal_superposition_measured, precision=10)

    # --- Input validation ---

    def test_non_qc_raises_typeerror(self):
        cp = CircuitPass(identity_pass)
        with pytest.raises(TypeError, match="Expected a QuantumCircuit"):
            cp.compare_measurement(42)

    def test_no_measurements_identity_passes(self):
        """A circuit without measurements should compare equal to itself."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        cp = CircuitPass(identity_pass)
        assert cp.compare_measurement(qc)

    def test_no_measurements_empty_circuit(self):
        """An empty circuit (no qubits touched) should compare equal to itself."""
        qc = QuantumCircuit(2)
        cp = CircuitPass(identity_pass)
        assert cp.compare_measurement(qc)

    def test_no_measurements_different_unitaries_still_pass(self):
        """Measurement-based comparison cannot distinguish unitaries without
        measurements — the empty distribution is trivially invariant."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        def add_x_pass(qc_in: QuantumCircuit) -> QuantumCircuit:
            qc_out = qc_in.copy()
            qc_out.x(qc_out.qubits[0])
            return qc_out

        cp = CircuitPass(add_x_pass)
        # All no-measurement circuits produce the same (empty) distribution.
        assert cp.compare_measurement(qc)

    # --- Integration: real passes ---

    def test_fuse_adjacents_measured(self, ghz_measured):
        """fuse_adjacents should preserve measurement statistics."""
        from qrisp.circuit.pass_management.passes.fuse_adjacents import fuse_adjacents

        qc = QuantumCircuit(3)
        for _ in range(3):
            qc.add_clbit()
        qc.h(0)
        qc.cx(0, 1)
        qc.x(1)
        qc.x(1)  # cancels
        qc.cx(1, 2)
        qc.measure(qc.qubits, qc.clbits)

        cp = fuse_adjacents
        assert cp.compare_measurement(qc)

    def test_convert_to_cz_measured(self):
        """convert_to_cz should preserve measurement statistics."""
        from qrisp.circuit.pass_management.passes.convert_to_cz import convert_to_cz

        qc = QuantumCircuit(2)
        for _ in range(2):
            qc.add_clbit()
        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        qc.measure(qc.qubits, qc.clbits)

        cp = CircuitPass(convert_to_cz())
        assert cp.compare_measurement(qc)

    def test_convert_to_cx_measured(self):
        """convert_to_cx should preserve measurement statistics."""
        from qrisp.circuit.pass_management.passes.convert_to_cx import convert_to_cx

        qc = QuantumCircuit(2)
        for _ in range(2):
            qc.add_clbit()
        qc.h(0)
        qc.cz(0, 1)
        qc.h(1)
        qc.measure(qc.qubits, qc.clbits)

        cp = CircuitPass(convert_to_cx())
        assert cp.compare_measurement(qc)

    def test_combine_single_qubit_gates_measured(self):
        """combine_single_qubit_gates should preserve measurement statistics."""
        from qrisp.circuit.pass_management.passes.combine_single_qubit_gates import combine_single_qubit_gates

        qc = QuantumCircuit(2)
        for _ in range(2):
            qc.add_clbit()
        qc.h(0)
        qc.z(0)
        qc.x(0)
        qc.cx(0, 1)
        qc.h(1)
        qc.measure(qc.qubits, qc.clbits)

        cp = combine_single_qubit_gates
        assert cp.compare_measurement(qc)

    # --- Mutation safety ---

    def test_original_circuit_not_mutated(self, ghz_measured):
        """compare_measurement must not modify the input circuit."""
        original_data_len = len(ghz_measured.data)
        cp = CircuitPass(identity_pass)
        cp.compare_measurement(ghz_measured)
        assert len(ghz_measured.data) == original_data_len

    # --- Outcome mismatch detection ---

    def test_detects_new_outcome_key(self):
        """When the pass introduces a completely new measurement outcome,
        compare_measurement should detect it."""

        def swap_qubits_pass(qc: QuantumCircuit) -> QuantumCircuit:
            new_qc = qc.clearcopy()
            for instr in qc.data:
                if instr.op.name == "measure":
                    new_qc.swap(qc.qubits[0], qc.qubits[1])
                new_qc.append(instr.op, instr.qubits, instr.clbits)
            return new_qc

        # Circuit where qubit 0 is always |1> and qubit 1 is always |0>
        qc = QuantumCircuit(2)
        for _ in range(2):
            qc.add_clbit()
        qc.x(qc.qubits[0])
        qc.measure(qc.qubits, qc.clbits)

        cp = CircuitPass(swap_qubits_pass)
        assert not cp.compare_measurement(qc)


# ---------------------------------------------------------------------------
# visualize
# ---------------------------------------------------------------------------


class TestCircuitPassVisualize:
    """Test the visualize() console output method."""

    def test_visualize_does_not_mutate_input(self, two_qubit_qc):
        """visualize must not modify the input circuit."""
        original_data = list(two_qubit_qc.data)
        cp = CircuitPass(identity_pass)
        cp.visualize(two_qubit_qc)
        assert list(two_qubit_qc.data) == original_data

    def test_visualize_output_contains_pass_name(self, capsys, two_qubit_qc):
        """Output must contain the pass name."""

        def my_pass(qc):
            return qc

        cp = CircuitPass(my_pass)
        cp.visualize(two_qubit_qc)

        captured = capsys.readouterr()
        assert "my_pass" in captured.out

    def test_visualize_output_contains_before_after_headers(self, capsys, two_qubit_qc):
        """Output must contain Before and After section headers."""
        cp = CircuitPass(identity_pass)
        cp.visualize(two_qubit_qc)

        captured = capsys.readouterr()
        assert "Before" in captured.out
        assert "After" in captured.out

    def test_visualize_output_contains_banner_separators(self, capsys, two_qubit_qc):
        """Output must contain the === banner and ─── section lines."""
        cp = CircuitPass(identity_pass)
        cp.visualize(two_qubit_qc)

        captured = capsys.readouterr()
        lines = captured.out.splitlines()
        # First line should be the ===== banner
        assert "=" in lines[0]
        # Second line should be the ───── Before ───── section header
        assert "─" in lines[1]
        # Last line should be the ===== footer
        assert "=" in lines[-1]

    def test_visualize_all_lines_same_width(self, capsys, two_qubit_qc):
        """The banner, section headers, and footer must all be the same width."""
        cp = CircuitPass(identity_pass)
        cp.visualize(two_qubit_qc)

        captured = capsys.readouterr()
        lines = captured.out.splitlines()
        # Check banner (line 0), Before (line 1), After, and footer (line -1)
        separator_lines = [lines[0], lines[1], lines[-1]]
        widths = {len(line) for line in separator_lines}
        # The Before/After headers are reconstructed by finding them
        before_idx = next(i for i, l in enumerate(lines) if "Before" in l)
        after_idx = next(i for i, l in enumerate(lines) if "After" in l)
        separator_lines = [lines[0], lines[before_idx], lines[after_idx], lines[-1]]
        widths = {len(line) for line in separator_lines}
        assert len(widths) == 1, f"Separators have mismatched widths: {widths}"

    def test_visualize_after_differs_from_before(self, capsys):
        """When the pass modifies the circuit, before/after outputs differ."""

        def reverse_circuit(qc: QuantumCircuit) -> QuantumCircuit:
            new_qc = qc.clearcopy()
            for instr in reversed(qc.data):
                new_qc.append(instr.op, instr.qubits, instr.clbits)
            return new_qc

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        cp = CircuitPass(reverse_circuit)
        cp.visualize(qc)

        captured = capsys.readouterr()
        before_start = captured.out.find("Before")
        before_end = captured.out.find("After")
        after_start = captured.out.find("After", before_end + 1)

        before_section = captured.out[before_start:before_end]
        after_section = captured.out[after_start:]
        assert before_section != after_section

    def test_visualize_with_real_pass(self, capsys):
        """Integration: visualize with fuse_adjacents on a circuit with CX-CX."""
        from qrisp.circuit.pass_management.passes.fuse_adjacents import fuse_adjacents

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(0, 1)

        fuse_adjacents.visualize(qc)

        captured = capsys.readouterr()
        assert "fuse_adjacents" in captured.out


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestCircuitPassImports:
    """Ensure CircuitPass is importable from all expected locations."""

    def test_import_from_qrisp_top_level(self):
        from qrisp import CircuitPass as CP

        assert CP is CircuitPass

    def test_import_from_qrisp_circuit_passes(self):
        from qrisp.circuit.pass_management import CircuitPass as CP

        assert CP is CircuitPass

    def test_import_from_leaf_module(self):
        from qrisp.circuit.pass_management.circuit_pass import CircuitPass as CP

        assert CP is CircuitPass
