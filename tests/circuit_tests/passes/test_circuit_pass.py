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

    def test_cancel_inverses_preserves_unitary(self):
        """cancel_inverses removes inverse pairs and should preserve the unitary."""
        from qrisp.circuit.passes.cancel_inverses import cancel_inverses

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(0)
        qc.x(0)  # X·X = I — cancels
        qc.cx(0, 1)
        qc.cx(0, 1)  # CX·CX = I — cancels
        qc.h(1)

        cp = CircuitPass(cancel_inverses)
        assert cp.compare_unitary(qc)

    def test_combine_single_qubit_gates_preserves_unitary(self):
        """combine_single_qubit_gates merges adjacent gates and should preserve the unitary."""
        from qrisp.circuit.passes.combine_single_qubit_gates import combine_single_qubit_gates

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.z(0)
        qc.x(0)
        qc.cx(0, 1)
        qc.h(1)

        cp = CircuitPass(combine_single_qubit_gates)
        assert cp.compare_unitary(qc)

    def test_convert_to_cz_preserves_unitary(self):
        """convert_to_cz rewrites CX→CZ·H and should preserve the unitary.
        Note: convert_to_cz is a factory that returns a CircuitPass when called."""
        from qrisp.circuit.passes.convert_to_cz import convert_to_cz

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)

        cp = CircuitPass(convert_to_cz())
        assert cp.compare_unitary(qc)

    # --- Mutation safety ---

    def test_original_circuit_not_mutated(self, two_qubit_qc):
        """compare_unitary must not modify the input circuit."""
        original_data_len = len(two_qubit_qc.data)
        cp = CircuitPass(identity_pass)
        cp.compare_unitary(two_qubit_qc)
        assert len(two_qubit_qc.data) == original_data_len


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestCircuitPassImports:
    """Ensure CircuitPass is importable from all expected locations."""

    def test_import_from_qrisp_top_level(self):
        from qrisp import CircuitPass as CP
        assert CP is CircuitPass

    def test_import_from_qrisp_circuit_passes(self):
        from qrisp.circuit.passes import CircuitPass as CP
        assert CP is CircuitPass

    def test_import_from_leaf_module(self):
        from qrisp.circuit.passes.pass_manager import CircuitPass as CP
        assert CP is CircuitPass
