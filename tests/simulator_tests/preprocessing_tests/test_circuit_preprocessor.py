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

from __future__ import annotations

import numpy as np

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.simulator.preprocessing.circuit_preprocessing import _circuit_preprocessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_circuit(n_qubits: int, depth: int, seed: int) -> QuantumCircuit:
    """Build a reproducible random circuit of H/CX/RZ gates for equivalence checks."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    for _ in range(depth):
        gate_type = rng.integers(0, 3)
        if gate_type == 0:
            qc.h(int(rng.integers(0, n_qubits)))
        elif gate_type == 1 and n_qubits >= 2:
            a, b = rng.choice(n_qubits, size=2, replace=False)
            qc.cx(int(a), int(b))
        else:
            qc.rz(float(rng.uniform(0, 2 * np.pi)), int(rng.integers(0, n_qubits)))
    return qc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCircuitPreprocessorEdgeCases:
    def test_empty_circuit_returns_copy(self):
        """An empty circuit is returned as an independent copy, untouched."""
        qc = QuantumCircuit(2)
        result = _circuit_preprocessor(qc)
        assert len(result.data) == 0
        assert result is not qc


class TestCircuitPreprocessorUnitaryEquivalence:
    """For a purely unitary circuit, the full pipeline must not change the
    computed unitary, regardless of grouping/reordering internals."""

    def test_random_circuit_unitary_preserved(self):
        """The full pipeline must not change the computed unitary."""
        qc = _random_circuit(n_qubits=5, depth=25, seed=7)
        result = _circuit_preprocessor(qc)
        assert np.allclose(qc.get_unitary(), result.get_unitary(), atol=1e-6)

    def test_random_circuit_unitary_preserved_multiple_seeds(self):
        """Unitary equivalence holds across several random circuits."""
        for seed in range(5):
            qc = _random_circuit(n_qubits=4, depth=20, seed=seed)
            result = _circuit_preprocessor(qc)
            assert np.allclose(qc.get_unitary(), result.get_unitary(), atol=1e-6)


class TestCircuitPreprocessorMeasurements:
    def test_measurement_count_preserved(self):
        """The pipeline must not add or drop measurement instructions."""
        qc = QuantumCircuit(3, 1)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0, 0)
        qc.h(2)
        result = _circuit_preprocessor(qc)
        assert sum(1 for instr in result.data if instr.op.name == "measure") == 1


class TestCircuitPreprocessorDisentangling:
    """Automatic disentangling only kicks in above the qubit-count threshold."""

    def test_disentangling_triggers_above_threshold(self):
        """Circuits with more than 45 qubits get automatic disentangling."""
        qc = QuantumCircuit(50)
        qc.h(0)
        for i in range(49):
            qc.cx(i, i + 1)
        result = _circuit_preprocessor(qc)
        assert any(instr.op.name == "disentangle" for instr in result.data)

    def test_disentangling_does_not_trigger_below_threshold(self):
        """Circuits with 45 or fewer qubits skip automatic disentangling."""
        qc = QuantumCircuit(10)
        qc.h(0)
        for i in range(9):
            qc.cx(i, i + 1)
        result = _circuit_preprocessor(qc)
        assert not any(instr.op.name == "disentangle" for instr in result.data)
