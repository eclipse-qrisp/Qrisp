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
from qrisp.simulator.preprocessing.circuit_reordering import reorder_circuit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _per_qubit_sequence(qc: QuantumCircuit) -> dict:
    """Map each qubit index to the ordered sequence of (op name, qubit indices)
    of the instructions touching it, used to check causal order is preserved."""
    seqs: dict = {i: [] for i in range(len(qc.qubits))}
    for instr in qc.data:
        key = (instr.op.name, tuple(qc.qubits.index(q) for q in instr.qubits))
        for qb in instr.qubits:
            seqs[qc.qubits.index(qb)].append(key)
    return seqs


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
# Tests: structural invariants
# ---------------------------------------------------------------------------


class TestReorderCircuitStructuralInvariants:
    def test_instruction_count_preserved(self):
        """Reordering only permutes instructions, never adds or drops any."""
        qc = _random_circuit(4, 20, seed=1)
        result = reorder_circuit(qc, [])
        assert len(result.data) == len(qc.data)

    def test_causal_order_preserved_per_qubit(self):
        """Reordering must never change the relative order of instructions
        that share a qubit, only how independent instructions interleave."""
        qc = QuantumCircuit(2, 1)
        qc.h(1)
        qc.h(0)
        qc.measure(0, 0)
        qc.x(1)

        result = reorder_circuit(qc, ["measure", "reset", "disentangle"])
        assert _per_qubit_sequence(result) == _per_qubit_sequence(qc)

    def test_empty_circuit(self):
        """An empty circuit stays empty."""
        qc = QuantumCircuit(2)
        result = reorder_circuit(qc, [])
        assert len(result.data) == 0


# ---------------------------------------------------------------------------
# Tests: semantic equivalence
# ---------------------------------------------------------------------------


class TestReorderCircuitUnitaryEquivalence:
    def test_random_circuit_unitary_preserved(self):
        """Reordering a random circuit must not change its unitary."""
        qc = _random_circuit(4, 20, seed=1)
        result = reorder_circuit(qc, [])
        assert np.allclose(qc.get_unitary(), result.get_unitary(), atol=1e-6)

    def test_random_circuit_unitary_preserved_multiple_seeds(self):
        """Unitary equivalence holds across several random circuits."""
        for seed in range(5):
            qc = _random_circuit(4, 20, seed=seed)
            result = reorder_circuit(qc, [])
            assert np.allclose(qc.get_unitary(), result.get_unitary(), atol=1e-6)
