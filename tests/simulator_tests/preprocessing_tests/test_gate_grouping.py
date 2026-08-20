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
from qrisp.simulator.preprocessing.gate_grouping import IntegerCircuit, group_qc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _num_gates(qc: QuantumCircuit) -> int:
    return len(qc.data)


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


class TestGroupQcStructuralInvariants:
    """group_qc must not add, drop, or illegally reorder any instruction."""

    def test_empty_circuit(self):
        """An empty circuit stays empty."""
        qc = QuantumCircuit(3)
        result = group_qc(qc)
        assert _num_gates(result) == 0

    def test_single_gate_untouched(self):
        """A single gate has nothing to group with."""
        qc = QuantumCircuit(1)
        qc.h(0)
        result = group_qc(qc)
        assert _num_gates(result) == 1

    def test_disjoint_qubits_are_not_merged(self):
        """Gates on entirely disjoint qubits never share a group (no connecting instruction)."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.x(1)
        qc.y(2)
        qc.z(3)
        result = group_qc(qc)
        assert _num_gates(result) == _num_gates(qc)

    def test_connected_chain_is_merged(self):
        """Gates connected via shared qubits are merged into a single group."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.h(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        result = group_qc(qc)
        assert _num_gates(result) == 1

    def test_measurement_blocks_grouping_across_it(self):
        """A measurement must never be absorbed into a unitary group."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.x(0)
        result = group_qc(qc)
        names = [instr.op.name for instr in result.data]
        assert names.count("measure") == 1

    def test_qubit_set_preserved(self):
        """Grouping must not add, drop, or rename qubits."""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(2, 3)
        result = group_qc(qc)
        assert set(result.qubits) == set(qc.qubits)


# ---------------------------------------------------------------------------
# Tests: semantic equivalence
# ---------------------------------------------------------------------------


class TestGroupQcUnitaryEquivalence:
    """The grouped circuit must compute exactly the same unitary."""

    def test_random_circuit_unitary_preserved(self):
        """Grouping a random circuit must not change its unitary."""
        qc = _random_circuit(n_qubits=5, depth=30, seed=42)
        result = group_qc(qc)
        assert np.allclose(qc.get_unitary(), result.get_unitary(), atol=1e-6)

    def test_random_circuit_unitary_preserved_multiple_seeds(self):
        """Unitary equivalence holds across several random circuits."""
        for seed in range(5):
            qc = _random_circuit(n_qubits=4, depth=20, seed=seed)
            result = group_qc(qc)
            assert np.allclose(qc.get_unitary(), result.get_unitary(), atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: dual-path (scalar vs. chunked bitmask) regression coverage
# ---------------------------------------------------------------------------


class TestIntegerCircuitDualPath:
    """The scalar (<63 qubits) and chunked (>=63 qubits) bitmask paths must agree."""

    def test_scalar_path_below_threshold(self):
        """Below 63 qubits, the scalar int64 bitmask path is used."""
        qc = QuantumCircuit(62)
        qc.h(0)
        assert IntegerCircuit(qc).use_chunks is False

    def test_chunked_path_at_and_above_threshold(self):
        """At 63 qubits and above, the chunked bitmask path is used."""
        qc = QuantumCircuit(63)
        qc.h(0)
        assert IntegerCircuit(qc).use_chunks is True

    def test_grouping_reduces_instruction_count_across_the_boundary(self):
        """A fully-connected chain circuit is grouped down to fewer
        instructions regardless of whether it uses the scalar or chunked
        bitmask path."""
        for n in (10, 62, 63, 64, 130):
            qc = QuantumCircuit(n)
            for i in range(n):
                qc.h(i)
            for i in range(n - 1):
                qc.cx(i, i + 1)
            result = group_qc(qc)
            assert _num_gates(result) < _num_gates(qc)
