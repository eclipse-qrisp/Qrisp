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

from qrisp import QuantumCircuit, Qubit
from qrisp.circuit.pass_management.passes.remove_barriers import remove_barriers


def _make_circuit(n: int = 2):
    qc = QuantumCircuit()
    qubits = [Qubit(f"q{i}") for i in range(n)]
    for q in qubits:
        qc.add_qubit(q)
    return qc, qubits


class TestRemoveBarriers:
    def test_barriers_removed(self):
        qc, qubits = _make_circuit(2)
        qc.h(qubits[0])
        qc.barrier(qubits)
        qc.cx(qubits[0], qubits[1])
        result = remove_barriers(qc)
        names = [i.op.name for i in result.data]
        assert "barrier" not in names
        assert names == ["h", "cx"]

    def test_non_barrier_gates_preserved(self):
        qc, qubits = _make_circuit(2)
        qc.h(qubits[0])
        qc.x(qubits[1])
        result = remove_barriers(qc)
        names = [i.op.name for i in result.data]
        assert names == ["h", "x"]

    def test_only_barriers(self):
        qc, qubits = _make_circuit(2)
        qc.barrier(qubits)
        qc.barrier(qubits)
        result = remove_barriers(qc)
        assert len(result.data) == 0

    def test_empty_circuit(self):
        qc, _ = _make_circuit(2)
        result = remove_barriers(qc)
        assert len(result.data) == 0
        assert result.num_qubits() == 2

    def test_multiple_barriers_removed(self):
        qc, qubits = _make_circuit(2)
        qc.h(qubits[0])
        qc.barrier(qubits)
        qc.h(qubits[1])
        qc.barrier(qubits)
        qc.cx(qubits[0], qubits[1])
        result = remove_barriers(qc)
        names = [i.op.name for i in result.data]
        assert names == ["h", "h", "cx"]

    def test_original_not_mutated(self):
        qc, qubits = _make_circuit(2)
        qc.h(qubits[0])
        qc.barrier(qubits)
        original_len = len(qc.data)
        remove_barriers(qc)
        assert len(qc.data) == original_len

    def test_qubit_count_preserved(self):
        qc, qubits = _make_circuit(3)
        qc.barrier(qubits)
        result = remove_barriers(qc)
        assert result.num_qubits() == 3


# ---------------------------------------------------------------------------
# Correctness: compare_unitary
# ---------------------------------------------------------------------------


class TestRemoveBarriersCorrectness:
    """Verify remove_barriers preserves the circuit unitary."""

    def test_unitary_preserved_simple(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)
        qc.barrier()
        qc.h(1)
        assert remove_barriers.compare_unitary(qc)

    def test_unitary_preserved_multiple_barriers(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.cz(1, 2)
        qc.barrier()
        qc.x(2)
        assert remove_barriers.compare_unitary(qc)

    def test_unitary_preserved_no_barriers(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        assert remove_barriers.compare_unitary(qc)
