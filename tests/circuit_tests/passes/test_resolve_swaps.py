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
from qrisp.circuit.pass_management.passes.resolve_swaps import resolve_swaps


def _make_circuit(n: int = 4):
    qc = QuantumCircuit()
    qubits = [Qubit(f"q{i}") for i in range(n)]
    for q in qubits:
        qc.add_qubit(q)
    return qc, qubits


def _gate_names(qc) -> list[str]:
    return [instr.op.name for instr in qc.data]


def _qubit_indices(qc, instr) -> list[int]:
    return [qc.qubits.index(q) for q in instr.qubits]


class TestResolveSwapsBasic:
    def test_no_swaps_unchanged(self):
        qc, qubits = _make_circuit(3)
        qc.h(qubits[0])
        qc.cx(qubits[0], qubits[1])
        qc.rz(0.5, qubits[2])
        result = resolve_swaps(qc)
        assert _gate_names(result) == ["h", "cx", "rz"]
        assert _qubit_indices(result, result.data[0]) == [0]
        assert _qubit_indices(result, result.data[1]) == [0, 1]
        assert _qubit_indices(result, result.data[2]) == [2]

    def test_single_swap_removed(self):
        qc, qubits = _make_circuit(2)
        qc.swap(qubits[0], qubits[1])
        result = resolve_swaps(qc)
        assert "swap" not in _gate_names(result)
        assert len(result.data) == 0

    def test_empty_circuit(self):
        qc = QuantumCircuit(3)
        result = resolve_swaps(qc)
        assert len(result.data) == 0

    def test_qubit_count_preserved(self):
        qc, qubits = _make_circuit(5)
        qc.swap(qubits[1], qubits[3])
        qc.h(qubits[1])
        result = resolve_swaps(qc)
        assert result.num_qubits() == 5


class TestResolveSwapsPermutation:
    def test_swap_remaps_subsequent_cx(self):
        qc, qubits = _make_circuit(3)
        qc.swap(qubits[0], qubits[1])
        qc.cx(qubits[0], qubits[1])
        result = resolve_swaps(qc)
        assert _gate_names(result) == ["cx"]
        assert _qubit_indices(result, result.data[0]) == [1, 0]

    def test_swap_does_not_affect_prior_gates(self):
        qc, qubits = _make_circuit(3)
        qc.h(qubits[0])
        qc.swap(qubits[0], qubits[1])
        qc.h(qubits[0])
        result = resolve_swaps(qc)
        assert _gate_names(result) == ["h", "h"]
        assert _qubit_indices(result, result.data[0]) == [0]
        assert _qubit_indices(result, result.data[1]) == [1]

    def test_two_consecutive_swaps_cancel(self):
        qc, qubits = _make_circuit(2)
        qc.swap(qubits[0], qubits[1])
        qc.swap(qubits[0], qubits[1])
        qc.cx(qubits[0], qubits[1])
        result = resolve_swaps(qc)
        assert _gate_names(result) == ["cx"]
        assert _qubit_indices(result, result.data[0]) == [0, 1]

    def test_chain_of_swaps(self):
        qc, qubits = _make_circuit(3)
        # After SWAP(0,1): perm = [1, 0, 2]
        qc.swap(qubits[0], qubits[1])
        # After SWAP(1,2): perm = [1, 2, 0]
        qc.swap(qubits[1], qubits[2])
        qc.h(qubits[0])
        qc.cx(qubits[1], qubits[2])
        result = resolve_swaps(qc)
        assert _gate_names(result) == ["h", "cx"]
        assert _qubit_indices(result, result.data[0]) == [1]
        assert _qubit_indices(result, result.data[1]) == [2, 0]

    def test_swap_with_single_qubit_gates(self):
        qc, qubits = _make_circuit(2)
        qc.x(qubits[0])
        qc.swap(qubits[0], qubits[1])
        qc.z(qubits[0])
        qc.swap(qubits[0], qubits[1])
        qc.y(qubits[1])
        result = resolve_swaps(qc)
        assert _gate_names(result) == ["x", "z", "y"]
        assert _qubit_indices(result, result.data[0]) == [0]
        assert _qubit_indices(result, result.data[1]) == [1]
        assert _qubit_indices(result, result.data[2]) == [1]

    def test_unitary_equivalence(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.swap(0, 2)
        qc.cx(0, 1)
        qc.rz(0.7, 2)
        result = resolve_swaps(qc)
        assert _gate_names(result) == ["h", "cx", "cx", "rz"]
        assert _qubit_indices(result, result.data[0]) == [0]
        assert _qubit_indices(result, result.data[1]) == [0, 1]
        assert _qubit_indices(result, result.data[2]) == [2, 1]
        assert _qubit_indices(result, result.data[3]) == [0]
