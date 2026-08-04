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

from qrisp.circuit import QuantumCircuit
from qrisp.circuit.pass_management.passes import compress_layers


def _gate_names(qc: QuantumCircuit) -> list[str]:
    """Return the ordered list of operation names in the circuit."""
    return [instr.op.name for instr in qc.data]


def _layer_positions(qc: QuantumCircuit) -> dict[str, int]:
    """Return a mapping from ``"op_name@qubit_id"`` → instruction index."""
    result = {}
    for i, instr in enumerate(qc.data):
        for q in instr.qubits:
            key = f"{instr.op.name}@{q.identifier}"
            result[key] = i
    return result


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


class TestCompressLayersBasic:
    """Smoke tests — basic invariants."""

    def test_returns_quantum_circuit(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = compress_layers(qc)
        assert isinstance(result, QuantumCircuit)

    def test_empty_circuit(self):
        qc = QuantumCircuit(3)
        result = compress_layers(qc)
        assert len(result.data) == 0

    def test_instruction_count_preserved(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(1, 2)
        result = compress_layers(qc)
        assert len(result.data) == len(qc.data)

    def test_gate_multiset_preserved(self):
        """compress_layers must not add or remove gates, only reorder."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(2, 3)
        qc.z(0)
        qc.measure(qc.qubits)

        before_names = sorted(instr.op.name for instr in qc.data)
        result = compress_layers(qc)
        after_names = sorted(instr.op.name for instr in result.data)

        assert before_names == after_names


# ---------------------------------------------------------------------------
# Reordering behaviour
# ---------------------------------------------------------------------------


class TestCompressLayersReordering:
    """Verify that disjoint gates are pulled into earlier layers."""

    def test_independent_single_qubit_gates_compacted(self):
        """h on q2 should move before cx(0,1) since it shares no qubits."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(2)       # independent
        qc.cx(1, 2)

        result = compress_layers(qc)
        names = _gate_names(result)
        # h(2) must appear before cx(0,1)
        assert names.index("h") < names.index("cx"), f"h(2) not before first cx: {names}"

    def test_independent_cx_gates_compacted(self):
        """CX gates on disjoint qubit pairs should move to the same layer."""
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(2, 3)   # h(2) and cx(2,3) share qubit 2 → must stay ordered

        result = compress_layers(qc)
        names = _gate_names(result)
        # cx(0,1) is independent of h(2) and can move before it
        assert names == ["cx", "h", "cx"], f"Unexpected order: {names}"

    def test_relative_order_on_same_qubit_preserved(self):
        """Operations on the same qubit must stay in original order."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.x(0)
        qc.z(0)
        qc.h(0)

        result = compress_layers(qc)
        names = _gate_names(result)
        # On a single qubit nothing can move — must be identical
        assert names == ["h", "x", "z", "h"], f"Single-qubit order changed: {names}"


# ---------------------------------------------------------------------------
# Bookkeeping instructions (qb_alloc / qb_dealloc)
# ---------------------------------------------------------------------------


class TestCompressLayersBookkeeping:
    """Verify that qb_alloc / qb_dealloc are excluded from scheduling."""

    def test_qb_alloc_does_not_delay_real_gates(self):
        """Alloc instructions must not push real gates to later layers."""
        from qrisp.circuit.standard_operations import QubitAlloc
        from qrisp.circuit import Qubit

        qc = QuantumCircuit(3)
        # Manually append qb_alloc instructions interleaved with gates
        q0, q1, q2 = qc.qubits

        qc.append(QubitAlloc(), [q0])
        qc.h(0)
        qc.append(QubitAlloc(), [q1])
        qc.cx(0, 1)
        qc.append(QubitAlloc(), [q2])
        qc.h(2)

        result = compress_layers(qc)
        names = _gate_names(result)

        # All qb_alloc should be at the very beginning
        alloc_indices = [i for i, n in enumerate(names) if n == "qb_alloc"]
        assert alloc_indices == [0, 1, 2], f"Allocs not at front: {alloc_indices}"

        # Real gates h, cx, h — h(2) is independent of cx(0,1) and moves up
        real_names = [n for n in names if n != "qb_alloc"]
        assert real_names == ["h", "h", "cx"], f"Real gate order broken: {real_names}"


# ---------------------------------------------------------------------------
# Correctness: unitary and measurement statistics
# ---------------------------------------------------------------------------


class TestCompressLayersCorrectness:
    """Verify compress_layers preserves circuit semantics."""

    def test_unitary_preserved_simple(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(1, 2)
        assert compress_layers.compare_unitary(qc)

    def test_unitary_preserved_layered(self):
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.h(i)
        for i in range(3):
            qc.cx(i, i + 1)
        assert compress_layers.compare_unitary(qc)

    def test_measurement_statistics_preserved(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(1, 2)
        qc.measure(qc.qubits)
        assert compress_layers.compare_measurement(qc)

    def test_unitary_preserved_with_swaps(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.swap(0, 1)
        qc.cx(1, 2)
        qc.h(2)
        assert compress_layers.compare_unitary(qc)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestCompressLayersEdgeCases:
    """Corner cases and robustness."""

    def test_idempotent(self):
        """Applying twice should give the same result as once."""
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.h(i)
        for i in range(3):
            qc.cx(i, i + 1)

        once = compress_layers(qc)
        twice = compress_layers(once)
        assert _gate_names(once) == _gate_names(twice)

    def test_all_independent_gates_stay_at_front(self):
        """All single-qubit gates on distinct qubits should be at the front."""
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.h(i)
        qc.cx(0, 1)

        result = compress_layers(qc)
        names = _gate_names(result)
        # All four h gates should come before the cx
        last_h = max(i for i, n in enumerate(names) if n == "h")
        first_cx = names.index("cx")
        assert last_h < first_cx, f"h gates not all before cx: {names}"
