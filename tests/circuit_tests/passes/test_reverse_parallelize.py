"""Tests for qrisp.circuit.passes.reverse_parallelize."""

import pytest

from qrisp.circuit import QuantumCircuit
from qrisp.circuit.pass_management.passes import reverse_parallelize


class TestReverseParallelizeBasic:
    def test_returns_quantum_circuit(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cz(0, 1)
        result = reverse_parallelize(qc)
        assert isinstance(result, QuantumCircuit)

    def test_empty_circuit(self):
        qc = QuantumCircuit(3)
        result = reverse_parallelize(qc)
        assert result.num_qubits() == 3
        assert len(result.data) == 0

    def test_qubit_count_preserved(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(1, 2)
        result = reverse_parallelize(qc)
        assert result.num_qubits() == qc.num_qubits()

    def test_gate_multiset_preserved(self):
        """reverse_parallelize must not add or remove gates."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cz(0, 1)
        qc.swap(1, 2)
        qc.cz(0, 1)
        qc.x(2)

        before_names = sorted(instr.op.name for instr in qc.data)
        result = reverse_parallelize(qc)
        after_names = sorted(instr.op.name for instr in result.data)

        assert before_names == after_names

    def test_single_qubit_circuit_unchanged_multiset(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.x(0)
        qc.z(0)
        before_names = sorted(instr.op.name for instr in qc.data)
        result = reverse_parallelize(qc)
        after_names = sorted(instr.op.name for instr in result.data)
        assert before_names == after_names


class TestReverseParallelizeParallelism:
    def test_does_not_increase_gate_count(self):
        """Parallelization never adds gates."""
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.h(i)
        for i in range(3):
            qc.cx(i, i + 1)
        result = reverse_parallelize(qc)
        assert len(result.data) <= len(qc.data)

    def test_idempotent_on_already_parallel_circuit(self):
        """Applying twice should give the same gate multiset as once."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        once = reverse_parallelize(qc)
        twice = reverse_parallelize(once)
        assert sorted(i.op.name for i in once.data) == sorted(
            i.op.name for i in twice.data
        )


# ---------------------------------------------------------------------------
# Correctness: compare_unitary
# ---------------------------------------------------------------------------


class TestReverseParallelizeCorrectness:
    """Verify reverse_parallelize preserves the circuit unitary."""

    def test_unitary_preserved_simple(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cz(0, 1)
        qc.swap(1, 2)
        qc.cz(0, 1)
        qc.x(2)
        assert reverse_parallelize.compare_unitary(qc)

    def test_unitary_preserved_layer_circuit(self):
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.h(i)
        for i in range(3):
            qc.cx(i, i + 1)
        assert reverse_parallelize.compare_unitary(qc)

    def test_unitary_preserved_single_qubit(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.x(0)
        qc.z(0)
        assert reverse_parallelize.compare_unitary(qc)
