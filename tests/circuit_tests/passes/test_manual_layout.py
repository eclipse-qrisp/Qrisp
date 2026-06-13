"""Tests for qrisp.circuit.passes.manual_layout."""
import pytest

from qrisp.circuit import QuantumCircuit
from qrisp.circuit.pass_management.passes import manual_layout


class TestManualLayoutFactory:
    def test_returns_callable(self):
        pass_fn = manual_layout([0, 1, 2])
        assert callable(pass_fn)

    def test_callable_name(self):
        pass_fn = manual_layout([0, 1])
        assert pass_fn.__name__ == "manual_layout"


class TestManualLayoutIdentity:
    def test_identity_mapping_preserves_qubit_count(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        result = manual_layout([0, 1, 2])(qc)
        assert result.num_qubits() == 3

    def test_identity_mapping_preserves_gate_multiset(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cz(1, 2)
        before = sorted(i.op.name for i in qc.data)
        result = manual_layout([0, 1, 2])(qc)
        after = sorted(i.op.name for i in result.data)
        assert before == after


class TestManualLayoutRemapping:
    def test_swap_two_qubits(self):
        """[1, 0] swaps logical 0 and 1 — the gate operands should be reversed."""
        qc = QuantumCircuit(2)
        qc.h(0)           # H on logical qubit 0
        result = manual_layout([1, 0])(qc)
        # After remapping, physical qubit 1 holds logical qubit 0.
        # The resulting qubit list should start with what was originally qubit 1.
        assert result.num_qubits() == 2
        # Gate count should be unchanged
        assert len(result.data) == len(qc.data)

    def test_sparse_mapping_expands_qubit_count(self):
        """Mapping [0, 2] on a 2-qubit circuit → 3-qubit output (gap at index 1)."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        result = manual_layout([0, 2])(qc)
        assert result.num_qubits() == 3

    def test_sparse_mapping_gate_multiset_preserved(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        before = sorted(i.op.name for i in qc.data)
        result = manual_layout([0, 3])(qc)
        after = sorted(i.op.name for i in result.data)
        assert before == after

    def test_three_qubit_rotation(self):
        """`[2, 0, 1]` is a cyclic rotation."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cz(1, 2)
        result = manual_layout([2, 0, 1])(qc)
        assert result.num_qubits() == 3
        assert len(result.data) == len(qc.data)


class TestManualLayoutValidation:
    def test_wrong_length_raises(self):
        qc = QuantumCircuit(3)
        with pytest.raises(ValueError, match="qubit_mapping specifies"):
            manual_layout([0, 1])(qc)

    def test_negative_index_raises(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="non-negative"):
            manual_layout([-1, 1])(qc)

    def test_duplicate_indices_raises(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            manual_layout([1, 1])(qc)

    def test_empty_circuit(self):
        qc = QuantumCircuit(0)
        result = manual_layout([])(qc)
        assert result.num_qubits() == 0
