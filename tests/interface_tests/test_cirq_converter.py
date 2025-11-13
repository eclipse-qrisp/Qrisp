import pytest

from cirq import Circuit, LineQubit

from cirq import CNOT, H, X, Y, Z, CZ, S, T, R, SWAP, rx, ry, rz, M, R
from unittest.mock import MagicMock, create_autospec
from qrisp.circuit import QuantumCircuit, ControlledOperation, ClControlledOperation

from qrisp.interface.converter.cirq_converter import convert_to_cirq
from qrisp import QuantumCircuit

# define circuits to be used by the unit tests

# 4 qubit circuit containing all single qubit gates
qc_single_qubit_gates = QuantumCircuit(4)
qc_single_qubit_gates.h(0)
qc_single_qubit_gates.x(1)
qc_single_qubit_gates.y(3)
qc_single_qubit_gates.z(2)
qc_single_qubit_gates.rx(0.3, 3)
qc_single_qubit_gates.ry(0.4, 1)
qc_single_qubit_gates.rz(0.2, 2)
qc_single_qubit_gates.s(0)
qc_single_qubit_gates.t(1)
qc_single_qubit_gates.t_dg(3)
qc_single_qubit_gates.s_dg(2)
qc_single_qubit_gates.measure(0)
qc_single_qubit_gates.reset(0)

expected_cirq_qc_single_qubit_gates_ops = [
        H(LineQubit(0)),
        X(LineQubit(1)),
        Y(LineQubit(3)),
        Z(LineQubit(2)),
        rx(rads=0.3).on(LineQubit(3)),
        ry(rads=0.4).on(LineQubit(1)),
        rz(rads=0.2).on(LineQubit(2)),
        S(LineQubit(0)),
        T(LineQubit(1)),
        (T**-1).on(LineQubit(3)),
        (S**-1).on(LineQubit(2)),
        M(LineQubit(0)),
        R(LineQubit(0))
        ]

# 4 qubit circuit containing all two qubit gates

# 4 qubit circuit containing all multi-controlled gates

@pytest.mark.parametrize("input_circuit, expected_res", [
    (qc_single_qubit_gates, expected_cirq_qc_single_qubit_gates_ops)])
def test_cirq_converter(input_circuit, expected_res):
    """Check the Qrisp to Cirq converter works for different circuits."""
    converted_circ = convert_to_cirq(input_circuit)
    assert expected_res == list(converted_circ.all_operations())


    

def test_two_qubit_circuit():
    """Check a Qrisp circuit containing all two qubit gates is properly converted to a Cirq circuit."""


def test_mc_qubit_circuit():
    """Check a Qrisp circuit containing all multi-controlled gates is properly converted to a Cirq circuit."""


def test_labeled_qubits():
    """Verify converter works for named qubits in the Qrisp circuit."""


def test_symbolic_parametrized_gates():
    """Verify error is raised when a symbolic parameter is provided to certain parametrized gates."""


@pytest.mark.parametrize("op, expected_msg", [
    (MagicMock(name='some_gate'), "some_gate gate is not supported by the Qrisp to Cirq converter."),
    (MagicMock(spec=ClControlledOperation), "Classically controlled gate is not supported by the Qrisp to Cirq converter.")
    ])
def test_unsupported_gate(op, expected_msg):
    """Mock unit test to verify an error is raised for an unsupported gate."""
    if isinstance(op, ClControlledOperation):
        op.configure_mock(name='Classically controlled')
    else:
        op.configure_mock(name='some_gate')
    
    qc = QuantumCircuit(1)
    qc.data = [MagicMock(op=op, qubits=[MagicMock()])]
    with pytest.raises(ValueError, match=expected_msg):
        convert_to_cirq(qc)
