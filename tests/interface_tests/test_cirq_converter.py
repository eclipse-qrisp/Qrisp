import pytest

from cirq import Circuit, LineQubit

from cirq import CNOT, H, X, Y, Z, CZ, S, T, R, SWAP, rx, ry, rz, M, R
from unittest.mock import MagicMock, create_autospec
from qrisp.circuit import QuantumCircuit, ControlledOperation, ClControlledOperation

from qrisp.interface.converter.cirq_converter import convert_to_cirq
from qrisp import QuantumVariable, mcx, cx, QuantumBool
from qrisp import QuantumVariable, QuantumArray, h, x, cx, reflection
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
# there is only 1 multicontrolled gate mcx
qc_mcx = QuantumCircuit(10)
qc_mcx.mcx([0, 1, 2, 4, 5, 7], [3, 8])

expected_mcx_circ = [
    X(LineQubit(3)).controlled_by(LineQubit(0), LineQubit(1), LineQubit(2), LineQubit(4), LineQubit(5), LineQubit(7)),
    X(LineQubit(8)).controlled_by(LineQubit(0), LineQubit(1), LineQubit(2), LineQubit(4), LineQubit(5), LineQubit(7))]

@pytest.mark.parametrize("input_circuit, expected_res", [
    (qc_single_qubit_gates, expected_cirq_qc_single_qubit_gates_ops),
    (qc_mcx, expected_mcx_circ)])
def test_cirq_converter(input_circuit, expected_res):
    """Check the Qrisp to Cirq converter works for different circuits."""
    converted_circ = convert_to_cirq(input_circuit)
    assert expected_res == list(converted_circ.all_operations())


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
    

@pytest.mark.parametrize("gate, qubits", [
    ("rxx", [1, 3]),
    ("rzz", [2, 0]),
    ("gphase", [0]),
])
def test_gphase_error(capsys, gate, qubits):
    """Verify a message is printed when the Qrisp circuit contains a global phase gate."""
    qc = QuantumCircuit(4)
    if gate == "rxx":
        qc.rxx(0.3, *qubits)
    elif gate == "rzz":
        qc.rzz(0.3, *qubits)
    elif gate == "gphase":
        qc.gphase(0.3, *qubits)

    convert_to_cirq(qc)
    captured = capsys.readouterr()
    assert "Qrisp circuit contains a global phase gate which will be skipped in the Qrisp to Cirq conversion." in captured.out


def test_converter_compiled_qs():
    """Verify the converter works as expected on a simple compiled QuantumSession circuit."""

    # multi controlled x in a QuantumSession
    ctrl = QuantumVariable(4)
    target = QuantumBool()
    mcx(ctrl, target)
    compiled_qc = ctrl.qs.compile()
    cirq_qc = convert_to_cirq(compiled_qc)
    assert [X(LineQubit(4)).controlled_by(LineQubit(0), LineQubit(1), LineQubit(2), LineQubit(3))] == list(cirq_qc.all_operations())

    # reflection around GHZ state
    # Prepare |1> state
    qv = QuantumVariable(5)
    x(qv)

    def ghz(qv):
        h(qv[0])

    for i in range(1, qv.size):
        cx(qv[0], qv[i])
    reflection(qv, ghz)

    reflection_compiled_qc = qv.qs.compile()
    reflection_cirq_qc = convert_to_cirq(reflection_compiled_qc)
    assert list(reflection_cirq_qc.all_operations()) == [
        X(LineQubit(1)),
        X(LineQubit(2)),
        X(LineQubit(3)),
        X(LineQubit(0)),
        CNOT(LineQubit(0), LineQubit(4)),
        H(LineQubit(4)),
        CNOT(LineQubit(0), LineQubit(1)),
        CNOT(LineQubit(0), LineQubit(2)),
        CNOT(LineQubit(0), LineQubit(3)),
        H(LineQubit(0)),
        X(LineQubit(4)).controlled_by(LineQubit(0), LineQubit(1), LineQubit(2), LineQubit(3)),
        H(LineQubit(4)),
        H(LineQubit(0)),
        X(LineQubit(4))]