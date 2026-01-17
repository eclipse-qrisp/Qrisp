import pytest
import numpy as np

from cirq import unitary

from unittest.mock import MagicMock
from qrisp.circuit import QuantumCircuit, ClControlledOperation
from qrisp.grover import diffuser
from qrisp.interface.converter.cirq_converter import convert_to_cirq
from qrisp import (
    QuantumVariable,
    mcx,
    cx,
    QuantumBool,
    h,
    x,
    reflection,
    p,
    QPE,
    auto_uncompute,
    z,
    QuantumFloat,
)
from cirq import final_state_vector, num_qubits


from math import pi


def test_n_qubit_gate_circuit():
    """Verifies the circuit works as expected fordifferent circuits."""

    # 4 qubit circuit containing some single qubit gates
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
    expected_unitary = qc_single_qubit_gates.get_unitary()
    converted_cirq = convert_to_cirq(qc_single_qubit_gates)
    calculated_unitary = unitary(converted_cirq)
    np.testing.assert_array_almost_equal(expected_unitary, calculated_unitary)

    # 4 qubit circuit containing all multi-controlled gates
    # there is only 1 multicontrolled gate mcx
    qc_mcx = QuantumCircuit(8)
    qc_mcx.mcx([0, 1, 2, 4, 5, 6], [3, 7])
    expected_unitary = qc_mcx.get_unitary()
    converted_cirq = convert_to_cirq(qc_mcx)
    calculated_unitary = unitary(converted_cirq)
    np.testing.assert_array_almost_equal(expected_unitary, calculated_unitary)


@pytest.mark.parametrize(
    "op, expected_msg",
    [
        (
            MagicMock(name="some_gate"),
            "some_gate gate is not supported by the Qrisp to Cirq converter.",
        ),
        (
            MagicMock(spec=ClControlledOperation),
            "Classically controlled gate is not supported by the Qrisp to Cirq converter.",
        ),
    ],
)
def test_unsupported_gate(op, expected_msg):
    """Mock unit test to verify an error is raised for an unsupported gate."""
    if isinstance(op, ClControlledOperation):
        op.configure_mock(name="Classically controlled")
    else:
        op.configure_mock(name="some_gate")

    qc = QuantumCircuit(1)
    qc.data = [MagicMock(op=op, qubits=[MagicMock()])]
    with pytest.raises(ValueError, match=expected_msg):
        convert_to_cirq(qc)


def test_gphase():
    
    qc = QuantumCircuit(1)

    qc.x(0)
    qc.x(0)
    qc.gphase(0.5, 0)
    circuit = qc.to_cirq()
    U = unitary(circuit)
    first_non_zero_val = U[0,0]
    # Calculate the phase angle
    phase_angle = np.angle(first_non_zero_val)
    
    assert phase_angle == 0.5

def test_converter_compiled_qs():
    """Verify the converter works as expected on a simple compiled QuantumSession circuit."""

    # multi controlled x in a QuantumSession
    ctrl = QuantumVariable(4)
    target = QuantumBool()
    mcx(ctrl, target)
    compiled_qc = ctrl.qs.compile()
    expected_unitary = compiled_qc.get_unitary()
    converted_cirq = convert_to_cirq(compiled_qc)
    calculated_unitary = unitary(converted_cirq)
    np.testing.assert_array_almost_equal(expected_unitary, calculated_unitary)

    # reflection around GHZ state in a QuantumSession
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

    expected_unitary = reflection_compiled_qc.get_unitary()
    # multiple by -1 to account for the global phase gate in the qrisp circuit
    calculated_unitary = -1 * unitary(reflection_cirq_qc)
    np.testing.assert_array_almost_equal(expected_unitary, calculated_unitary)


def test_transpiled_qc():
    """Verify the converter works without any issues for a transpiled circuit that has a composite gate."""
    from qrisp import QuantumVariable

    def U(qv):
        x = 0.5
        y = 0.125

        p(x * 2 * np.pi, qv[0])
        p(y * 2 * np.pi, qv[1])

    qv = QuantumVariable(2)

    h(qv)

    QPE(qv, U, precision=3)
    test_circuit = qv.qs.compile()
    # test_circuit has a composite gate QFT_dg
    # the function transpiles the composite gate, if possible

    expected_unitary = test_circuit.get_unitary()
    converted_cirq = convert_to_cirq(test_circuit)
    calculated_unitary = unitary(converted_cirq)
    np.testing.assert_array_almost_equal(expected_unitary, calculated_unitary)


def test_grover_example():
    """Verify the function works as expected for the Grover example from
    the 101 tutorial."""

    @auto_uncompute
    def sqrt_oracle(qf):
        temp_qbool = qf * qf == 0.25
        z(temp_qbool)

    qf = QuantumFloat(3, -1, signed=True)
    sqrt_oracle(qf)

    qf = QuantumFloat(3, -1, signed=True)

    n = qf.size
    iterations = int(0.25 * pi * (2**n / 2) ** 0.5)

    h(qf)

    for i in range(iterations):
        sqrt_oracle(qf)
        diffuser(qf)

    qrisp_circuit = qf.qs.compile()
    cirq_circuit = convert_to_cirq(qrisp_circuit)
    qrisp_sv = qrisp_circuit.statevector_array()
    cirq_sv = final_state_vector(cirq_circuit)
    zero_array_shape = np.shape(cirq_sv)
    np.testing.assert_array_almost_equal(
        np.zeros(
            zero_array_shape,
        ),
        np.round(qrisp_sv - cirq_sv),
    )

def test_recursive_conversion():
    """Verify an op that is non-elementary and has the definition attribute can be converted recursively."""

    qc = QuantumCircuit(4)
    qc.rxx(0.3, 0, 1)
    qc.rxx(0.3, 2, 3)

    cirq_circuit = qc.to_cirq()
    assert qc.num_qubits() == num_qubits(cirq_circuit)
