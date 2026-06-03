from math import pi
from unittest.mock import MagicMock

import numpy as np
import pytest
from cirq import final_state_vector, num_qubits, unitary

from qrisp import (
    QPE,
    QuantumBool,
    QuantumFloat,
    QuantumVariable,
    auto_uncompute,
    cx,
    h,
    mcx,
    p,
    reflection,
    x,
    z,
)
from qrisp.circuit import ClControlledOperation, QuantumCircuit
from qrisp.grover import diffuser
from qrisp.interface.converter.cirq_converter import convert_to_cirq, convert_from_cirq


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
    first_non_zero_val = U[0, 0]
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
    calculated_unitary = unitary(reflection_cirq_qc)
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


# ------------------------------------------------------------------
# Tests for convert_from_cirq
# ------------------------------------------------------------------


def test_convert_from_cirq_single_qubit_gates():
    """Round-trip test: Qrisp -> Cirq -> Qrisp should preserve unitary for single-qubit gates."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.x(1)
    qc.y(2)
    qc.z(3)
    qc.s(0)
    qc.t(1)
    qc.s_dg(2)
    qc.t_dg(3)
    qc.sx(0)
    qc.sx_dg(1)
    qc.id(2)

    expected_unitary = qc.get_unitary()

    cirq_circ = convert_to_cirq(qc)
    qrisp_qc = convert_from_cirq(cirq_circ)

    np.testing.assert_array_almost_equal(
        qrisp_qc.get_unitary(), expected_unitary
    )


def test_convert_from_cirq_parametrized_gates():
    """Round-trip test for parametrized single-qubit gates."""
    qc = QuantumCircuit(4)
    qc.rx(0.3, 0)
    qc.ry(0.4, 1)
    qc.rz(0.2, 2)
    qc.p(0.5, 3)

    expected_unitary = qc.get_unitary()

    cirq_circ = convert_to_cirq(qc)
    qrisp_qc = convert_from_cirq(cirq_circ)
    np.testing.assert_array_almost_equal(
        qrisp_qc.get_unitary(), expected_unitary
    )


def test_convert_from_cirq_two_qubit_gates():
    """Round-trip test for two-qubit gates."""
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cz(2, 3)
    qc.swap(0, 2)

    expected_unitary = qc.get_unitary()

    cirq_circ = convert_to_cirq(qc)
    qrisp_qc = convert_from_cirq(cirq_circ)
    np.testing.assert_array_almost_equal(
        qrisp_qc.get_unitary(), expected_unitary
    )


def test_convert_from_cirq_gphase():
    """Round-trip test for global phase."""
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.x(0)
    qc.gphase(0.5, 0)

    expected_unitary = qc.get_unitary()

    cirq_circ = convert_to_cirq(qc)
    qrisp_qc = convert_from_cirq(cirq_circ)

    qrisp_unitary = qrisp_qc.get_unitary()
    np.testing.assert_array_almost_equal(qrisp_unitary, expected_unitary)


def test_convert_from_cirq_fully_parametrized():
    """Round-trip test combining many gate types."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.25, 1)
    qc.cx(1, 2)
    qc.ry(0.5, 2)
    qc.cz(0, 3)
    qc.p(0.75, 3)
    qc.s(0)
    qc.t(1)

    expected_unitary = qc.get_unitary()

    cirq_circ = convert_to_cirq(qc)
    qrisp_qc = convert_from_cirq(cirq_circ)
    np.testing.assert_array_almost_equal(
        qrisp_qc.get_unitary(), expected_unitary
    )


def test_convert_from_cirq_direct_cirq_circuit():
    """Test converting a Cirq circuit built directly (not via round-trip)."""
    import cirq

    q0, q1 = cirq.LineQubit.range(2)

    cirq_circ = cirq.Circuit(
        [cirq.H(q0), cirq.CNOT(q0, q1), cirq.rz(0.5)(q1), cirq.measure(q0, q1)]
    )

    qrisp_qc = convert_from_cirq(cirq_circ)
    assert qrisp_qc.num_qubits() == 2
    assert len(qrisp_qc.data) == 5
    assert qrisp_qc.data[0].op.name == "h"
    assert qrisp_qc.data[1].op.name == "cx"
    assert qrisp_qc.data[2].op.name == "rz"
    assert qrisp_qc.data[3].op.name == "measure"
    assert qrisp_qc.data[4].op.name == "measure"


def test_convert_from_cirq_via_classmethod():
    """Test the QuantumCircuit.from_cirq classmethod."""
    import cirq

    q0, q1 = cirq.LineQubit.range(2)
    cirq_circ = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])

    qrisp_qc = QuantumCircuit.from_cirq(cirq_circ)
    assert qrisp_qc.num_qubits() == 2

    cirq_unitary = cirq.unitary(cirq_circ)
    qrisp_unitary = qrisp_qc.get_unitary()

    np.testing.assert_array_almost_equal(
        np.abs(cirq_unitary), np.abs(qrisp_unitary)
    )


def test_convert_from_cirq_swap():
    """Test round-trip for swap gate."""
    qc = QuantumCircuit(2)
    qc.swap(0, 1)
    qc.h(0)

    expected_unitary = qc.get_unitary()

    cirq_circ = convert_to_cirq(qc)
    qrisp_qc = convert_from_cirq(cirq_circ)
    np.testing.assert_array_almost_equal(
        qrisp_qc.get_unitary(), expected_unitary
    )


def test_convert_from_cirq_reset():
    """Test round-trip for reset."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.reset(0)
    qc.cx(0, 1)

    cirq_circ = convert_to_cirq(qc)
    qrisp_qc = convert_from_cirq(cirq_circ)
    assert qrisp_qc.num_qubits() == 2
    assert qrisp_qc.data[0].op.name == "h"
    assert qrisp_qc.data[1].op.name == "reset"
    assert qrisp_qc.data[2].op.name == "cx"


def test_convert_from_cirq_unsupported_gate():
    """Test that an error is raised for unsupported Cirq gates."""
    import cirq

    q0 = cirq.LineQubit(0)
    cirq_circ = cirq.Circuit([cirq.ISWAP(q0, cirq.LineQubit(1))])

    with pytest.raises(ValueError, match="not supported by the Cirq to Qrisp converter"):
        convert_from_cirq(cirq_circ)
