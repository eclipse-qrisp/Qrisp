from math import pi
from unittest.mock import MagicMock

import cirq
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
from qrisp.circuit import QuantumCircuit
from qrisp.grover import diffuser
from qrisp.interface.converter.cirq_converter import convert_to_cirq, convert_from_cirq


def _build_single_qubit_circ():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.x(1)
    qc.y(3)
    qc.z(2)
    qc.rx(0.3, 3)
    qc.ry(0.4, 1)
    qc.rz(0.2, 2)
    qc.s(0)
    qc.t(1)
    qc.t_dg(3)
    qc.s_dg(2)
    return qc


def _build_mcx_circ():
    qc = QuantumCircuit(8)
    qc.mcx([0, 1, 2, 4, 5, 6], [3, 7])
    return qc


@pytest.mark.parametrize(
    "circ_builder",
    [_build_single_qubit_circ, _build_mcx_circ],
    ids=["single_qubit_gates", "mcx"],
)
def test_convert_to_cirq_preserves_unitary(circ_builder):
    qc = circ_builder()
    expected_unitary = qc.get_unitary()
    converted_cirq = convert_to_cirq(qc)
    np.testing.assert_array_almost_equal(expected_unitary, unitary(converted_cirq))


def _build_mcx_qs():
    ctrl = QuantumVariable(4)
    target = QuantumBool()
    mcx(ctrl, target)
    return ctrl.qs.compile()


def _build_reflection_qs():
    qv = QuantumVariable(5)
    x(qv)

    def ghz(qv):
        h(qv[0])

    for i in range(1, qv.size):
        cx(qv[0], qv[i])
    reflection(qv, ghz)
    return qv.qs.compile()


@pytest.mark.parametrize(
    "circ_builder",
    [_build_mcx_qs, _build_reflection_qs],
    ids=["mcx_in_qs", "reflection"],
)
def test_convert_to_cirq_compiled_qs(circ_builder):
    compiled_qc = circ_builder()
    expected_unitary = compiled_qc.get_unitary()
    converted_cirq = convert_to_cirq(compiled_qc)
    np.testing.assert_array_almost_equal(expected_unitary, unitary(converted_cirq))


def test_gphase():
    qc = QuantumCircuit(1)

    qc.x(0)
    qc.x(0)
    qc.gphase(0.5, 0)
    circuit = qc.to_cirq()
    U = unitary(circuit)
    first_non_zero_val = U[0, 0]
    phase_angle = np.angle(first_non_zero_val)

    assert phase_angle == 0.5


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

    expected_unitary = test_circuit.get_unitary()
    converted_cirq = convert_to_cirq(test_circuit)
    np.testing.assert_array_almost_equal(expected_unitary, unitary(converted_cirq))


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
        np.zeros(zero_array_shape),
        np.round(qrisp_sv - cirq_sv),
    )


def test_recursive_conversion():
    """Verify an op that is non-elementary and has the definition attribute can be converted recursively."""

    qc = QuantumCircuit(4)
    qc.rxx(0.3, 0, 1)
    qc.rxx(0.3, 2, 3)

    cirq_circuit = qc.to_cirq()
    assert qc.num_qubits() == num_qubits(cirq_circuit)


def test_convert_to_cirq_cirq_qubits_passthrough():
    """Providing custom cirq_qubits should be preserved after transpilation."""
    from qrisp import QuantumVariable, h, QPE, p

    def U(qv):
        p(0.5 * 2 * np.pi, qv[0])
        p(0.125 * 2 * np.pi, qv[1])

    qv = QuantumVariable(2)
    h(qv)
    QPE(qv, U, precision=3)
    qrisp_circ = qv.qs.compile()

    custom_qubits = [cirq.LineQubit(i * 10) for i in range(qrisp_circ.num_qubits())]
    cirq_circ = convert_to_cirq(qrisp_circ, cirq_qubits=custom_qubits)

    expected_unitary = qrisp_circ.get_unitary()
    np.testing.assert_array_almost_equal(expected_unitary, cirq.unitary(cirq_circ))


def _mock_unknown_circ():
    qc = QuantumCircuit(1)
    qc.data = [MagicMock(op=MagicMock(name="some_gate", params=[]), qubits=[MagicMock()])]
    return qc


def _mock_none_gate_circ():
    qc = QuantumCircuit(2)
    mock_op = MagicMock()
    mock_op.name = "rxx"
    mock_op.params = []
    mock_op.definition = None
    qc.data = [MagicMock(op=mock_op, qubits=list(qc.qubits))]
    return qc


@pytest.mark.parametrize(
    "circ_builder, match",
    [
        (_mock_unknown_circ, "could not be transpiled and are not supported"),
        (_mock_none_gate_circ, "has no Cirq equivalent"),
    ],
    ids=["unknown_gate", "none_gate_no_definition"],
)
def test_convert_to_cirq_raises(circ_builder, match):
    with pytest.raises(ValueError, match=match):
        convert_to_cirq(circ_builder())


def _build_single_qubit_roundtrip():
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
    return qc


def _build_parametrized_roundtrip():
    qc = QuantumCircuit(4)
    qc.rx(0.3, 0)
    qc.ry(0.4, 1)
    qc.rz(0.2, 2)
    qc.p(0.5, 3)
    return qc


def _build_two_qubit_roundtrip():
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cz(2, 3)
    qc.swap(0, 2)
    return qc


def _build_gphase_roundtrip():
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.x(0)
    qc.gphase(0.5, 0)
    return qc


def _build_swap_roundtrip():
    qc = QuantumCircuit(2)
    qc.swap(0, 1)
    qc.h(0)
    return qc


def _build_mixed_roundtrip():
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
    return qc


@pytest.mark.parametrize(
    "circ_builder",
    [
        _build_single_qubit_roundtrip,
        _build_parametrized_roundtrip,
        _build_two_qubit_roundtrip,
        _build_gphase_roundtrip,
        _build_swap_roundtrip,
        _build_mixed_roundtrip,
    ],
    ids=[
        "single_qubit",
        "parametrized",
        "two_qubit",
        "gphase",
        "swap",
        "mixed",
    ],
)
def test_roundtrip_preserves_unitary(circ_builder):
    qc = circ_builder()
    expected_unitary = qc.get_unitary()

    cirq_circ = convert_to_cirq(qc)
    qrisp_qc = convert_from_cirq(cirq_circ)

    np.testing.assert_array_almost_equal(qrisp_qc.get_unitary(), expected_unitary)



def test_convert_from_cirq_direct_cirq_circuit():
    """Test converting a Cirq circuit built directly (not via round-trip)."""
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
    q0, q1 = cirq.LineQubit.range(2)
    cirq_circ = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])

    qrisp_qc = QuantumCircuit.from_cirq(cirq_circ)
    assert qrisp_qc.num_qubits() == 2

    cirq_unitary = cirq.unitary(cirq_circ)
    qrisp_unitary = qrisp_qc.get_unitary()

    np.testing.assert_array_almost_equal(
        np.abs(cirq_unitary), np.abs(qrisp_unitary)
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
    q0 = cirq.LineQubit(0)
    cirq_circ = cirq.Circuit([cirq.ISWAP(q0, cirq.LineQubit(1))])

    with pytest.raises(ValueError, match="not supported by the Cirq to Qrisp converter"):
        convert_from_cirq(cirq_circ)


def test_convert_from_cirq_empty_circuit():
    """Test converting an empty Cirq circuit."""
    cirq_circ = cirq.Circuit()
    qrisp_qc = convert_from_cirq(cirq_circ)
    assert qrisp_qc.num_qubits() == 0
    assert len(qrisp_qc.data) == 0


@pytest.mark.parametrize(
    "gate_key, expected_name",
    [
        ("X", "x"),
        ("Y", "y"),
        ("Z", "z"),
        ("H", "h"),
        ("S", "s"),
        ("T", "t"),
        ("S_dg", "s_dg"),
        ("T_dg", "t_dg"),
        ("sx", "sx"),
        ("sx_dg", "sx_dg"),
        ("I", "id"),
    ],
)
def test_convert_from_cirq_single_gate(gate_key, expected_name):
    """Test direct conversion of individual single-qubit gates."""
    q = cirq.LineQubit(0)

    gate_map = {
        "X": cirq.X(q),
        "Y": cirq.Y(q),
        "Z": cirq.Z(q),
        "H": cirq.H(q),
        "S": cirq.S(q),
        "T": cirq.T(q),
        "S_dg": cirq.S(q) ** -1,
        "T_dg": cirq.T(q) ** -1,
        "sx": cirq.X(q) ** 0.5,
        "sx_dg": cirq.X(q) ** -0.5,
        "I": cirq.I(q),
    }

    cirq_circ = cirq.Circuit([gate_map[gate_key]])

    qrisp_qc = convert_from_cirq(cirq_circ)
    assert qrisp_qc.data[0].op.name == expected_name

    expected_unitary = cirq.unitary(cirq_circ)
    np.testing.assert_array_almost_equal(
        np.abs(qrisp_qc.get_unitary()), np.abs(expected_unitary)
    )


@pytest.mark.parametrize(
    "gate_fn, exponent, should_raise",
    [
        (cirq.H, 0.5, True),
        (cirq.CNOT, 0.5, True),
        (cirq.CZ, 0.5, True),
        (cirq.SWAP, 0.5, True),
        (cirq.H, -1, False),
        (cirq.CNOT, -1, False),
        (cirq.CZ, -1, False),
        (cirq.SWAP, -1, False),
    ],
    ids=[
        "H^0.5_raises", "CX^0.5_raises", "CZ^0.5_raises", "SWAP^0.5_raises",
        "H^-1_allowed", "CX^-1_allowed", "CZ^-1_allowed", "SWAP^-1_allowed",
    ],
)
def test_convert_from_cirq_exponent_guards(gate_fn, exponent, should_raise):
    """Non-unit exponents are rejected for H/CX/CZ/SWAP; self-inverse ones work."""
    q = cirq.LineQubit.range(3)
    g = gate_fn**exponent
    if g.num_qubits() == 1:
        cirq_circ = cirq.Circuit([g(q[0])])
    else:
        cirq_circ = cirq.Circuit([g(q[0], q[1])])

    if should_raise:
        with pytest.raises(ValueError, match="exponent"):
            convert_from_cirq(cirq_circ)
    else:
        qrisp_qc = convert_from_cirq(cirq_circ)
        expected_unitary = cirq.unitary(cirq_circ)
        np.testing.assert_array_almost_equal(
            qrisp_qc.get_unitary(), expected_unitary
        )


@pytest.mark.parametrize(
    "key",
    [
        "X.controlled()",
        "ControlledOperation",
        "Toffoli_via_ControlledGate",
        "anti_control_via_control_values",
    ],
)
def test_convert_from_cirq_controlled_gates(key):
    """Controlled gates via ControlledGate and ControlledOperation APIs."""
    q0, q1, q2 = cirq.LineQubit.range(3)

    builders = {
        "X.controlled()":
            cirq.Circuit([cirq.X.controlled()(q0, q1)]),
        "ControlledOperation":
            cirq.Circuit([cirq.ControlledOperation([q0], cirq.X(q1))]),
        "Toffoli_via_ControlledGate":
            cirq.Circuit([cirq.ControlledGate(cirq.X, num_controls=2)(q0, q1, q2)]),
        "anti_control_via_control_values":
            cirq.Circuit([cirq.X.controlled(control_values=[0, 1])(q0, q1, q2)]),
    }

    cirq_circ = builders[key]
    qrisp_qc = convert_from_cirq(cirq_circ)

    expected_unitary = cirq.unitary(cirq_circ)
    np.testing.assert_array_almost_equal(
        qrisp_qc.get_unitary(), expected_unitary
    )


def _build_multi_valued_circ():
    q0, q1, q2 = cirq.LineQubit.range(3)
    g = cirq.ControlledGate(cirq.X, control_values=[(0, 1), 1])
    return cirq.Circuit([g(q0, q1, q2)])


def _build_global_phase_only_circ():
    return cirq.Circuit([cirq.GlobalPhaseGate(1j).on()])


def _build_controlled_no_gate_circ():
    q0, q1 = cirq.LineQubit.range(2)
    inner = cirq.FrozenCircuit([cirq.X(q0)])
    co = cirq.CircuitOperation(inner)
    return cirq.Circuit([cirq.ControlledOperation([q1], co)])


@pytest.mark.parametrize(
    "circ_builder, match",
    [
        (_build_multi_valued_circ, "Multi-valued control"),
        (_build_controlled_no_gate_circ, "not supported"),
    ],
    ids=["multi_valued_control", "controlled_sub_op_no_gate"],
)
def test_convert_from_cirq_raises(circ_builder, match):
    with pytest.raises(ValueError, match=match):
        convert_from_cirq(circ_builder())


def test_convert_from_cirq_global_phase_only():
    """A GlobalPhaseGate-only circuit (zero qubits) should convert without error."""
    circ = cirq.Circuit([cirq.GlobalPhaseGate(1j).on()])
    qrisp_qc = convert_from_cirq(circ)
    assert qrisp_qc.num_qubits() == 0
    assert len(qrisp_qc.data) == 0