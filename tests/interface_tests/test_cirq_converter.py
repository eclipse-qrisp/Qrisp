from math import pi
from unittest.mock import MagicMock, patch

import cirq
import numpy as np
import pytest
from cirq import final_state_vector, num_qubits, unitary
from qrisp.grover import diffuser

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
from qrisp.circuit import ControlledOperation, QuantumCircuit
from qrisp.interface.converter.cirq_converter import (
    convert_from_cirq,
    convert_to_cirq,
    _decompose_unknown_ops,
)


def _build_single_qubit_circ():
    """Qrisp circuit with all single-qubit gates (H, X, Y, Z, RX, RY, RZ, S, T, S†, T†) and cy."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.x(1)
    qc.y(3)
    qc.z(2)
    qc.cy(0, 2)
    qc.rx(0.3, 3)
    qc.ry(0.4, 1)
    qc.rz(0.2, 2)
    qc.s(0)
    qc.t(1)
    qc.t_dg(3)
    qc.s_dg(2)
    return qc


def _build_mcx_circ():
    """Qrisp circuit with multi-controlled X (Toffoli-like) gates on 8 qubits."""
    qc = QuantumCircuit(8)
    qc.mcx([0, 1, 2, 4, 5, 6], [3, 7])
    return qc


@pytest.mark.parametrize(
    "circ_builder",
    [_build_single_qubit_circ, _build_mcx_circ],
    ids=["single_qubit_gates", "mcx"],
)
def test_convert_to_cirq_preserves_unitary(circ_builder):
    """Verify unitary is preserved when converting Qrisp circuits to Cirq."""
    qc = circ_builder()
    expected_unitary = qc.get_unitary()
    converted_cirq = convert_to_cirq(qc)
    np.testing.assert_array_almost_equal(expected_unitary, unitary(converted_cirq))


def _build_mcx_qs():
    """Compiled quantum session with a 4-controlled multi-controlled X gate."""
    ctrl = QuantumVariable(4)
    target = QuantumBool()
    mcx(ctrl, target)
    return ctrl.qs.compile()


def _build_reflection_qs():
    """Compiled quantum session with a reflection operator over GHZ state."""
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
    """Verify unitary is preserved for compiled quantum sessions."""
    compiled_qc = circ_builder()
    expected_unitary = compiled_qc.get_unitary()
    converted_cirq = convert_to_cirq(compiled_qc)
    np.testing.assert_array_almost_equal(expected_unitary, unitary(converted_cirq))


def test_gphase():
    """Verify global phase gate is correctly converted to Cirq."""
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


def test_grover():
    """Verify Grover circuit via to_cirq and from_cirq roundtrip."""

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
    qrisp_sv = qrisp_circuit.statevector_array()

    cirq_circuit = convert_to_cirq(qrisp_circuit)
    cirq_sv = final_state_vector(cirq_circuit)
    np.testing.assert_array_almost_equal(
        np.zeros(np.shape(cirq_sv)),
        np.round(qrisp_sv - cirq_sv),
    )

    qrisp_restored = QuantumCircuit.from_cirq(cirq_circuit)
    np.testing.assert_array_almost_equal(qrisp_restored.statevector_array(), qrisp_sv, decimal=5)


def test_recursive_conversion():
    """Verify an op that is non-elementary and has the definition attribute can be converted recursively."""
    qc = QuantumCircuit(4)
    qc.rxx(0.3, 0, 1)
    qc.rxx(0.3, 2, 3)

    cirq_circuit = qc.to_cirq()
    assert qc.num_qubits() == num_qubits(cirq_circuit)


def test_convert_to_cirq_cirq_qubits_passthrough():
    """Providing custom cirq_qubits should be preserved after transpilation."""
    from qrisp import QPE, QuantumVariable, h, p

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
    """Qrisp circuit with a MagicMock operation (triggers Exception during transpile)."""
    qc = QuantumCircuit(1)
    qc.data = [MagicMock(op=MagicMock(name="some_gate", params=[]), qubits=[MagicMock()])]
    return qc


def _mock_none_gate_circ():
    """Qrisp circuit with a gate mapping to None and no definition (rxx)."""
    qc = QuantumCircuit(2)
    mock_op = MagicMock()
    mock_op.name = "rxx"
    mock_op.params = []
    mock_op.definition = None
    qc.data = [MagicMock(op=mock_op, qubits=list(qc.qubits))]
    return qc


def _real_unknown_circ():
    """Circuit with a real operation not in gate_map and no definition."""
    from qrisp.circuit import Instruction, Operation

    qc = QuantumCircuit(1)
    op = Operation(name="foo", num_qubits=1)
    qc.data = [Instruction(op, qc.qubits, [])]
    return qc


@pytest.mark.parametrize(
    "circ_builder, match",
    [
        (_mock_unknown_circ, "could not be transpiled and are not supported"),
        (_mock_none_gate_circ, "has no Cirq equivalent"),
        (_real_unknown_circ, "could not be decomposed into elementary"),
    ],
    ids=["unknown_gate", "none_gate_no_definition", "real_unknown_no_progress"],
)
def test_convert_to_cirq_raises(circ_builder, match):
    """Verify convert_to_cirq raises ValueError for unsupported gates."""
    with pytest.raises(ValueError, match=match):
        convert_to_cirq(circ_builder())


def _build_single_qubit_roundtrip():
    """Qrisp circuit with all non-parametrized single-qubit gates (H, X, Y, Z, S, T, S†, T†, SX, SX†, ID)."""
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
    """Qrisp circuit with parametrized single-qubit gates (RX, RY, RZ, P)."""
    qc = QuantumCircuit(4)
    qc.rx(0.3, 0)
    qc.ry(0.4, 1)
    qc.rz(0.2, 2)
    qc.p(0.5, 3)
    return qc


def _build_two_qubit_roundtrip():
    """Qrisp circuit with two-qubit gates (CX, CZ, SWAP)."""
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cz(2, 3)
    qc.swap(0, 2)
    return qc


def _build_gphase_roundtrip():
    """Qrisp circuit with global phase gate after identity (X†X)."""
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.x(0)
    qc.gphase(0.5, 0)
    return qc


def _build_swap_roundtrip():
    """Qrisp circuit with SWAP followed by H."""
    qc = QuantumCircuit(2)
    qc.swap(0, 1)
    qc.h(0)
    return qc


def _build_iswap_roundtrip():
    """Qrisp circuit with ISWAP."""
    qc = QuantumCircuit(2)
    qc.iswap(0, 1)
    return qc


def _build_mixed_roundtrip():
    """Qrisp circuit mixing single-qubit, two-qubit, and parametrized gates."""
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
        _build_iswap_roundtrip,
        _build_mixed_roundtrip,
    ],
    ids=[
        "single_qubit",
        "parametrized",
        "two_qubit",
        "gphase",
        "swap",
        "iswap",
        "mixed",
    ],
)
def test_roundtrip_preserves_unitary(circ_builder):
    """Verify unitary is preserved through a Qrisp→Cirq→Qrisp round-trip."""
    qc = circ_builder()
    expected_unitary = qc.get_unitary()

    cirq_circ = convert_to_cirq(qc)
    qrisp_qc = convert_from_cirq(cirq_circ)

    np.testing.assert_array_almost_equal(qrisp_qc.get_unitary(), expected_unitary)


def _circ_h_cx_rz_measure():
    """Cirq circuit with H, CX, RZ, and a two-qubit measurement."""
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.rz(0.5)(q1), cirq.measure(q0, q1)])


def _circ_h_cx():
    """Cirq circuit with H and CX only (no measurement)."""
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])


@pytest.mark.parametrize(
    "circ_builder, use_classmethod, expected_names",
    [
        (_circ_h_cx_rz_measure, False, ["h", "cx", "rz", "measure", "measure"]),
        (_circ_h_cx, True, ["h", "cx"]),
    ],
    ids=["direct_cirq_circuit", "via_classmethod"],
)
def test_convert_from_cirq_circuits(circ_builder, use_classmethod, expected_names):
    """Test converting Cirq circuits built directly (not via round-trip)."""
    cirq_circ = circ_builder()

    if use_classmethod:
        qrisp_qc = QuantumCircuit.from_cirq(cirq_circ)
    else:
        qrisp_qc = convert_from_cirq(cirq_circ)

    assert qrisp_qc.num_qubits() == len(cirq_circ.all_qubits())
    assert [d.op.name for d in qrisp_qc.data] == expected_names

    # verify unitary when circuit has no measurement gates
    if not any(d.op.name == "measure" for d in qrisp_qc.data):
        np.testing.assert_array_almost_equal(np.abs(qrisp_qc.get_unitary()), np.abs(cirq.unitary(cirq_circ)))


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
    np.testing.assert_array_almost_equal(np.abs(qrisp_qc.get_unitary()), np.abs(expected_unitary))


@pytest.mark.parametrize(
    "gate_fn, exponent",
    [
        (cirq.H, 0.5),
        (cirq.CNOT, 0.5),
        (cirq.CZ, 0.5),
        (cirq.SWAP, 0.5),
        (cirq.H, -1),
        (cirq.CNOT, -1),
        (cirq.CZ, -1),
        (cirq.SWAP, -1),
    ],
    ids=[
        "H^0.5",
        "CX^0.5",
        "CZ^0.5",
        "SWAP^0.5",
        "H^-1",
        "CX^-1",
        "CZ^-1",
        "SWAP^-1",
    ],
)
def test_convert_from_cirq_exponent_guards(gate_fn, exponent):
    """Fractional exponents for H/CX/CZ/SWAP now decompose correctly.

    The comparison is phase-exact (no abs), so global-phase errors in the
    conversion would be caught.
    """
    q = cirq.LineQubit.range(3)
    g = gate_fn**exponent
    if g.num_qubits() == 1:
        cirq_circ = cirq.Circuit([g(q[0])])
    else:
        cirq_circ = cirq.Circuit([g(q[0], q[1])])

    qrisp_qc = convert_from_cirq(cirq_circ)
    expected_unitary = cirq.unitary(cirq_circ)
    np.testing.assert_array_almost_equal(qrisp_qc.get_unitary(), expected_unitary, decimal=5)


@pytest.mark.parametrize(
    "gate_fn, qubit_count",
    [
        (lambda q: (cirq.X.controlled() ** 0.25)(q[0], q[1]), 2),
        (lambda q: cirq.CXPowGate(exponent=0.5)(q[0], q[1]), 2),
        (lambda q: (cirq.H.controlled() ** 0.5)(q[0], q[1]), 2),
        (lambda q: (cirq.SWAP**0.5)(q[0], q[1]).controlled_by(q[2]), 3),
        (lambda q: (cirq.ISWAP**0.5)(q[0], q[1]), 2),
        (lambda q: (cirq.X.controlled() ** 2)(q[0], q[1]), 2),
        (lambda q: cirq.CCXPowGate(exponent=0.5)(q[0], q[1], q[2]), 3),
        (lambda q: cirq.CCZPowGate(exponent=0.5)(q[0], q[1], q[2]), 3),
    ],
    ids=[
        "controlled_x_fractional",
        "cx_fractional",
        "controlled_h_fractional",
        "controlled_swap_fractional",
        "iswap_fractional",
        "controlled_x_even",
        "ccx_fractional",
        "ccz_fractional",
    ],
)
def test_convert_from_cirq_fractional_powers_phase_exact(gate_fn, qubit_count):
    """Fractional and even power gates in controlled contexts preserve cirq's exact phase.

    Cirq uses the convention X^t = exp(i*pi*t/2) * RX(pi*t).  A direct
    conversion of the controlled variant would turn that global phase into
    a *relative* phase between control branches; these gates are therefore
    decomposed by cirq first.  The unitary comparison is phase-exact so
    any phase error would be caught.

    The round-trip (convert back to cirq) additionally verifies that the
    resulting operations' *definitions* are correct, not just their
    reported unitaries.
    """
    q = cirq.LineQubit.range(qubit_count)
    cirq_circ = cirq.Circuit([gate_fn(q)])
    qrisp_qc = convert_from_cirq(cirq_circ)
    np.testing.assert_array_almost_equal(
        qrisp_qc.get_unitary(), cirq.unitary(cirq_circ), decimal=5
    )
    # Round-trip back to cirq to verify the operations' *definitions* are
    # correct, not just their reported unitaries.  Skipped when transpilation
    # dropped redundant (identity-supporting) qubits.
    u_roundtrip = cirq.unitary(convert_to_cirq(qrisp_qc))
    if u_roundtrip.shape == cirq.unitary(cirq_circ).shape:
        np.testing.assert_array_almost_equal(u_roundtrip, cirq.unitary(cirq_circ), decimal=5)


def test_convert_from_cirq_even_powers_are_identity():
    """Even powers of X convert to the identity gate (phase-exact)."""
    q0, q1 = cirq.LineQubit.range(2)

    qrisp_qc = convert_from_cirq(cirq.Circuit([cirq.X(q0) ** 2]))
    assert qrisp_qc.data[0].op.name == "id"
    np.testing.assert_array_almost_equal(qrisp_qc.get_unitary(), np.eye(2))

    qrisp_qc = convert_from_cirq(cirq.Circuit([cirq.X(q0) ** -1]))
    assert qrisp_qc.data[0].op.name == "x"
    np.testing.assert_array_almost_equal(qrisp_qc.get_unitary(), cirq.unitary(cirq.X))

    qrisp_qc = convert_from_cirq(cirq.Circuit([(cirq.X.controlled() ** 2)(q0, q1)]))
    np.testing.assert_array_almost_equal(qrisp_qc.get_unitary(), np.eye(4))


def test_convert_from_cirq_fractional_iswap_decomposes():
    """Fractional ISWAP powers are decomposed by cirq instead of raising."""
    q0, q1 = cirq.LineQubit.range(2)
    cirq_circ = cirq.Circuit([cirq.ISWAP(q0, q1) ** 0.5])
    qrisp_qc = convert_from_cirq(cirq_circ)
    assert len(qrisp_qc.data) > 1
    np.testing.assert_array_almost_equal(
        qrisp_qc.get_unitary(), cirq.unitary(cirq_circ), decimal=5
    )


@pytest.mark.parametrize(
    "cirq_circ",
    [
        cirq.Circuit([(cirq.X.controlled() ** 0.25)(cirq.LineQubit(0), cirq.LineQubit(1))]),
        cirq.Circuit([(cirq.H.controlled() ** 0.5)(cirq.LineQubit(0), cirq.LineQubit(1))]),
    ],
    ids=["controlled_x_fractional_roundtrip", "controlled_h_fractional_roundtrip"],
)
def test_convert_from_cirq_fractional_controlled_statevector(cirq_circ):
    """Full statevector (incl. global phase) matches for fractional controlled gates."""
    qrisp_qc = convert_from_cirq(cirq_circ)
    np.testing.assert_array_almost_equal(
        qrisp_qc.statevector_array(),
        cirq.final_state_vector(cirq_circ),
        decimal=5,
    )


# -----------------------------------------------------------------------
# Documented global-phase deviations (xfail, strict)
#
# Cirq's convention is X^t = exp(i*pi*t/2) * RX(pi*t) (likewise for Y),
# whereas Qrisp's RX/RY carry no phase and SXGate == RX(pi/2).  For an
# uncontrolled gate this only costs a global phase, which is unobservable
# and tolerated by mitiq's up-to-global-phase comparisons.  Controlled
# fractional powers are converted exactly via cirq.decompose instead (see
# test_convert_from_cirq_fractional_powers_phase_exact).
# -----------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="Qrisp RX/RY carry no global phase; cirq uses X^t = exp(i*pi*t/2)*RX(pi*t). "
    "Standalone fractional X/Y powers therefore match only up to a global phase.",
)
@pytest.mark.parametrize(
    "gate",
    [cirq.X**0.25, cirq.Y**0.25, cirq.X**0.1, cirq.Y**0.9],
    ids=["X^0.25", "Y^0.25", "X^0.1", "Y^0.9"],
)
def test_standalone_fractional_pauli_phase_exact_xfail(gate):
    """Phase-exact (no abs) conversion of a standalone fractional Pauli power.

    Documents the known global-phase deviation. If this ever starts passing,
    the converter became phase-exact and the xfail marker should be removed.
    """
    q0 = cirq.LineQubit(0)
    cirq_circ = cirq.Circuit([gate(q0)])
    qrisp_qc = convert_from_cirq(cirq_circ)
    np.testing.assert_array_almost_equal(
        qrisp_qc.get_unitary(), cirq.unitary(cirq_circ), decimal=5
    )


@pytest.mark.xfail(
    strict=True,
    reason="Qrisp SXGate == RX(pi/2), cirq X**0.5 == exp(i*pi/4)*RX(pi/2); "
    "to_cirq(sx) differs by a global phase.",
)
def test_to_cirq_sx_phase_exact_xfail():
    """Phase-exact to_cirq of sx/sx_dg.

    Documents the known global-phase deviation in the Qrisp->Cirq direction
    (see the Notes section of convert_to_cirq). If this ever starts passing,
    the xfail marker should be removed.
    """
    qc = QuantumCircuit(1)
    qc.sx(0)
    np.testing.assert_array_almost_equal(
        qc.get_unitary(), cirq.unitary(convert_to_cirq(qc)), decimal=5
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
        "X.controlled()": cirq.Circuit([cirq.X.controlled()(q0, q1)]),
        "ControlledOperation": cirq.Circuit([cirq.ControlledOperation([q0], cirq.X(q1))]),
        "Toffoli_via_ControlledGate": cirq.Circuit([cirq.ControlledGate(cirq.X, num_controls=2)(q0, q1, q2)]),
        "anti_control_via_control_values": cirq.Circuit([cirq.X.controlled(control_values=[0, 1])(q0, q1, q2)]),
    }

    cirq_circ = builders[key]
    qrisp_qc = convert_from_cirq(cirq_circ)

    expected_unitary = cirq.unitary(cirq_circ)
    np.testing.assert_array_almost_equal(qrisp_qc.get_unitary(), expected_unitary)


def test_convert_from_cirq_iswap():
    """ISWAP gate should convert correctly."""
    q0, q1 = cirq.LineQubit.range(2)
    cirq_circ = cirq.Circuit([cirq.ISWAP(q0, q1)])
    qrisp_qc = convert_from_cirq(cirq_circ)
    expected_unitary = cirq.unitary(cirq_circ)
    np.testing.assert_array_almost_equal(
        np.abs(qrisp_qc.get_unitary()),
        np.abs(expected_unitary),
    )


def test_convert_from_cirq_iswap_adjoint():
    """ISWAP† gate should convert correctly."""
    q0, q1 = cirq.LineQubit.range(2)
    cirq_circ = cirq.Circuit([cirq.ISWAP(q0, q1) ** -1])
    qrisp_qc = convert_from_cirq(cirq_circ)
    expected_unitary = cirq.unitary(cirq_circ)
    np.testing.assert_array_almost_equal(
        np.abs(qrisp_qc.get_unitary()),
        np.abs(expected_unitary),
    )


def test_convert_from_cirq_ccx():
    """Toffoli (CCX) gate should convert correctly."""
    q0, q1, q2 = cirq.LineQubit.range(3)
    cirq_circ = cirq.Circuit([cirq.TOFFOLI(q0, q1, q2)])
    qrisp_qc = convert_from_cirq(cirq_circ)
    expected_unitary = cirq.unitary(cirq_circ)
    np.testing.assert_array_almost_equal(qrisp_qc.get_unitary(), expected_unitary)


def test_convert_from_cirq_ccz():
    """CCZ gate should convert correctly."""
    q0, q1, q2 = cirq.LineQubit.range(3)
    cirq_circ = cirq.Circuit([cirq.CCZPowGate()(q0, q1, q2)])
    qrisp_qc = convert_from_cirq(cirq_circ)
    expected_unitary = cirq.unitary(cirq_circ)
    np.testing.assert_array_almost_equal(qrisp_qc.get_unitary(), expected_unitary)


def test_convert_from_cirq_unsupported_gate_raises():
    """A custom Cirq gate without _decompose_ still raises ValueError."""
    class _UnsupportedGate(cirq.Gate):
        def _num_qubits_(self): return 1
        def _unitary_(self): return np.array([[1, 0], [0, -1]])
    q0 = cirq.LineQubit(0)
    circ = cirq.Circuit([_UnsupportedGate()(q0)])
    with pytest.raises(ValueError, match="could not be decomposed"):
        convert_from_cirq(circ)


# -----------------------------------------------------------------------
# Direct unit tests for _decompose_unknown_ops
# -----------------------------------------------------------------------


def test_decompose_known_gates_pass_through():
    """A circuit with all-known gates is returned unchanged."""
    q = cirq.LineQubit(0)
    circ = cirq.Circuit([cirq.H(q), cirq.X(q), cirq.Z(q)])
    result = _decompose_unknown_ops(circ)
    assert result == circ


def test_decompose_empty_circuit():
    """An empty circuit passes through without error."""
    circ = cirq.Circuit()
    result = _decompose_unknown_ops(circ)
    assert result == circ


def test_decompose_fsim_gate():
    """FSimGate (previously unsupported) should decompose successfully."""
    q0, q1 = cirq.LineQubit.range(2)
    circ = cirq.Circuit([cirq.FSimGate(theta=0.5, phi=0.25)(q0, q1)])
    result = _decompose_unknown_ops(circ)
    # Should produce more ops than the single FSimGate
    assert len(list(result.all_operations())) > 1
    qc = convert_from_cirq(result)
    np.testing.assert_array_almost_equal(
        np.abs(qc.get_unitary()), np.abs(cirq.unitary(circ))
    )


def test_decompose_cswap_gate():
    """CSwapGate (Fredkin) should decompose successfully."""
    q0, q1, q2 = cirq.LineQubit.range(3)
    circ = cirq.Circuit([cirq.CSWAP(q0, q1, q2)])
    result = _decompose_unknown_ops(circ)
    assert len(list(result.all_operations())) > 1
    qc = convert_from_cirq(result)
    np.testing.assert_array_almost_equal(
        np.abs(qc.get_unitary()), np.abs(cirq.unitary(circ))
    )


def test_decompose_unknown_gate_raises():
    """A gate without _decompose_ raises ValueError with a descriptive message."""
    class _StuckGate(cirq.Gate):
        def _num_qubits_(self): return 1
        def _unitary_(self): return np.array([[1, 0], [0, -1]])
    q0 = cirq.LineQubit(0)
    circ = cirq.Circuit([_StuckGate()(q0)])
    with pytest.raises(ValueError, match="could not be decomposed"):
        _decompose_unknown_ops(circ)


def test_decompose_multi_valued_control():
    """Multi-valued control (0 OR 1) now decomposes successfully.
    
    The redundant control qubit is dropped by Cirq during decomposition,
    so the resulting circuit has fewer qubits. We verify unitary is correct
    on the remaining subsystem.
    """
    from cirq.ops.control_values import ProductOfSums

    q0, q1 = cirq.LineQubit.range(2)
    cv = ProductOfSums([(0, 1)])
    g = cirq.ControlledGate(cirq.X, control_values=cv)
    circ = cirq.Circuit([cirq.GateOperation(g, [q0, q1])])
    result = _decompose_unknown_ops(circ)
    assert len(list(result.all_operations())) > 0
    # The decomposed circuit should work for X gate on the second qubit
    qc = convert_from_cirq(result)
    assert qc.data[0].op.name == "x"


def test_decompose_circuit_operation():
    """CircuitOperation (no .gate) is decomposed into a simple operation."""
    q0 = cirq.LineQubit(0)
    inner = cirq.FrozenCircuit([cirq.X(q0)])
    circ = cirq.Circuit([cirq.CircuitOperation(inner, use_repetition_ids=True)])
    result = _decompose_unknown_ops(circ)
    ops = list(result.all_operations())
    assert len(ops) == 1
    assert isinstance(ops[0].gate, cirq.XPowGate)


def test_decompose_controlled_circuit_operation():
    """ControlledOperation wrapping a CircuitOperation is decomposed."""
    q0, q1 = cirq.LineQubit.range(2)
    inner = cirq.FrozenCircuit([cirq.X(q0)])
    co = cirq.CircuitOperation(inner, use_repetition_ids=True)
    circ = cirq.Circuit([cirq.ControlledOperation([q1], co)])
    result = _decompose_unknown_ops(circ)
    qc = convert_from_cirq(result)
    np.testing.assert_array_almost_equal(
        np.abs(qc.get_unitary()), np.abs(cirq.unitary(circ))
    )


def test_decompose_multiple_unsupported_gates():
    """Multiple unsupported gates in sequence all get decomposed."""
    q0, q1 = cirq.LineQubit.range(2)
    circ = cirq.Circuit([
        cirq.FSimGate(theta=0.3, phi=0.1)(q0, q1),
        cirq.FSimGate(theta=0.7, phi=0.4)(q0, q1),
    ])
    result = _decompose_unknown_ops(circ)
    qc = convert_from_cirq(result)
    np.testing.assert_array_almost_equal(
        np.abs(qc.get_unitary()), np.abs(cirq.unitary(circ))
    )


def test_decompose_iterative_progress():
    """Decomposition that requires multiple iterations still succeeds.
    
    A ControlledOp with multi-valued control (0 OR 1) on one qubit and
    normal control on another.  Cirq drops the redundant control qubit
    during decomposition, reducing the circuit width.
    """
    q0, q1, q2 = cirq.LineQubit.range(3)
    g = cirq.ControlledGate(cirq.X, control_values=[(0, 1), 1])
    circ = cirq.Circuit([g(q0, q1, q2)])
    result = _decompose_unknown_ops(circ)
    # Redundant qubit q0 was dropped — should work on remaining qubits
    qc = convert_from_cirq(result)
    assert len(qc.data) > 0


def test_convert_from_cirq_circuit_operation():
    """CircuitOperation (no .gate attribute) should now be decomposed."""
    q0 = cirq.LineQubit(0)
    inner = cirq.FrozenCircuit([cirq.X(q0)])
    circ = cirq.Circuit([cirq.CircuitOperation(inner, use_repetition_ids=True)])
    qrisp_qc = convert_from_cirq(circ)
    assert qrisp_qc.data[0].op.name == "x"


def test_convert_from_cirq_controlled_circuit_operation():
    """ControlledOperation wrapping a CircuitOperation should be decomposed."""
    q0, q1 = cirq.LineQubit.range(2)
    inner = cirq.FrozenCircuit([cirq.X(q0)])
    co = cirq.CircuitOperation(inner, use_repetition_ids=True)
    circ = cirq.Circuit([cirq.ControlledOperation([q1], co)])
    qrisp_qc = convert_from_cirq(circ)
    expected_unitary = cirq.unitary(circ)
    np.testing.assert_array_almost_equal(
        np.abs(qrisp_qc.get_unitary()), np.abs(expected_unitary)
    )


def _empty_circ():
    """Empty Cirq circuit (no qubits, no operations)."""
    return cirq.Circuit()


def _global_phase_circ():
    """Cirq circuit with only a global phase gate (acts on 0 qubits)."""
    return cirq.Circuit([cirq.GlobalPhaseGate(1j).on()])


@pytest.mark.parametrize(
    "circ_builder",
    [_empty_circ, _global_phase_circ],
    ids=["empty_circuit", "global_phase_only"],
)
def test_convert_from_cirq_edge_circuits(circ_builder):
    """Edge case circuits should convert without error."""
    qrisp_qc = convert_from_cirq(circ_builder())
    assert qrisp_qc.num_qubits() == 0
    assert len(qrisp_qc.data) == 0


def test_convert_from_cirq_single_qubit_measure():
    """Single-qubit measure hits the len(qrisp_qubits) == 1 branch."""
    q0 = cirq.LineQubit(0)
    circ = cirq.Circuit([cirq.measure(q0)])
    qrisp_qc = convert_from_cirq(circ)
    assert qrisp_qc.data[0].op.name == "measure"


@pytest.mark.parametrize(
    "control_values",
    [[1], [0]],
    ids=["control_value_1", "control_value_0"],
)
def test_convert_from_cirq_controlled_gate_unwrap(control_values):
    """GateOperation wrapping a ControlledGate enters the unwrap while-loop."""
    q0, q1 = cirq.LineQubit.range(2)
    g = cirq.ControlledGate(cirq.X, num_controls=1, control_values=control_values)
    circ = cirq.Circuit([cirq.GateOperation(g, [q0, q1])])
    qrisp_qc = convert_from_cirq(circ)
    assert qrisp_qc.data[0].op.name == "cx"


@pytest.mark.parametrize(
    "gate, expected_name",
    [
        (cirq.X**0.25, "rx"),
        (cirq.Y**0.25, "ry"),
    ],
    ids=["xpowgate_generic", "ypowgate_generic"],
)
def test_convert_from_cirq_generic_rotation(gate, expected_name):
    """Generic rotation gates (non-special exponent) fall to RX/RY."""
    q0 = cirq.LineQubit(0)
    circ = cirq.Circuit([gate(q0)])
    qrisp_qc = convert_from_cirq(circ)
    assert qrisp_qc.data[0].op.name == expected_name


@pytest.mark.parametrize(
    "name, params",
    [
        ("sx", []),
        ("sx_dg", []),
        ("p", [0.5]),
        ("rx", [0.5]),
        ("y", []),
    ],
    ids=[
        "controlled_sx",
        "controlled_sx_dg",
        "controlled_p",
        "controlled_params",
        "controlled_fallback",
    ],
)
def test_convert_to_cirq_controlled_branches(name, params):
    """Cover branches in the ControlledOperation block of convert_to_cirq."""
    from qrisp.circuit.standard_operations import XGate

    qc = QuantumCircuit(2)
    op = ControlledOperation(XGate(), 1)
    op.name = name
    op.params = params
    qc.append(op, qc.qubits)
    result = convert_to_cirq(qc)
    assert isinstance(result, cirq.Circuit)


def _make_import_fail():
    import builtins

    real = builtins.__import__

    def mock(name, *args, **kwargs):
        if name == "cirq":
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real(name, *args, **kwargs)

    return patch("builtins.__import__", mock)


@pytest.mark.parametrize(
    "func, circuit",
    [
        (convert_to_cirq, QuantumCircuit(1)),
        (convert_from_cirq, cirq.Circuit()),
    ],
    ids=["convert_to_cirq", "convert_from_cirq"],
)
def test_convert_import_error(func, circuit):
    """ImportError when cirq is not installed."""
    with _make_import_fail():
        with pytest.raises(ImportError, match="Cirq must be installed"):
            func(circuit)


def test_convert_from_cirq_mixed_qubit_types():
    """Mixed qubit types raise ValueError caught by converter."""
    q0 = cirq.LineQubit(0)
    circ = cirq.Circuit([cirq.H(q0)])

    class BadQubit:
        def __init__(self):
            self.dimension = 2

        def __repr__(self):
            return "BadQubit()"

        def _comparison_key(self):
            return 0

    with patch.object(circ, "all_qubits", return_value={q0, BadQubit()}):
        with pytest.raises(ValueError, match="Mixed qubit types"):
            convert_from_cirq(circ)


def _build_invalid_control_inner_circ():
    """GateOperation wrapping a ControlledGate with control value 2 (bypassed Cirq validation),
    triggering unsupported-value error in the unwrap while-loop.
    """
    from cirq.ops.control_values import ProductOfSums

    q0, q1 = cirq.LineQubit.range(2)
    g = cirq.ControlledGate(cirq.X, num_controls=1, control_values=[1])
    g._control_values = ProductOfSums([2])
    return cirq.Circuit([cirq.GateOperation(g, [q0, q1])])


def _build_invalid_control_outer_circ():
    """ControlledOperation with control value 2 (bypassed Cirq validation),
    triggering unsupported-value error in the extra_controls path.
    """
    from cirq.ops.control_values import ProductOfSums

    q0, q1 = cirq.LineQubit.range(2)
    co = cirq.ControlledOperation([q0], cirq.X(q1), control_values=[1])
    co._control_values = ProductOfSums([2])
    return cirq.Circuit([co])


@pytest.mark.parametrize(
    "circ_builder, match",
    [
        (_build_invalid_control_inner_circ, "could not be decomposed"),
        (_build_invalid_control_outer_circ, "could not be decomposed"),
    ],
    ids=["invalid_control_inner", "invalid_control_outer"],
)
def test_convert_from_cirq_control_value_errors(circ_builder, match):
    """Control value validation errors in both unwrap and extra_controls paths."""
    with pytest.raises(ValueError, match=match):
        convert_from_cirq(circ_builder())
