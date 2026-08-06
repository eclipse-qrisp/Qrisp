from qrisp import QuantumCircuit, QuantumVariable
from pyzx import Circuit

from fractions import Fraction
import numpy as np
from unittest.mock import MagicMock
import pytest


def _build_single_qubit_qrisp_circuit():
    """Qrisp circuit with single-qubit gates.
    Example adapted from cirq converter test"""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.x(1)
    qc.y(3)
    qc.z(2)
    qc.rx(0.3, 3)
    qc.ry(0.4, 1)
    qc.rz(0.2, 2)
    qc.u3(0.2, 0.3, 0.4, 0)
    qc.p(0.6, 0)
    qc.s(0)
    qc.t(1)
    qc.sx(3)
    qc.t_dg(3)
    qc.s_dg(2)
    qc.sx_dg(3)
    qc.gphase(0.5, 0)
    qc.id(0)
    return qc


def _build_multi_qubit_qrisp_circuit():
    """Qrisp circuit with multi-qubit gates."""
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cy(2, 3)
    qc.cz(0, 2)
    qc.swap(2, 3)
    qc.xxyy(0, 1, 2, 3)
    qc.rxx(0.1, 0, 3)
    qc.rzz(0.2, 1, 2)
    return qc


def _build_single_qubit_pyzx_circuit():
    """PyZX circuit with single-qubit gates."""
    c = Circuit(4)
    c.add_gate("NOT", 0)
    c.add_gate("Y", 1)
    (c.add_gate("Z", 2),)
    c.add_gate("HAD", 3)
    c.add_gate("XPhase", 0, Fraction(2, 3))
    c.add_gate("YPhase", 1, Fraction(1, 3))
    c.add_gate("ZPhase", 2, Fraction(1, 6))
    c.add_gate("U2", 3, Fraction(2, 7), Fraction(9, 8))
    c.add_gate("U3", 0, Fraction(6, 7), Fraction(3, 2), Fraction(5, 4))
    c.add_gate("SX", 1)
    c.add_gate("S", 2)
    c.add_gate("T", 3)
    return c


def _build_multi_qubit_pyzx_circuit():
    """PyZX circuit with multi-qubit gates."""
    c = Circuit(4)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("CY", 2, 3)
    c.add_gate("CZ", 1, 2)
    c.add_gate("CRX", 0, 2, Fraction(2, 3))
    c.add_gate("CRY", 0, 2, Fraction(4, 3))
    c.add_gate("CRZ", 0, 2, Fraction(1, 2))
    c.add_gate("CPhase", 0, 1, Fraction(4, 5))
    c.add_gate("ParityPhase", Fraction(6, 5), 0, 2, 3)
    c.add_gate("XCX", 0, 1)
    # SX gate has a different phase definition in PyZX, hence CSX gives a unitary that is equivalent, but does not only differ in a global phase
    # leave out for now
    # c.add_gate("CSX", 0, 1)
    c.add_gate("SWAP", 0, 2)
    c.add_gate("CSWAP", 0, 1, 3)
    c.add_gate("CHAD", 0, 1)
    c.add_gate("TOF", 1, 2, 3)
    c.add_gate("CCZ", 0, 2, 3)
    c.add_gate("CU3", 0, 1, Fraction(6, 5), Fraction(3, 2), Fraction(7, 4))
    c.add_gate("CU", 1, 3, Fraction(7, 9), Fraction(6, 5), Fraction(3, 2), Fraction(7, 4))
    c.add_gate("RZZ", 0, 2, Fraction(2, 9))
    c.add_gate("RXX", 0, 2, Fraction(7, 10))

    return c


def _build_mock_qrisp_circuit():
    """Qrisp circuit with a MagicMock operation (triggers Exception during conversion)."""
    qc = QuantumCircuit(1)
    qc.data = [MagicMock(op=MagicMock(name="some_gate", params=[]), qubits=[MagicMock()])]
    return qc


def _build_mock_pyzx_circuit():
    """Qrisp circuit with a MagicMock operation (triggers Exception during conversion)."""
    c = Circuit(1)
    c.add_gate(MagicMock(name="some_gate"), 0)
    return c


def _compare_unitaries(U1, U2):
    """Compare two unitaries, taking into account a potential phase mismatch"""
    # PyZX does not keep track of global phases and instead normalizes the phase,
    # which leads to a potential phase mismatch.
    # We determine the phase difference by dividing the matrix entries with the largest magnitude
    ind_max = np.unravel_index(np.argmax(np.abs(U1)), U1.shape)
    phase = U2[ind_max] / U1[ind_max]

    np.testing.assert_array_almost_equal(phase * U1, U2)


def _test_qrisp_to_pyzx(qc):
    """Test conversion of qrisp circuit qc to pyzx circuit"""
    c = qc.to_pyzx()
    _compare_unitaries(qc.get_unitary(), c.to_matrix())


def _test_pyzx_to_qrisp(c):
    """Test conversion of pyzx circuit c to qrisp circuit"""
    qc = QuantumCircuit.from_pyzx(c)
    _compare_unitaries(qc.get_unitary(), c.to_matrix())


def _test_roundtrip(qc):
    """Test roundtrip starting from qrisp circuit qc"""
    c = qc.to_pyzx()
    qc2 = QuantumCircuit.from_pyzx(c)
    _compare_unitaries(qc.get_unitary(), qc2.get_unitary())


def _test_roundtrip_reverse(c):
    """Test roundtrip starting from pyzx circuit c"""
    qc = QuantumCircuit.from_pyzx(c)
    c2 = qc.to_pyzx()
    _compare_unitaries(c.to_matrix(), c2.to_matrix())


def test_single_qubit_circuit_qrisp_to_pyzx():
    """Test conversion of qrisp circuit to pyzx circuit for single qubit gates"""
    qc = _build_single_qubit_qrisp_circuit()
    _test_qrisp_to_pyzx(qc)


def test_multi_qubit_circuit_qrisp_to_pyzx():
    """Test conversion of qrisp circuit to pyzx circuit for multi qubit gates"""
    qc = _build_multi_qubit_qrisp_circuit()
    _test_qrisp_to_pyzx(qc)


def test_single_qubit_circuit_pyzx_to_qrisp():
    """Test conversion of pyzx circuit to qrisp circuit for single qubit gates"""
    c = _build_single_qubit_pyzx_circuit()
    _test_pyzx_to_qrisp(c)


def test_multi_qubit_circuit_pyzx_to_qrisp():
    """Test conversion of pyzx circuit to qrisp circuit for multi qubit gates"""
    c = _build_multi_qubit_pyzx_circuit()
    _test_pyzx_to_qrisp(c)


def test_single_qubit_circuit_roundtrip():
    """Test roundtrip starting from qrisp circuit for single qubit gates"""
    qc = _build_single_qubit_qrisp_circuit()
    _test_roundtrip(qc)


def test_multi_qubit_circuit_roundtrip():
    """Test roundtrip starting from qrisp circuit for multi qubit gates"""
    qc = _build_multi_qubit_qrisp_circuit()
    _test_roundtrip(qc)


def test_single_qubit_circuit_roundtrip_reverse():
    """Test roundtrip starting from pyzx circuit for single qubit gates"""
    c = _build_single_qubit_pyzx_circuit()
    _test_roundtrip_reverse(c)


def test_multi_qubit_circuit_roundtrip_reverse():
    """Test roundtrip starting from pyzx circuit for multi qubit gates"""
    c = _build_multi_qubit_pyzx_circuit()
    _test_roundtrip_reverse(c)


def test_qrisp_transpilation():
    """Test transpilation capability of converter for a circuit that has to be transpiled.
    Example taken from test_cirq_converter.py"""
    from qrisp import p, h, QPE

    def U(qv):
        x = 0.5
        y = 0.125

        p(x * 2 * np.pi, qv[0])
        p(y * 2 * np.pi, qv[1])

    qv = QuantumVariable(2)
    h(qv)
    QPE(qv, U, precision=3)
    qc = qv.qs.compile()

    _test_qrisp_to_pyzx(qc)
    _test_roundtrip(qc)


def test_non_unitaries():
    """Test measurement and reset"""
    qc = QuantumCircuit(2)
    qc.measure(0)
    qc.reset(1)
    c = qc.to_pyzx()
    assert [g.name for g in c.gates] == ["Measurement", "Reset"]
    assert [g.target for g in c.gates] == [0, 1]

    c = Circuit(2)
    c.add_gate("Measurement", 0)
    c.add_gate("Reset", 1)
    qc = QuantumCircuit.from_pyzx(c)
    assert [g.op.name for g in qc.data] == ["measure", "reset"]


def test_error_qrisp_to_pyzx():
    qc = _build_mock_qrisp_circuit()
    with pytest.raises(ValueError):
        c = qc.to_pyzx()


def test_error_pyzx_to_qrisp():
    c = _build_mock_pyzx_circuit()
    with pytest.raises(ValueError):
        qc = QuantumCircuit.from_pyzx(c)
