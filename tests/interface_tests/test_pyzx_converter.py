"""Tests for the Qrisp <-> PyZX circuit converter."""

import sys
from fractions import Fraction
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pyzx import Circuit
from pyzx.circuit import gates as pyzx_gates

from qrisp import QuantumCircuit, QuantumVariable, U1Gate
from qrisp.circuit import standard_operations as qrisp_ops


def test_import_error():
    """to_pyzx raises ImportError when PyZX is not installed."""
    with pytest.raises(ImportError):
        with patch.dict(sys.modules, {"pyzx": None}):
            qc = QuantumCircuit(1)
            qc.to_pyzx()


def _build_single_qubit_qrisp_circuit():
    """Qrisp circuit with single-qubit gates. Example adapted from cirq converter test"""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.x(1)
    qc.y(3)
    qc.z(2)
    qc.rx(0.3, 3)
    qc.ry(0.4, 1)
    qc.rz(0.2, 2)
    qc.append(U1Gate(1.2), 0)
    qc.u3(0.2, 0.3, 0.4, 0)
    qc.r(0.1, 0.5, 2)
    qc.p(0.6, 0)
    qc.s(0)
    qc.t(1)
    qc.sx(3)
    qc.t_dg(3)
    qc.s_dg(2)
    qc.sx_dg(3)
    qc.gphase(0.5, 0)
    qc.id(0)
    qc.barrier()
    return qc


def _build_multi_qubit_qrisp_circuit():
    """Qrisp circuit with multi-qubit gates."""
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cy(2, 3)
    qc.cz(0, 2)
    qc.swap(2, 3)
    qc.xxyy(0.1, 0.3, 2, 3)
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
    c.add_gate("SX", 1, True)  # adjoint versions
    c.add_gate("S", 2, True)
    c.add_gate("T", 3, True)
    c.add_gate("SX", 3, False)  # non-adjoint versions
    c.add_gate("S", 0, False)
    c.add_gate("T", 1, False)
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
    c.add_gate("PhaseGadget", Fraction(3, 5), 1, 2, 3)
    c.add_gate("XCX", 0, 1)
    c.add_gate("CSX", 0, 1)
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
    """Test transpilation for a circuit that has to be decomposed first.

    Example adapted from test_cirq_converter.py.
    """
    from qrisp import QPE, h, p

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
    qc = QuantumCircuit(2, 1)
    qc.measure(0, 0)
    qc.reset(1)
    c = qc.to_pyzx()
    assert [g.name for g in c.gates] == ["Measurement", "Reset"]
    assert [g.target for g in c.gates] == [0, 1]
    assert c.gates[0].result_bit == 0

    c = Circuit(2, bit_amount=2)
    c.add_gate("Measurement", 0, 1)
    c.add_gate("Reset", 1)
    qc = QuantumCircuit.from_pyzx(c)
    assert [g.op.name for g in qc.data] == ["measure", "reset"]
    assert [g.qubits[0] for g in qc.data] == [qc.qubits[0], qc.qubits[1]]
    assert qc.data[0].clbits[0] == qc.clbits[1]


def _mock_unknown_circuit():
    """Qrisp circuit with a MagicMock operation (triggers Exception during transpile)."""
    qc = QuantumCircuit(1)
    qc.data = [MagicMock(op=MagicMock(name="some_gate", params=[]), qubits=[MagicMock()])]
    return qc


def _mock_none_gate_circuit():
    """Qrisp circuit with a gate mapping to None and no definition (rxx)."""
    qc = QuantumCircuit(2)
    mock_op = MagicMock()
    mock_op.name = "xxyy"
    mock_op.params = []
    mock_op.definition = None
    qc.data = [MagicMock(op=mock_op, qubits=list(qc.qubits))]
    return qc


def _real_unknown_circuit():
    """Circuit with a real operation not in gate_map and no definition."""
    from qrisp.circuit import Instruction, Operation

    qc = QuantumCircuit(1)
    op = Operation(name="foo", num_qubits=1)
    qc.data = [Instruction(op, qc.qubits, [])]
    return qc


@pytest.mark.parametrize(
    "circuit_builder, match",
    [
        (_mock_unknown_circuit, "could not be transpiled and are not supported"),
        (_mock_none_gate_circuit, "has no PyZX equivalent"),
        (_real_unknown_circuit, "could not be decomposed into elementary"),
    ],
    ids=["unknown_gate", "none_gate_no_definition", "real_unknown_no_progress"],
)
def test_convert_to_pyzx_raises(circuit_builder, match):
    """Verify convert_to_cirq raises ValueError for unsupported gates."""
    with pytest.raises(ValueError, match=match):
        qc = circuit_builder()
        qc.to_pyzx()


def _build_pyzx_circuit_with_mock_gate():
    """Qrisp circuit with a MagicMock operation (triggers immediate exception during conversion)."""
    c = Circuit(1)
    c.add_gate(MagicMock(name="some_gate"), 0)
    return c


def _build_pyzx_circuit_with_undecomposable_gate():
    """Qrisp circuit with a gate with mock decomposition (triggers exception upon attempt to decompose gate)."""
    c = Circuit(2)
    from pyzx.circuit import ParityPhase

    g = ParityPhase(0, 0, 1)

    def _f():
        return [ParityPhase(0, 0, 1)]

    g.to_basic_gates = _f
    c.add_gate(g, 0)
    return c


def test_error_pyzx_to_qrisp_mock_gate():
    """from_pyzx raises ValueError for a gate unknown to PyZX."""
    c = _build_pyzx_circuit_with_mock_gate()
    with pytest.raises(ValueError, match="of PyZX is unknown"):
        QuantumCircuit.from_pyzx(c)


def test_error_pyzx_to_qrisp_undecomposable_gate():
    """from_pyzx raises ValueError for a gate PyZX cannot decompose."""
    c = _build_pyzx_circuit_with_undecomposable_gate()
    with pytest.raises(ValueError, match="cannot be decomposed either"):
        QuantumCircuit.from_pyzx(c)


# ---------------------------------------------------------------------------
# Smoke and gap tests
#
# The tests above verify correctness (unitary equivalence, round-trips) for a
# representative set of gates.  These tests add complementary breadth:
#
#   * ``*_smoke`` walks the complete gate inventory and only asserts graceful
#     handling -- the converter either converts or raises a clear ValueError,
#     never an undocumented exception.
#   * ``*_gap_gate`` covers the gates the fixtures above do not touch, so no
#     conversion assertion is duplicated.
# ---------------------------------------------------------------------------

# Aliases accepted by Circuit.add_gate that resolve to a canonical gate class.
PYZX_ALIASES = {"CP": "CPhase", "H": "HAD", "U": "U3", "TOF": "Tof"}

# Canonical gates already exercised by the single/multi-qubit fixtures above.
PYZX_FIXTURE_GATES = {
    "NOT",
    "Y",
    "Z",
    "HAD",
    "XPhase",
    "YPhase",
    "ZPhase",
    "U2",
    "U3",
    "SX",
    "S",
    "T",
    "CNOT",
    "CY",
    "CZ",
    "CRX",
    "CRY",
    "CRZ",
    "CPhase",
    "ParityPhase",
    "XCX",
    "CSX",
    "SWAP",
    "CSWAP",
    "CHAD",
    "Tof",
    "CCZ",
    "CU3",
    "CU",
    "RZZ",
    "RXX",
    "Measurement",
    "Reset",
}


# (gate class, positional args, keyword args) for each canonical PyZX gate.
PYZX_GATE_SPECS = {
    "XPhase": (pyzx_gates.XPhase, (0, Fraction(1, 4)), {}),
    "NOT": (pyzx_gates.NOT, (0,), {}),
    "SX": (pyzx_gates.SX, (0,), {"adjoint": False}),
    "CSX": (pyzx_gates.CSX, (0, 1), {}),
    "YPhase": (pyzx_gates.YPhase, (0, Fraction(1, 3)), {}),
    "Y": (pyzx_gates.Y, (0,), {}),
    "CY": (pyzx_gates.CY, (0, 1), {}),
    "ZPhase": (pyzx_gates.ZPhase, (0, Fraction(1, 5)), {}),
    "CPhase": (pyzx_gates.CPhase, (0, 1, Fraction(1, 5)), {}),
    "Z": (pyzx_gates.Z, (0,), {}),
    "S": (pyzx_gates.S, (0,), {"adjoint": False}),
    "T": (pyzx_gates.T, (0,), {"adjoint": False}),
    "CNOT": (pyzx_gates.CNOT, (0, 1), {}),
    "CZ": (pyzx_gates.CZ, (0, 1), {}),
    "ParityPhase": (pyzx_gates.ParityPhase, (Fraction(1, 2), 0, 1, 2), {}),
    "PhaseGadget": (pyzx_gates.PhaseGadget, (Fraction(1, 2), 0, 1, 2), {}),
    "XCX": (pyzx_gates.XCX, (0, 1), {}),
    "SWAP": (pyzx_gates.SWAP, (0, 1), {}),
    "CSWAP": (pyzx_gates.CSWAP, (0, 1, 2), {}),
    "CRZ": (pyzx_gates.CRZ, (0, 1, Fraction(1, 5)), {}),
    "HAD": (pyzx_gates.HAD, (0,), {}),
    "CHAD": (pyzx_gates.CHAD, (0, 1), {}),
    "Tof": (pyzx_gates.Tofolli, (0, 1, 2), {}),
    "CCZ": (pyzx_gates.CCZ, (0, 1, 2), {}),
    "U2": (pyzx_gates.U2, (0, Fraction(2, 7), Fraction(9, 8)), {}),
    "U3": (pyzx_gates.U3, (0, Fraction(6, 7), Fraction(3, 2), Fraction(5, 4)), {}),
    "CU3": (pyzx_gates.CU3, (0, 1, Fraction(6, 5), Fraction(3, 2), Fraction(7, 4)), {}),
    "CU": (pyzx_gates.CU, (1, 3, Fraction(7, 9), Fraction(6, 5), Fraction(3, 2), Fraction(7, 4)), {}),
    "CRX": (pyzx_gates.CRX, (0, 2, Fraction(2, 3)), {}),
    "CRY": (pyzx_gates.CRY, (0, 2, Fraction(4, 3)), {}),
    "RZZ": (pyzx_gates.RZZ, (0, 2, Fraction(2, 9)), {}),
    "RXX": (pyzx_gates.RXX, (0, 2, Fraction(7, 10)), {}),
    "FSim": (pyzx_gates.FSim, (0, 1, Fraction(1, 2), Fraction(1, 3)), {}),
    "InitAncilla": (pyzx_gates.InitAncilla, (0,), {}),
    "Reset": (pyzx_gates.Reset, (1,), {}),
    "PostSelect": (pyzx_gates.PostSelect, (0,), {}),
    "DiscardBit": (pyzx_gates.DiscardBit, (0,), {}),
    "Measurement": (pyzx_gates.Measurement, (0,), {"result_bit": 0}),
    "ConditionalGate": (
        pyzx_gates.ConditionalGate,
        ("c", 1, pyzx_gates.ZPhase(2, Fraction(1, 2)), 1),
        {},
    ),
}


def pyzx_gate_instance(name):
    """Return a fresh instance of the named PyZX gate."""
    gate_cls, args, kwargs = PYZX_GATE_SPECS[name]
    return gate_cls(*args, **kwargs)


# PyZX gates that legitimately have no Qrisp equivalent and no decomposition.
PYZX_UNSUPPORTED = {"FSim", "InitAncilla", "PostSelect", "DiscardBit", "ConditionalGate"}

# Converter gaps that are known to raise today but should convert.
PYZX_KNOWN_GAPS = {"PhaseGadget"}


def pyzx_gap_params():
    """Parametrize PyZX gates not covered by the pre-existing fixtures."""
    params = []
    for name in sorted(PYZX_GATE_SPECS):
        if name in PYZX_FIXTURE_GATES:
            continue
        marks = []
        if name in PYZX_KNOWN_GAPS:
            marks.append(
                pytest.mark.xfail(
                    reason="known bug: PhaseGadget is not un-gadgetted before decomposition",
                    strict=False,
                )
            )
        params.append(pytest.param(name, name in PYZX_UNSUPPORTED, id=name, marks=marks))
    return params


@pytest.mark.parametrize("key", sorted(pyzx_gates.gate_types))
def test_pyzx_gate_key_smoke(key):
    """Every PyZX gate key either converts or raises a clear ValueError."""
    canonical = PYZX_ALIASES.get(key, key)
    c = Circuit(4, bit_amount=1)
    c.add_gate(pyzx_gate_instance(canonical))
    try:
        result = QuantumCircuit.from_pyzx(c)
    except ValueError:
        return
    assert isinstance(result, QuantumCircuit)


@pytest.mark.parametrize("name, expect_raise", pyzx_gap_params())
def test_pyzx_gap_gate(name, expect_raise):
    """PyZX gates absent from the fixtures convert or raise as expected."""
    c = Circuit(4)
    c.add_gate(pyzx_gate_instance(name))
    if expect_raise:
        with pytest.raises(ValueError):
            QuantumCircuit.from_pyzx(c)
    else:
        assert isinstance(QuantumCircuit.from_pyzx(c), QuantumCircuit)


# (number of qubits, QuantumCircuit method, positional args) per Qrisp gate.
# The method ``append`` is forwarded verbatim.
QRISP_GATE_SPECS = {
    "h": (1, "h", (0,)),
    "x": (1, "x", (0,)),
    "y": (1, "y", (0,)),
    "z": (1, "z", (0,)),
    "s": (1, "s", (0,)),
    "s_dg": (1, "s_dg", (0,)),
    "t": (1, "t", (0,)),
    "t_dg": (1, "t_dg", (0,)),
    "sx": (1, "sx", (0,)),
    "sx_dg": (1, "sx_dg", (0,)),
    "id": (1, "id", (0,)),
    "p": (1, "p", (0.3, 0)),
    "u1": (1, "append", (qrisp_ops.U1Gate(0.3), [0])),
    "u3": (1, "u3", (0.3, 0.2, 0.1, 0)),
    "rx": (1, "rx", (0.3, 0)),
    "ry": (1, "ry", (0.3, 0)),
    "rz": (1, "rz", (0.3, 0)),
    "r": (1, "r", (0.3, 0.2, 0)),
    "gphase": (1, "gphase", (0.3, 0)),
    "cx": (2, "cx", (0, 1)),
    "cy": (2, "cy", (0, 1)),
    "cz": (2, "cz", (0, 1)),
    "swap": (2, "swap", (0, 1)),
    "cp": (2, "cp", (0.3, 0, 1)),
    "rxx": (2, "rxx", (0.3, 0, 1)),
    "ryy": (2, "ryy", (0.3, 0, 1)),
    "rzz": (2, "rzz", (0.3, 0, 1)),
    "xxyy": (2, "xxyy", (0.3, 0.2, 0, 1)),
    "ccx": (3, "ccx", (0, 1, 2)),
    "mcx": (3, "mcx", ([0, 1], 2)),
    "mcrx": (3, "append", (qrisp_ops.MCRXGate(0.3, 2), [0, 1, 2])),
    "measure": (1, "measure", (0,)),
    "reset": (1, "reset", (0,)),
    "barrier": (2, "barrier", ([0, 1],)),
}


def qrisp_gate_circuit(name):
    """Build a fresh Qrisp circuit containing the named standard gate."""
    num_qubits, method, args = QRISP_GATE_SPECS[name]
    qc = QuantumCircuit(num_qubits)
    if method == "append":
        qc.append(*args)
    else:
        getattr(qc, method)(*args)
    return qc


# Qrisp standard gates already exercised by the single/multi-qubit fixtures above.
QRISP_FIXTURE_GATES = {
    "h",
    "x",
    "y",
    "z",
    "rx",
    "ry",
    "rz",
    "u3",
    "p",
    "s",
    "t",
    "sx",
    "t_dg",
    "s_dg",
    "sx_dg",
    "gphase",
    "id",
    "cx",
    "cy",
    "cz",
    "swap",
    "xxyy",
    "rxx",
    "rzz",
    "measure",
    "reset",
}

# Qrisp standard gates that are known to raise today but should convert.
QRISP_KNOWN_GAPS = {"u1", "r", "barrier"}


@pytest.mark.parametrize("name", sorted(QRISP_GATE_SPECS))
def test_qrisp_gate_smoke(name):
    """Every Qrisp standard gate converts or raises a clear ValueError."""
    try:
        result = qrisp_gate_circuit(name).to_pyzx()
    except ValueError:
        return
    assert isinstance(result, Circuit)


def qrisp_gap_params():
    """Parametrize Qrisp gates not covered by the pre-existing fixtures."""
    params = []
    for name in sorted(QRISP_GATE_SPECS):
        if name in QRISP_FIXTURE_GATES:
            continue
        if name in QRISP_KNOWN_GAPS:
            params.append(
                pytest.param(
                    name,
                    id=name,
                    marks=pytest.mark.xfail(
                        reason="known gap: gate has no mapping and no definition to transpile",
                        strict=False,
                    ),
                )
            )
        else:
            params.append(pytest.param(name, id=name))
    return params


@pytest.mark.parametrize("name", qrisp_gap_params())
def test_qrisp_gap_gate(name):
    """Qrisp gates absent from the fixtures convert as expected."""
    assert isinstance(qrisp_gate_circuit(name).to_pyzx(), Circuit)


def test_empty_circuit_smoke():
    """Empty circuits convert in both directions without error."""
    c = QuantumCircuit(0).to_pyzx()
    assert isinstance(c, Circuit)
    assert c.qubits == 0 and c.gates == []

    qc = QuantumCircuit.from_pyzx(Circuit(0))
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits() == 0


# Qrisp gap gates that are expected to convert (i.e. not known gaps).
QRISP_GAP_SUPPORTED = sorted(set(QRISP_GATE_SPECS) - QRISP_FIXTURE_GATES - QRISP_KNOWN_GAPS)


@pytest.mark.parametrize("name", QRISP_GAP_SUPPORTED)
def test_qrisp_gap_gate_roundtrip(name):
    """Gap gates that convert survive a Qrisp -> PyZX -> Qrisp round-trip."""
    qc = qrisp_gate_circuit(name)
    qc2 = QuantumCircuit.from_pyzx(qc.to_pyzx())
    assert qc2.num_qubits() == qc.num_qubits()
    _compare_unitaries(qc.get_unitary(), qc2.get_unitary())


@pytest.mark.xfail(reason="known bug: classical destination is dropped on Qrisp -> PyZX", strict=False)
def test_measurement_destination_qrisp_to_pyzx():
    """A measurement's explicit clbit must survive as a PyZX result_bit."""
    qc = QuantumCircuit(2)
    qc.measure(0, qc.add_clbit())
    assert qc.to_pyzx().gates[0].result_bit == 0


@pytest.mark.xfail(reason="known bug: Qrisp -> PyZX measurements have no QASM destination", strict=False)
def test_measurement_qasm_export_qrisp_to_pyzx():
    """A converted measurement must carry the destination required by QASM."""
    qc = QuantumCircuit(2)
    qc.measure(0, qc.add_clbit())
    qc.to_pyzx().to_qasm()


@pytest.mark.xfail(reason="known bug: result_bit is dropped on PyZX -> Qrisp", strict=False)
def test_measurement_destination_pyzx_to_qrisp():
    """PyZX measurements sharing a result_bit must share a Qrisp clbit."""
    c = Circuit(2, bit_amount=2)
    c.add_gate("Measurement", 0, result_bit=1)
    c.add_gate("Measurement", 1, result_bit=1)

    qc = QuantumCircuit.from_pyzx(c)
    assert qc.data[0].clbits == qc.data[1].clbits
