"""Tests for the Qrisp <-> PyZX circuit converter."""

from __future__ import annotations

import sys
from collections.abc import Callable
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pyzx
from pyzx import Circuit
from pyzx.circuit import gates as pyzx_gates

from qrisp import QuantumCircuit, QuantumVariable, U1Gate
from qrisp.circuit import standard_operations as qrisp_ops
from qrisp.interface.converter.pyzx_converter import (
    _check_ctrl_state,
    _check_pyzx_compatibility,
    _import_pyzx,
    _phase,
)


def test_import_error() -> None:
    """to_pyzx raises ImportError when PyZX is not installed."""
    _import_pyzx.cache_clear()
    with pytest.raises(ImportError):
        with patch.dict(sys.modules, {"pyzx": None}):
            qc = QuantumCircuit(1)
            qc.to_pyzx()


@pytest.mark.parametrize(
    "angle, expected",
    [
        (0.0, Fraction(0)),
        (np.pi / 4, Fraction(1, 4)),
        (np.pi / 3, Fraction(1, 3)),
        (2 * np.pi / 3, Fraction(2, 3)),
        (-np.pi / 2, Fraction(-1, 2)),
        (np.pi, Fraction(1)),
    ],
)
def test_phase_snaps_to_small_denominators(angle: float, expected: Fraction) -> None:
    """Common angles are snapped to small rational phases (in units of pi)."""
    assert _phase(angle) == expected


def test_phase_keeps_non_rational_angles_exact() -> None:
    """Angles that are not close to a small rational are not snapped."""
    max_denominator = 1024
    phase = _phase(0.3)
    assert float(phase) == 0.3 / np.pi
    assert phase.denominator > max_denominator


@pytest.mark.parametrize(
    "method, args, expected",
    [
        ("p", (np.pi / 3, 0), Fraction(1, 3)),
        ("s_dg", (0,), Fraction(-1, 2)),
        ("t_dg", (0,), Fraction(-1, 4)),
    ],
)
def test_convert_to_pyzx_zphase_gates(method: str, args: tuple, expected: Fraction) -> None:
    """Phase gates are emitted as snapped PyZX ZPhase gates."""
    qc = QuantumCircuit(1)
    getattr(qc, method)(*args)

    gate = qc.to_pyzx().gates[0]

    assert gate.name == "ZPhase"
    assert gate.phase == expected


def test_prx_maps_to_u3_decomposition() -> None:
    """PRX is converted with the correct three-parameter U3 decomposition."""
    theta, phi = 0.3, 0.2
    qc = QuantumCircuit(1)
    qc.prx(theta, phi, 0)

    gate = qc.to_pyzx().gates[0]

    assert gate.name == "U3"
    expected = [theta / np.pi, (phi - np.pi / 2) / np.pi, (np.pi / 2 - phi) / np.pi]
    np.testing.assert_allclose([float(gate.theta), float(gate.phi), float(gate.rho)], expected)


@pytest.mark.parametrize(
    "pyzx_module, gate_types, expect_error",
    [
        (pyzx, pyzx_gates.gate_types, False),
        (SimpleNamespace(__version__="0.9.0"), {}, True),
    ],
    ids=["current_version", "too_old"],
)
def test_pyzx_compatibility_check(
    pyzx_module: object,
    gate_types: dict[str, object],
    expect_error: bool,
) -> None:
    """The compatibility check accepts supported PyZX and rejects old versions."""
    if expect_error:
        with pytest.raises(ImportError, match="PyZX >= 0.10.1"):
            _check_pyzx_compatibility(pyzx_module, Circuit, gate_types)
    else:
        _check_pyzx_compatibility(pyzx_module, Circuit, gate_types)


def test_non_default_ctrl_state_raises() -> None:
    """Controlled operations that trigger on |0> are rejected, not silently converted."""
    qc = QuantumCircuit(2)
    qc.append(qrisp_ops.XGate().control(ctrl_state=0), [0, 1])

    with pytest.raises(ValueError, match="ctrl_state"):
        qc.to_pyzx()


@pytest.mark.parametrize(
    "ctrl_state, expect_error",
    [
        (3, False),
        (2, True),
        ("11", False),
        ("10", True),
        ("1", True),
    ],
    ids=[
        "integer_all_ones",
        "integer_mixed",
        "string_all_ones",
        "string_mixed",
        "string_length_mismatch",
    ],
)
def test_check_ctrl_state(ctrl_state: int | str, expect_error: bool) -> None:
    """Integer and string control states are validated against the number of controls."""
    op = SimpleNamespace(name="cx", ctrl_state=ctrl_state, controls=[object(), object()])

    if expect_error:
        with pytest.raises(ValueError, match="ctrl_state"):
            _check_ctrl_state(op)
    else:
        _check_ctrl_state(op)


def _convert_symbolic_qrisp_circuit() -> None:
    """Convert a Qrisp circuit with an unresolved symbolic parameter."""
    from sympy import symbols

    theta = symbols("theta")
    qc = QuantumCircuit(1)
    qc.p(theta, 0)
    qc.to_pyzx()


def _convert_symbolic_pyzx_circuit() -> None:
    """Convert a PyZX circuit with an unresolved symbolic phase."""
    from pyzx.symbolic import new_var

    c = Circuit(1)
    c.add_gate("ZPhase", 0, new_var("theta", False))
    QuantumCircuit.from_pyzx(c)


@pytest.mark.parametrize(
    "convert",
    [_convert_symbolic_qrisp_circuit, _convert_symbolic_pyzx_circuit],
    ids=["qrisp_to_pyzx", "pyzx_to_qrisp"],
)
def test_symbolic_parameter_raises(convert: Callable[[], None]) -> None:
    """Symbolic parameters produce a clear error instead of a TypeError."""
    with pytest.raises(ValueError, match="Bind symbolic parameters"):
        convert()


def _build_single_qubit_qrisp_circuit() -> QuantumCircuit:
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


def _build_multi_qubit_qrisp_circuit() -> QuantumCircuit:
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


def _build_single_qubit_pyzx_circuit() -> Circuit:
    """PyZX circuit with single-qubit gates."""
    c = Circuit(4)
    c.add_gate("NOT", 0)
    c.add_gate("Y", 1)
    c.add_gate("Z", 2)
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


def _build_multi_qubit_pyzx_circuit() -> Circuit:
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


def _compare_unitaries(U1: np.ndarray, U2: np.ndarray) -> None:
    """Compare two unitaries, taking into account a potential phase mismatch"""
    # PyZX does not keep track of global phases and instead normalizes the phase,
    # which leads to a potential phase mismatch.
    # We determine the phase difference by dividing the matrix entries with the largest magnitude
    ind_max = np.unravel_index(np.argmax(np.abs(U1)), U1.shape)
    phase = U2[ind_max] / U1[ind_max]

    np.testing.assert_array_almost_equal(phase * U1, U2)


def _test_qrisp_to_pyzx(qc: QuantumCircuit) -> None:
    """Test conversion of qrisp circuit qc to pyzx circuit"""
    c = qc.to_pyzx()
    _compare_unitaries(qc.get_unitary(), c.to_matrix())


def _test_pyzx_to_qrisp(c: Circuit) -> None:
    """Test conversion of pyzx circuit c to qrisp circuit"""
    qc = QuantumCircuit.from_pyzx(c)
    _compare_unitaries(qc.get_unitary(), c.to_matrix())


def _test_roundtrip(qc: QuantumCircuit) -> None:
    """Test roundtrip starting from qrisp circuit qc"""
    c = qc.to_pyzx()
    qc2 = QuantumCircuit.from_pyzx(c)
    _compare_unitaries(qc.get_unitary(), qc2.get_unitary())


def _test_roundtrip_reverse(c: Circuit) -> None:
    """Test roundtrip starting from pyzx circuit c"""
    qc = QuantumCircuit.from_pyzx(c)
    c2 = qc.to_pyzx()
    _compare_unitaries(c.to_matrix(), c2.to_matrix())


QRISP_CIRCUIT_BUILDERS = [
    pytest.param(_build_single_qubit_qrisp_circuit, id="single_qubit"),
    pytest.param(_build_multi_qubit_qrisp_circuit, id="multi_qubit"),
]
PYZX_CIRCUIT_BUILDERS = [
    pytest.param(_build_single_qubit_pyzx_circuit, id="single_qubit"),
    pytest.param(_build_multi_qubit_pyzx_circuit, id="multi_qubit"),
]


@pytest.mark.parametrize("builder", QRISP_CIRCUIT_BUILDERS)
def test_circuit_qrisp_to_pyzx(builder: Callable[[], QuantumCircuit]) -> None:
    """Test conversion of qrisp circuits to pyzx circuits."""
    _test_qrisp_to_pyzx(builder())


@pytest.mark.parametrize("builder", PYZX_CIRCUIT_BUILDERS)
def test_circuit_pyzx_to_qrisp(builder: Callable[[], Circuit]) -> None:
    """Test conversion of pyzx circuits to qrisp circuits."""
    _test_pyzx_to_qrisp(builder())


@pytest.mark.parametrize("builder", QRISP_CIRCUIT_BUILDERS)
def test_circuit_roundtrip(builder: Callable[[], QuantumCircuit]) -> None:
    """Test roundtrip starting from qrisp circuits."""
    _test_roundtrip(builder())


@pytest.mark.parametrize("builder", PYZX_CIRCUIT_BUILDERS)
def test_circuit_roundtrip_reverse(builder: Callable[[], Circuit]) -> None:
    """Test roundtrip starting from pyzx circuits."""
    _test_roundtrip_reverse(builder())


def test_qrisp_transpilation() -> None:
    """Test transpilation for a circuit that has to be decomposed first.

    Example adapted from test_cirq_converter.py.
    """
    from qrisp import QPE, h, p

    def U(qv: QuantumVariable) -> None:
        """Apply phase gates with non-trivial angles to ``qv``."""
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


def test_convert_from_pyzx_does_not_mutate_input() -> None:
    """Converting a phase gadget must not change the source PyZX circuit."""
    c = Circuit(3)
    c.add_gate("PhaseGadget", Fraction(1, 2), 0, 1, 2)

    QuantumCircuit.from_pyzx(c)

    assert c.gates[0].as_gadget is True


def test_measurement_without_result_bit_allocates_clbit() -> None:
    """A measurement with no PyZX destination gets a fresh classical bit."""
    c = Circuit(1)
    c.add_gate("Measurement", 0)

    qc = QuantumCircuit.from_pyzx(c)

    assert len(qc.clbits) == 1
    assert qc.data[0].clbits == [qc.clbits[0]]


def test_measurement_result_bit_out_of_range_raises() -> None:
    """An explicit but undeclared measurement destination raises a clear error."""
    c = Circuit(1, bit_amount=1)
    c.add_gate("Measurement", 0, result_bit=1)

    with pytest.raises(ValueError, match="targets result bit 1"):
        QuantumCircuit.from_pyzx(c)


def test_non_unitaries() -> None:
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


def _mock_unknown_circuit() -> QuantumCircuit:
    """Qrisp circuit with a genuinely unknown mock operation (triggers an error during transpile)."""
    op = MagicMock()
    op.name = "some_gate"
    op.params = []

    qc = QuantumCircuit(1)
    qc.data = [MagicMock(op=op, qubits=[MagicMock()])]
    return qc


def _real_unknown_circuit() -> QuantumCircuit:
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
        (_real_unknown_circuit, "could not be decomposed into elementary"),
    ],
    ids=["unknown_gate", "real_unknown_no_progress"],
)
def test_convert_to_pyzx_raises(circuit_builder: Callable[[], QuantumCircuit], match: str) -> None:
    """Verify convert_to_pyzx raises ValueError for unsupported gates."""
    with pytest.raises(ValueError, match=match):
        qc = circuit_builder()
        qc.to_pyzx()


def _build_pyzx_circuit_with_mock_gate() -> Circuit:
    """PyZX circuit with an unknown mock gate."""
    gate = MagicMock()
    gate.name = "some_gate"

    c = Circuit(1)
    c.add_gate(gate)
    return c


def _build_pyzx_circuit_with_undecomposable_gate() -> Circuit:
    """PyZX circuit with a gate whose decomposition still contains the same gate."""
    c = Circuit(2)
    from pyzx.circuit import ParityPhase

    g = ParityPhase(0, 0, 1)

    def _f() -> list[pyzx_gates.Gate]:
        """Return a decomposition that still contains the undecomposable gate."""
        return [ParityPhase(0, 0, 1)]

    g.to_basic_gates = _f
    c.add_gate(g)
    return c


@pytest.mark.parametrize(
    "circuit_builder, match",
    [
        (_build_pyzx_circuit_with_mock_gate, "of PyZX is unknown"),
        (_build_pyzx_circuit_with_undecomposable_gate, "cannot be decomposed either"),
    ],
    ids=["unknown_gate", "undecomposable_gate"],
)
def test_convert_from_pyzx_raises(circuit_builder: Callable[[], Circuit], match: str) -> None:
    """Verify convert_from_pyzx raises ValueError for unsupported gates."""
    with pytest.raises(ValueError, match=match):
        QuantumCircuit.from_pyzx(circuit_builder())


# ---------------------------------------------------------------------------
# Smoke and gap tests
#
# The tests above verify correctness (unitary equivalence, round-trips) for a
# representative set of gates. These tests add complementary breadth:
#
#   * ``*_smoke`` checks the expected outcome for every gate key: supported
#     gates convert and explicitly unsupported gates raise a ``ValueError``.
#   * ``*_gap_gate`` adds unitary checks for gates the fixtures above do not
#     cover.
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


def pyzx_gate_instance(name: str) -> pyzx_gates.Gate:
    """Return a fresh instance of the named PyZX gate."""
    gate_cls, args, kwargs = PYZX_GATE_SPECS[name]
    return gate_cls(*args, **kwargs)


# PyZX gates that legitimately have no Qrisp equivalent and no decomposition.
PYZX_UNSUPPORTED = {"FSim", "InitAncilla", "PostSelect", "DiscardBit", "ConditionalGate"}


def pyzx_gap_params() -> list:
    """Parametrize PyZX gates not covered by the pre-existing fixtures."""
    return [
        pytest.param(name, name in PYZX_UNSUPPORTED, id=name)
        for name in sorted(PYZX_GATE_SPECS)
        if name not in PYZX_FIXTURE_GATES
    ]


@pytest.mark.parametrize("key", sorted(pyzx_gates.gate_types))
def test_pyzx_gate_key_smoke(key: str) -> None:
    """Every PyZX gate key either converts or is explicitly unsupported."""
    canonical = PYZX_ALIASES.get(key, key)
    if canonical not in PYZX_GATE_SPECS:
        pytest.fail(f"New PyZX gate {key!r}: add it to PYZX_GATE_SPECS (and PYZX_UNSUPPORTED if needed).")

    c = Circuit(4, bit_amount=1)
    c.add_gate(pyzx_gate_instance(canonical))

    if canonical in PYZX_UNSUPPORTED:
        with pytest.raises(ValueError):
            QuantumCircuit.from_pyzx(c)
    else:
        assert isinstance(QuantumCircuit.from_pyzx(c), QuantumCircuit)


@pytest.mark.parametrize("name, expect_raise", pyzx_gap_params())
def test_pyzx_gap_gate(name: str, expect_raise: bool) -> None:
    """PyZX gates absent from the fixtures convert unitarily or raise as expected."""
    c = Circuit(4)
    c.add_gate(pyzx_gate_instance(name))
    if expect_raise:
        with pytest.raises(ValueError):
            QuantumCircuit.from_pyzx(c)
    else:
        _test_pyzx_to_qrisp(c)


@pytest.mark.parametrize("name", ["CRX", "CRY", "CRZ", "CPhase", "RXX", "RZZ", "CU3"])
def test_pyzx_two_qubit_parametric_gate_roundtrip(name: str) -> None:
    """PyZX two-qubit parametric gates round-trip without attribute errors."""
    c = Circuit(4)
    c.add_gate(pyzx_gate_instance(name))
    _test_roundtrip_reverse(c)


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
    "prx": (1, "prx", (0.3, 0.2, 0)),
    "r": (1, "r", (0.3, 0.2, 0)),
    "gphase": (1, "gphase", (0.3, 0)),
    "cx": (2, "cx", (0, 1)),
    "cy": (2, "cy", (0, 1)),
    "cz": (2, "cz", (0, 1)),
    "swap": (2, "swap", (0, 1)),
    "cp": (2, "cp", (0.3, 0, 1)),
    "crx": (2, "crx", (0.3, 0, 1)),
    "cry": (2, "append", (qrisp_ops.RYGate(0.3).control(), [0, 1])),
    "crz": (2, "append", (qrisp_ops.RZGate(0.3).control(), [0, 1])),
    "cu3": (2, "append", (qrisp_ops.u3Gate(0.3, 0.2, 0.1).control(), [0, 1])),
    "ch": (2, "append", (qrisp_ops.HGate().control(), [0, 1])),
    "rxx": (2, "rxx", (0.3, 0, 1)),
    "ryy": (2, "ryy", (0.3, 0, 1)),
    "rzz": (2, "rzz", (0.3, 0, 1)),
    "xxyy": (2, "xxyy", (0.3, 0.2, 0, 1)),
    "ccx": (3, "ccx", (0, 1, 2)),
    "ccz": (3, "append", (qrisp_ops.ZGate().control(2), [0, 1, 2])),
    "mcx": (3, "mcx", ([0, 1], 2)),
    "mcrx": (3, "append", (qrisp_ops.MCRXGate(0.3, 2), [0, 1, 2])),
    "measure": (1, "measure", (0,)),
    "reset": (1, "reset", (0,)),
    "barrier": (2, "barrier", ([0, 1],)),
}


def qrisp_gate_circuit(name: str) -> QuantumCircuit:
    """Build a fresh Qrisp circuit containing the named standard gate."""
    num_qubits, method, args = QRISP_GATE_SPECS[name]
    qc = QuantumCircuit(num_qubits)
    if method == "append":
        qc.append(*args)
    else:
        getattr(qc, method)(*args)
    return qc


@pytest.mark.parametrize(
    "name, expected",
    [("ccx", "Tof"), ("cp", "CPhase"), ("ccz", "CCZ"), ("ch", "CHAD")],
)
def test_native_pyzx_gate_mapping(name: str, expected: str) -> None:
    """Qrisp gates with a native PyZX equivalent map directly, without decomposition."""
    gates = qrisp_gate_circuit(name).to_pyzx().gates
    assert [gate.name for gate in gates] == [expected]


def test_phase_tolerant_toffoli_has_distinct_name() -> None:
    """Phase-tolerant Toffoli variants are not confused with the native TOF mapping."""
    qc = QuantumCircuit(3)
    qc.mcx([0, 1], 2, method="gray_pt")

    assert qc.data[0].op.name == "pt2cx"
    assert "Tof" not in [gate.name for gate in qc.to_pyzx().gates]

    _test_qrisp_to_pyzx(qc)


CONTROLLED_PARAMETRIC_GATES = ["crx", "cry", "crz", "cp", "cu3", "rxx", "rzz"]


@pytest.mark.parametrize("name", CONTROLLED_PARAMETRIC_GATES)
def test_qrisp_controlled_parametric_gate_roundtrip_phase_is_global(name: str) -> None:
    """A Qrisp controlled round-trip may differ by a global phase only.

    These Qrisp gates are transpiled before conversion, so this test covers that
    path rather than the direct PyZX CRX/CRY/CRZ/CU3 handlers (which are covered
    by ``test_pyzx_two_qubit_parametric_gate_roundtrip``). PyZX intentionally
    drops global phases, so exact equality is not preserved; wrapping the gate
    under control would expose any relative phase error.
    """
    qc = qrisp_gate_circuit(name)
    qc_roundtrip = QuantumCircuit.from_pyzx(qc.to_pyzx())

    U1 = qc.get_unitary()
    U2 = qc_roundtrip.get_unitary()
    residual = U2 @ U1.conj().T
    phase = residual[0, 0]

    np.testing.assert_array_almost_equal(residual, phase * np.eye(U1.shape[0]))


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


# Qrisp standard gates that legitimately cannot be converted.
QRISP_UNSUPPORTED: set[str] = set()


@pytest.mark.parametrize("name", sorted(QRISP_GATE_SPECS))
def test_qrisp_gate_smoke(name: str) -> None:
    """Every Qrisp standard gate either converts or is explicitly unsupported."""
    qc = qrisp_gate_circuit(name)

    if name in QRISP_UNSUPPORTED:
        with pytest.raises(ValueError):
            qc.to_pyzx()
    else:
        assert isinstance(qc.to_pyzx(), Circuit)


def test_empty_circuit_smoke() -> None:
    """Empty circuits convert in both directions without error."""
    c = QuantumCircuit(0).to_pyzx()
    assert isinstance(c, Circuit)
    assert c.qubits == 0 and c.gates == []

    qc = QuantumCircuit.from_pyzx(Circuit(0))
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits() == 0


# Qrisp gap gates that are expected to convert.
QRISP_GAP_SUPPORTED = sorted(set(QRISP_GATE_SPECS) - QRISP_FIXTURE_GATES)


@pytest.mark.parametrize("name", QRISP_GAP_SUPPORTED)
def test_qrisp_gap_gate_roundtrip(name: str) -> None:
    """Gap gates survive both Qrisp -> PyZX and a full round-trip."""
    qc = qrisp_gate_circuit(name)

    _test_qrisp_to_pyzx(qc)

    qc2 = QuantumCircuit.from_pyzx(qc.to_pyzx())
    assert qc2.num_qubits() == qc.num_qubits()
    _compare_unitaries(qc.get_unitary(), qc2.get_unitary())


def test_measurement_destination_qrisp_to_pyzx() -> None:
    """A measurement's explicit clbit survives as a PyZX result bit and in QASM."""
    qc = QuantumCircuit(2)
    qc.measure(0, qc.add_clbit())

    pyzx_circuit = qc.to_pyzx()

    assert pyzx_circuit.gates[0].result_bit == 0
    assert "measure" in pyzx_circuit.to_qasm()


def test_measurement_destination_pyzx_to_qrisp() -> None:
    """PyZX measurements sharing a result_bit must share a Qrisp clbit."""
    c = Circuit(2, bit_amount=2)
    c.add_gate("Measurement", 0, result_bit=1)
    c.add_gate("Measurement", 1, result_bit=1)

    qc = QuantumCircuit.from_pyzx(c)
    assert qc.data[0].clbits == qc.data[1].clbits
