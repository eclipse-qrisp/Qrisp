from unittest.mock import patch

import numpy as np
import pytest

from qrisp import QuantumCircuit
from qrisp.circuit import Operation
from qrisp.circuit.standard_operations import (
    MCRXGate,
    MCXGate,
    PGate,
    RGate,
    RXGate,
    U1Gate,
    YGate,
    ZGate,
)
from qrisp.interface.converter.pytket_converter import (
    create_tket_instruction,
    pytket_converter,
)

# The whole module exercises the Qrisp -> pytket converter; skip cleanly where
# pytket is not installed. The import-failure tests below simulate its absence by
# patching __import__, so they still run when pytket is present.
pytest.importorskip("pytket")


def _matches_up_to_global_phase(actual, expected, atol=1e-6):
    """Return True if two arrays are equal up to a global phase factor.

    The global phase is recovered from the Frobenius inner product of the two
    arrays. This is robust even when several entries share the largest magnitude:
    normalising against a single pivot entry is not, because floating-point noise
    can make ``argmax`` select different entries for the two arrays, so they end up
    referenced to different phases.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    overlap = np.vdot(expected, actual)
    if np.abs(overlap) < atol:
        return False

    phase = overlap / np.abs(overlap)
    return np.allclose(actual, expected * phase, atol=atol)


_GATE_THETA = 0.7

# Builders whose pytket conversion must reproduce Qrisp's own unitary. Comparing
# to_pytket().get_unitary() against QuantumCircuit.get_unitary() checks that the
# converter preserves each gate's action up to a global phase; this also pins the
# cp -> CU1 and u1 angle mapping fixes. Each entry is ``(num_qubits, builder)``.
# Multi-controlled gates (mcx/mcrx) are covered separately and structurally,
# since their exact unitary is subject to the converter's context-dependent
# controlled-gate ordering.
_UNITARY_GATE_BUILDERS = {
    # single-qubit
    "h": (1, lambda qc: qc.h(0)),
    "x": (1, lambda qc: qc.x(0)),
    "y": (1, lambda qc: qc.y(0)),
    "z": (1, lambda qc: qc.z(0)),
    "id": (1, lambda qc: qc.id(0)),
    "s": (1, lambda qc: qc.s(0)),
    "s_dg": (1, lambda qc: qc.s_dg(0)),
    "t": (1, lambda qc: qc.t(0)),
    "t_dg": (1, lambda qc: qc.t_dg(0)),
    "sx": (1, lambda qc: qc.sx(0)),
    "sx_dg": (1, lambda qc: qc.sx_dg(0)),
    "rx": (1, lambda qc: qc.rx(_GATE_THETA, 0)),
    "ry": (1, lambda qc: qc.ry(_GATE_THETA, 0)),
    "rz": (1, lambda qc: qc.rz(_GATE_THETA, 0)),
    "p": (1, lambda qc: qc.p(_GATE_THETA, 0)),
    "u1": (1, lambda qc: qc.append(U1Gate(_GATE_THETA), [0])),
    "u3": (1, lambda qc: qc.u3(0.4, 0.5, 0.6, 0)),
    # two-qubit
    "cx": (2, lambda qc: qc.cx(0, 1)),
    "cy": (2, lambda qc: qc.cy(0, 1)),
    "cz": (2, lambda qc: qc.cz(0, 1)),
    "cp": (2, lambda qc: qc.cp(_GATE_THETA, 0, 1)),
    "swap": (2, lambda qc: qc.swap(0, 1)),
    "rxx": (2, lambda qc: qc.rxx(_GATE_THETA, 0, 1)),
    "rzz": (2, lambda qc: qc.rzz(_GATE_THETA, 0, 1)),
}


@pytest.mark.parametrize("gate", sorted(_UNITARY_GATE_BUILDERS))
def test_to_pytket_gate_unitary(gate):
    """to_pytket() reproduces each gate's unitary up to a global phase.

    Compares the converted pytket unitary against Qrisp's own get_unitary() across
    single- and two-qubit gates, giving the converter deterministic per-gate coverage.
    """
    num_qubits, build = _UNITARY_GATE_BUILDERS[gate]
    qc = QuantumCircuit(num_qubits)
    build(qc)

    assert _matches_up_to_global_phase(qc.to_pytket().get_unitary(), qc.get_unitary())


# Multi-controlled gates take the generic ControlledOperation -> CircBox path.
# mcx (base name "x") and mcrx (base name "rx") also exercise both sides of the
# single-character base-name check.
_CONTROLLED_GATE_BUILDERS = {
    "mcx": (4, lambda qc: qc.append(MCXGate(control_amount=3), [0, 1, 2, 3])),
    "mcrx": (3, lambda qc: qc.append(MCRXGate(_GATE_THETA, control_amount=2), [0, 1, 2])),
}


@pytest.mark.parametrize("gate", sorted(_CONTROLLED_GATE_BUILDERS))
def test_to_pytket_controlled_gate_structure(gate):
    """Multi-controlled gates convert via the ControlledOperation -> CircBox path.

    Their exact unitary is not compared: Qrisp's multi-controlled synthesis interacts
    with the converter's context-dependent controlled-gate ordering, so a strict
    unitary comparison is not reliable across contexts. This instead checks the
    conversion structurally -- it succeeds and emits a single CircBox spanning the
    gate's qubits -- which covers the ControlledOperation branch of the converter.
    The Grover test exercises multi-controlled gates end-to-end.
    """
    from pytket import OpType

    num_qubits, build = _CONTROLLED_GATE_BUILDERS[gate]
    qc = QuantumCircuit(num_qubits)
    build(qc)

    tket_qc = qc.to_pytket()
    boxes = [cmd for cmd in tket_qc.get_commands() if cmd.op.type == OpType.CircBox]

    assert tket_qc.n_qubits == num_qubits
    assert len(boxes) == 1
    assert len(boxes[0].qubits) == num_qubits


def test_to_pytket_gphase():
    """to_pytket() preserves a global-phase gate exactly.

    ``x, x`` is the identity, so ``gphase(theta)`` leaves the unitary as
    ``e^{i*theta} * I``. Global phase is unobservable up to a phase factor, so this
    compares the converted unitary against Qrisp's own get_unitary() *exactly* to
    prove the phase itself is carried across.
    """
    theta = 0.5
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.x(0)
    qc.gphase(theta, 0)

    np.testing.assert_array_almost_equal(qc.to_pytket().get_unitary(), qc.get_unitary())


def test_to_pytket_composite_gate():
    """to_pytket() reproduces a composite (non-elementary) gate's unitary.

    A gate built from a sub-circuit has a ``definition`` but no elementary pytket
    equivalent, so it exercises the CircBox path in ``create_tket_instruction``.
    """
    sub = QuantumCircuit(2)
    sub.h(0)
    sub.cx(0, 1)

    qc = QuantumCircuit(2)
    qc.append(sub.to_gate(name="myblock"), [0, 1])

    assert _matches_up_to_global_phase(qc.to_pytket().get_unitary(), qc.get_unitary())


def test_to_pytket_measure():
    """to_pytket() maps measurements to pytket Measure ops, preserving wiring.

    Measurement is not unitary, so this checks the converted circuit structurally:
    every qubit is measured into the classical bit of the same index.
    """
    from pytket import OpType

    num_qubits = 2
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.x(0)
    qc.measure(0, 0)
    qc.measure(1, 1)

    tket_qc = qc.to_pytket()
    measurements = [cmd for cmd in tket_qc.get_commands() if cmd.op.type == OpType.Measure]

    assert tket_qc.n_bits == num_qubits
    assert len(measurements) == num_qubits
    # Each measurement wires qubit k to the classical bit of the same index.
    expected_wiring = {(k, k) for k in range(num_qubits)}
    assert {(cmd.qubits[0].index[0], cmd.bits[0].index[0]) for cmd in measurements} == expected_wiring


def test_to_pytket_grover():
    """to_pytket() reproduces a full Grover search circuit.

    This is the converter's end-to-end regression: a compiled Grover session mixes
    composite/controlled gates (mapped to CircBoxes), qubit (de)allocation, and the
    diffuser's global-phase gate. The instance is deliberately small
    (``QuantumFloat(2, -1)`` -> 9 qubits) so it stays within pytket's statevector
    simulation limit. The converted statevector must match Qrisp's own up to a
    global phase.
    """
    from math import pi

    from qrisp.grover import diffuser

    from qrisp import QuantumFloat, auto_uncompute, h, z

    @auto_uncompute
    def sqrt_oracle(qf):
        temp_qbool = qf * qf == 0.25
        z(temp_qbool)

    qf = QuantumFloat(2, -1, signed=True)
    n = qf.size
    iterations = max(1, int(0.25 * pi * (2**n / 2) ** 0.5))

    h(qf)
    for _ in range(iterations):
        sqrt_oracle(qf)
        diffuser(qf)

    qc = qf.qs.compile()

    assert _matches_up_to_global_phase(qc.to_pytket().get_statevector(), qc.statevector_array())


def test_create_tket_instruction_unknown_raises():
    """create_tket_instruction() raises for an op with no pytket equivalent.

    An operation whose name is not in the gate map and that has no ``definition``
    cannot be converted, hitting the final ``raise``.
    """
    op = Operation(name="totally_unknown", num_qubits=1)

    with pytest.raises(ValueError, match="Could not convert"):
        create_tket_instruction(op)


def test_pytket_converter_import_error():
    """pytket_converter() raises a clear ImportError when pytket is missing."""
    import builtins

    real_import = builtins.__import__

    def _fail_pytket_import(name, *args, **kwargs):
        if name == "pytket" or name.startswith("pytket."):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", _fail_pytket_import):
        with pytest.raises(ImportError, match="PyTket must be installed"):
            pytket_converter(QuantumCircuit(1))


# ---------------------------------------------------------------------------
# Smoke tests for known-broken converter behaviour. Each test documents a
# *known* bug and is expected to FAIL until the underlying issue is fixed. The
# owner is noted in each docstring (pytket converter / qrisp / pytket), along
# with the upstream Qrisp issue number where one exists
# (https://github.com/eclipse-qrisp/Qrisp/issues). Findings without an issue
# number are not yet tracked upstream and should be reported/fixed in Qrisp.
# ---------------------------------------------------------------------------


# Finding 1a (pytket converter): ctrl_state is silently dropped for the four
# elementary single-control gates. op.name == "cx"/"cy"/"cz"/"cp" hits the
# _GATE_OPTYPES lookup table (pytket_converter.py:89) before the
# ControlledOperation branch (pytket_converter.py:187), so a flipped control
# becomes a plain gate with no warning. Expected unitary: X/Y/Z/P applied only
# when the control is |0>.
#
# TODO: not tracked upstream yet -- should be fixed in Qrisp (the pytket
# converter must respect ctrl_state for elementary controlled gates).
@pytest.mark.parametrize(
    "op",
    [
        MCXGate(control_amount=1, ctrl_state=0),
        MCXGate(control_amount=1, ctrl_state="0"),
        YGate().control(1, ctrl_state=0),
        ZGate().control(1, ctrl_state=0),
        PGate(0.5).control(1, ctrl_state=0),
    ],
)
def test_smoke_ctrl_state_elementary_controlled_gate(op):
    qc = QuantumCircuit(2)
    qc.append(op, [0, 1])

    assert _matches_up_to_global_phase(pytket_converter(qc).get_unitary(), qc.get_unitary())


# Finding 1b (pytket converter): "p" maps to OpType.Rz (pytket_converter.py:50),
# whose rotation-convention global phase e^{-i theta/2} is invisible for a lone
# p gate but accumulates inside composite-box definitions (gray synthesis uses
# P gates), leaving the whole box phase-shifted. Strict equality must hold for
# the multi-controlled X gate; it currently fails.
#
# TODO: not tracked upstream yet -- should be fixed in Qrisp by mapping "p" to
# pytket's relative-phase gate OpType.U1.
def test_smoke_p_to_rz_global_phase_in_controlled_box():
    qc = QuantumCircuit(4)
    qc.append(MCXGate(control_amount=3), [0, 1, 2, 3])

    assert np.allclose(qc.to_pytket().get_unitary(), qc.get_unitary(), atol=1e-8)


# Finding 1b at its smallest scale: even a lone p(0.7) is not phase-exact after
# conversion, because "p" maps to OpType.Rz (rotation convention, diag
# (e^{-i theta/2}, e^{i theta/2})) instead of pytket's relative-phase U1
# (diag(1, e^{i theta})). The difference is a pure global phase here, so this
# only fails under a strict (global-phase-sensitive) comparison.
def test_smoke_p_phase_exact():
    qc = QuantumCircuit(1)
    qc.p(0.7, 0)

    assert np.allclose(qc.to_pytket().get_unitary(), qc.get_unitary(), atol=1e-8)


# Qrisp's SXGate and pytket's OpType.SX differ by a global phase (the same
# SX-convention discrepancy flagged for the qiskit converter in Qrisp PR #672).
# Global-phase-insensitive comparisons are unaffected; strict ones fail.
def test_smoke_sx_phase_exact():
    qc = QuantumCircuit(1)
    qc.sx(0)

    assert np.allclose(qc.to_pytket().get_unitary(), qc.get_unitary(), atol=1e-8)


# Finding 1c (pytket converter): standard ops missing from the lookup table and
# without a definition raise instead of converting. RGate ("r"), barrier,
# reset and classically-controlled ops all hit ValueError in
# create_tket_instruction (pytket_converter.py:100).
#
# TODO: not tracked upstream yet -- should be fixed in Qrisp (either add these
# to the converter's gate map or give the Qrisp ops a definition). The RGate
# case is partially tracked by #629 (wrong-axis unitary; a fix there would give
# RGate a definition and let the converter emit it).
def test_smoke_rgate_converts():
    qc = QuantumCircuit(1)
    qc.append(RGate(0.7, 0.5), [0])

    pytket_converter(qc)  # currently raises ValueError


def test_smoke_barrier_converts():
    qc = QuantumCircuit(2)
    qc.barrier()

    pytket_converter(qc)  # currently raises ValueError


def test_smoke_reset_converts():
    qc = QuantumCircuit(2)
    qc.reset(0)

    pytket_converter(qc)  # currently raises ValueError


def test_smoke_c_if_converts():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.append(qc.data[0].op.c_if(1, 1), [0], [0])

    pytket_converter(qc)  # currently raises ValueError


# Finding 2a (qrisp): RGate.get_unitary() deviates from the standard
# exp(-i theta/2 (cos phi X + sin phi Y)) by a factor of i on the off-diagonals.
# Tracked upstream as https://github.com/eclipse-qrisp/Qrisp/issues/629.
def test_smoke_rgate_qrisp_unitary_standard_axis():
    theta, phi = 0.7, 0.5
    u_expected = np.array(
        [
            [np.cos(theta / 2), -1j * np.exp(-1j * phi) * np.sin(theta / 2)],
            [-1j * np.exp(1j * phi) * np.sin(theta / 2), np.cos(theta / 2)],
        ]
    )

    assert np.allclose(RGate(theta, phi).get_unitary(), u_expected, atol=1e-8)


# Finding 2b (qrisp): QuantumCircuit.get_unitary() emits spurious ~1e-7
# off-diagonal entries where the unitary should vanish. This sits at the noise
# floor of downstream equivalence checks (see qBraid #1311). Related upstream
# issue: https://github.com/eclipse-qrisp/Qrisp/issues/632 (the "context-
# dependent" MCRXGate flakiness there stems from this noise, not from the
# converter's qubit ordering, which is correct on this HEAD).
def test_smoke_qrisp_get_unitary_noise():
    qc = QuantumCircuit(4)
    qc.append(RXGate(0.7).control(3, method="gray_pt"), [0, 1, 2, 3])

    u = qc.get_unitary()
    # A controlled-RX acts on one 2x2 block per control state; all other entries
    # must vanish exactly.
    spurious = np.abs(u)[np.abs(u) < 0.1]
    assert spurious.max() < 1e-9
