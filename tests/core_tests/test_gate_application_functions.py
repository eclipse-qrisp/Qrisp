# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Tests for the gate application functions in Qrisp."""

import numpy as np
import pytest
import sympy

from qrisp import (
    CXGate,
    QuantumBool,
    QuantumFloat,
    QuantumSession,
    QuantumVariable,
    XGate,
    invert,
    multi_measurement,
)
from qrisp.core.gate_application_functions import (
    append_operation,
    barrier,
    cp,
    crz,
    cx,
    cy,
    cz,
    gphase,
    h,
    mcp,
    mcx,
    mcz,
    p,
    reset,
    rx,
    rxx,
    ry,
    ryy,
    rz,
    rzz,
    s,
    s_dg,
    swap,
    sx,
    sx_dg,
    t,
    t_dg,
    u3,
    unitary,
    x,
    xxyy,
    y,
    z,
)

# Every gate-application function that was changed to return None instead of
# echoing back its qubit arguments. Each entry applies the gate to fresh
# qubits of a 6-qubit QuantumVariable, which is enough for every arity here.
GATE_CALLS = {
    "cx": lambda qv: cx(qv[0], qv[1]),
    "cy": lambda qv: cy(qv[0], qv[1]),
    "cz": lambda qv: cz(qv[0], qv[1]),
    "h": lambda qv: h(qv[0]),
    "x": lambda qv: x(qv[0]),
    "y": lambda qv: y(qv[0]),
    "z": lambda qv: z(qv[0]),
    "mcx": lambda qv: mcx(qv[:3], qv[3]),
    "mcz": lambda qv: mcz(qv[:4]),
    "mcp": lambda qv: mcp(0.3, qv[:4]),
    "p": lambda qv: p(0.3, qv[0]),
    "cp": lambda qv: cp(0.3, qv[0], qv[1]),
    "rx": lambda qv: rx(0.3, qv[0]),
    "ry": lambda qv: ry(0.3, qv[0]),
    "rz": lambda qv: rz(0.3, qv[0]),
    "crz": lambda qv: crz(0.3, qv[0], qv[1]),
    "s": lambda qv: s(qv[0]),
    "t": lambda qv: t(qv[0]),
    "s_dg": lambda qv: s_dg(qv[0]),
    "t_dg": lambda qv: t_dg(qv[0]),
    "sx": lambda qv: sx(qv[0]),
    "sx_dg": lambda qv: sx_dg(qv[0]),
    "gphase": lambda qv: gphase(0.3, qv[0]),
    "xxyy": lambda qv: xxyy(0.3, 0.2, qv[0], qv[1]),
    "rzz": lambda qv: rzz(0.3, qv[0], qv[1]),
    "rxx": lambda qv: rxx(0.3, qv[0], qv[1]),
    "ryy": lambda qv: ryy(0.3, qv[0], qv[1]),
    "u3": lambda qv: u3(0.1, 0.2, 0.3, qv[0]),
    "barrier": lambda qv: barrier([qv[0], qv[1]]),
    "swap": lambda qv: swap(qv[0], qv[1]),
}


@pytest.mark.parametrize("name", GATE_CALLS.keys())
def test_gate_application_functions_return_none(name):
    """Every gate-application function returns None.

    The gate is applied by side effect, the qubit arguments are not echoed back.
    """
    qv = QuantumVariable(6)
    assert GATE_CALLS[name](qv) is None


@pytest.mark.parametrize("controls_amount", [0, 1, 2, 3])
def test_mcx_returns_none_for_every_control_count(controls_amount):
    """Each control count returns None.

    mcx used to special-case 0 and 1 controls with their own early ``return``.
    """
    qv = QuantumVariable(controls_amount + 1)
    assert mcx(qv[:controls_amount], qv[controls_amount]) is None


@pytest.mark.parametrize("method", ["gray", "auto", "balauca"])
def test_mcz_applies_the_gate_exactly_once(method):
    """Regression test for mcz applying the gate twice.

    Removing mcz's ``return qubits`` also removed the early exit it doubled as:
    the "gray"/"auto" branch fell through into also calling ``mcz_inner``,
    applying the multi-controlled Z gate twice. Since Z*Z = I, this silently
    cancelled the phase tag instead of raising an error, so it is checked via
    the resulting phase rather than just the return value (which is None either
    way).
    """
    qv = QuantumVariable(3)
    x(qv)
    assert mcz(list(qv.reg), method=method) is None
    assert str(session_of(qv).statevector()) == "-|111>"


# Helpers


def session_of(holder):
    """Return the QuantumSession of a QuantumVariable in static mode."""
    qs = holder.qs
    assert isinstance(qs, QuantumSession)
    return qs


def unitary_of(gate):
    """Return the unitary of a gate application function applied to one qubit."""
    qv = QuantumVariable(1)
    gate(qv[0])
    return session_of(qv).compile().get_unitary()


def appended(holder):
    """Return the instructions of a QuantumSession, without qubit allocations."""
    data = session_of(holder).data
    return [instruction for instruction in data if instruction.op.name != "qb_alloc"]


def equal_up_to_global_phase(a, b):
    """Return whether two matrices are equal up to a global phase."""
    index = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    phase = a[index] / b[index]
    return bool(np.isclose(abs(phase), 1) and np.allclose(a, phase * b))


def flipped_values(apply, hadamard_qubit, n=3):
    """Return the values measured after a phase gate on a QuantumFloat in superposition.

    The H gate applied after the phase gate on hadamard_qubit turns a phase of -1 into
    a bit flip of hadamard_qubit, which reveals the basis state the phase acted on.
    """
    qf = QuantumFloat(n)
    h(qf)
    apply(qf)
    h(qf[hadamard_qubit])
    return {int(value) for value in qf.get_measurement()}


def expected_phase_flips(basis_state, hadamard_qubit, n=3):
    """Return the values of flipped_values for a phase of -1 on basis_state."""
    bit = 1 << hadamard_qubit
    unflipped = [value for value in range(2**n) if not value & bit]
    return {value | bit if value == basis_state & ~bit else value for value in unflipped}


# append_operation


def test_append_operation_ignores_empty_qubit_lists():
    """Appending without qubits, or to empty qubit lists, does nothing."""
    qv = QuantumVariable(1)
    append_operation(XGate())
    append_operation(CXGate(), [[], []])
    cx([], [])
    assert not appended(qv)


def test_append_operation_without_quantum_session_raises():
    """Qubit arguments without a QuantumSession that are not empty lists raise."""
    with pytest.raises(Exception, match="Couldn't find QuantumSession"):
        append_operation(CXGate(), [[], [5]])


def test_append_operation_reraises_other_errors():
    """Errors of the QuantumSession, such as mismatched qubit lists, propagate."""
    a, b = QuantumVariable(3), QuantumVariable(3)
    with pytest.raises(ValueError, match="Don't know how to combine"):
        cx(a, b[:1])


# Gates


SQRT_X = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2
SINGLE_QUBIT_UNITARIES = {
    h: np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    x: np.array([[0, 1], [1, 0]]),
    y: np.array([[0, -1j], [1j, 0]]),
    z: np.diag([1, -1]),
    s: np.diag([1, 1j]),
    t: np.diag([1, np.exp(1j * np.pi / 4)]),
    s_dg: np.diag([1, -1j]),
    t_dg: np.diag([1, np.exp(-1j * np.pi / 4)]),
    sx: SQRT_X,
}


@pytest.mark.parametrize("gate", SINGLE_QUBIT_UNITARIES, ids=lambda gate: gate.__name__)
def test_single_qubit_gate_unitaries(gate):
    """Each single-qubit gate applies its matrix, up to a global phase."""
    assert equal_up_to_global_phase(unitary_of(gate), SINGLE_QUBIT_UNITARIES[gate])


@pytest.mark.xfail(strict=True, reason="sx_dg applies SXDGGate().inverse(), which is SX")
def test_sx_dg_is_the_inverse_of_sx():
    """sx_dg applies the conjugate transpose of SX."""
    assert equal_up_to_global_phase(unitary_of(sx_dg), SQRT_X.conj().T)


GATE_STRUCTURE = [
    (lambda q: cx(q[1], q[0]), "cx", [], [1, 0]),
    (lambda q: cy(q[1], q[0]), "cy", [], [1, 0]),
    (lambda q: cz(q[1], q[0]), "cz", [], [1, 0]),
    (lambda q: swap(q[1], q[0]), "swap", [], [1, 0]),
    (lambda q: p(0.3, q[0]), "p", [0.3], [0]),
    (lambda q: rx(0.3, q[0]), "rx", [0.3], [0]),
    (lambda q: ry(0.3, q[0]), "ry", [0.3], [0]),
    (lambda q: rz(0.3, q[0]), "rz", [0.3], [0]),
    (lambda q: gphase(0.3, q[0]), "gphase", [0.3], [0]),
    (lambda q: cp(0.3, q[1], q[0]), "cp", [0.3], [1, 0]),
    (lambda q: crz(0.3, q[1], q[0]), "crz", [0.3], [1, 0]),
    (lambda q: rzz(0.3, q[1], q[0]), "rzz", [0.3], [1, 0]),
    (lambda q: rxx(0.3, q[1], q[0]), "rxx", [0.3], [1, 0]),
    (lambda q: ryy(0.3, q[1], q[0]), "ryy", [0.3], [1, 0]),
    (lambda q: xxyy(0.3, 0.2, q[1], q[0]), "xxyy", [0.3, 0.2], [1, 0]),
    (lambda q: u3(0.1, 0.2, 0.3, q[0]), "u3", [0.1, 0.2, 0.3], [0]),
    (lambda q: reset(q[0]), "reset", [], [0]),
    (lambda q: barrier(q[1]), "barrier", [], [1]),
    (lambda q: barrier([q[1], q[0]]), "barrier", [], [1, 0]),
    (barrier, "barrier", [], [0, 1]),
]


@pytest.mark.parametrize("apply, name, params, qubits", GATE_STRUCTURE)
def test_gate_structure(apply, name, params, qubits):
    """Each function appends its operation with the given parameters, on the given qubits."""
    qv = QuantumVariable(2)
    apply(qv)
    (instruction,) = appended(qv)
    assert instruction.op.name == name
    assert list(instruction.op.params) == params
    assert [qv.reg.index(qubit) for qubit in instruction.qubits] == qubits


def test_symbolic_parameters_are_kept():
    """A SymPy symbol is appended as the parameter of the gate."""
    qv = QuantumVariable(2)
    phi = sympy.Symbol("phi")
    rz(phi, qv[0])
    mcp(phi, qv, method="gray")
    first, last = appended(qv)
    assert first.op.params == [phi]
    assert phi in last.op.base_operation.params


@pytest.mark.parametrize("seed", range(4))
def test_unitary_applies_a_random_unitary(seed):
    """A 2x2 unitary is decomposed into a U3 gate and a global phase."""
    rng = np.random.default_rng(seed)
    matrix, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    assert np.allclose(unitary_of(lambda qubit: unitary(matrix, qubit)), matrix)


SPECIAL_UNITARIES = [
    np.eye(2),
    np.array([[0, 1], [1, 0]]),
    np.array([[1, 1], [1, -1]]) / np.sqrt(2),
]


@pytest.mark.parametrize("matrix", SPECIAL_UNITARIES)
def test_unitary_applies_special_unitaries(matrix):
    """Matrices with zero entries are handled too."""
    assert np.allclose(unitary_of(lambda qubit: unitary(matrix.astype(complex), qubit)), matrix)


# mcx

MCX_METHODS = [
    "gray",
    "gray_pt",
    "gray_pt_inv",
    "maslov",
    "balauca",
    "balauca_dirty",
    "yong",
    "hybrid",
    "amy",
    "jones",
    "gidney",
    "auto",
    "khattar",
]


def mcx_truth_table(apply, n):
    """Apply an mcx to a QuantumFloat in superposition and return the measured pairs."""
    controls, target = QuantumFloat(n), QuantumBool()
    h(controls)
    apply(controls, target)
    return {(int(value), bool(flipped)) for value, flipped in multi_measurement([controls, target])}


def expected_truth_table(activating_value, n):
    """Return the pairs of an mcx on n controls that flips the target for activating_value."""
    return {(value, value == activating_value) for value in range(2**n)}


@pytest.mark.parametrize("ctrl_state", [-1, 1, "10"])
@pytest.mark.parametrize("method", MCX_METHODS)
def test_mcx_methods_two_controls(method, ctrl_state):
    """Each method flips the target exactly for the controls matching ctrl_state."""
    table = mcx_truth_table(lambda c, t: mcx(c, t, method=method, ctrl_state=ctrl_state), 2)
    # "10" is qubit 0 in 1 and qubit 1 in 0, that is the value 1.
    activating_value = {-1: 3, 1: 1, "10": 1}[ctrl_state]
    assert table == expected_truth_table(activating_value, 2)


ANY_CONTROLS_METHODS = ["gray", "maslov", "balauca", "yong", "hybrid", "khattar", "auto"]


@pytest.mark.parametrize("method", ANY_CONTROLS_METHODS)
def test_mcx_methods_three_controls(method):
    """Methods for any number of controls, with 3 controls (maslov then uses an ancilla)."""
    table = mcx_truth_table(lambda c, t: mcx(c, t, method=method, ctrl_state=5), 3)
    assert table == expected_truth_table(5, 3)


@pytest.mark.parametrize("ctrl_state, activating_value", [(-1, 1), (1, 1), (0, 0)])
def test_mcx_one_control(ctrl_state, activating_value):
    """With one control, mcx is a CX gate on the control state."""
    table = mcx_truth_table(lambda c, t: mcx(c, t, ctrl_state=ctrl_state), 1)
    assert table == expected_truth_table(activating_value, 1)


def test_mcx_without_controls_does_nothing():
    """An empty list of controls appends nothing."""
    target = QuantumBool()
    mcx([], target)
    assert not appended(target)


def test_mcx_accepts_quantum_bools_and_qubits():
    """QuantumBools in the list of controls are used as their qubit, a Qubit as target."""
    a, b, target = QuantumBool(), QuantumBool(), QuantumBool()
    x(a)
    x(b)
    mcx([a, b], target[0])
    assert target.get_measurement() == {True: 1.0}


def test_mcx_gidney_inverse_uncomputes():
    """gidney_inv uncomputes the logical AND computed by gidney."""
    controls, target = QuantumFloat(2), QuantumBool()
    h(controls)
    mcx(controls, target, method="gidney")
    mcx(controls, target, method="gidney_inv")
    assert {target_value for _, target_value in multi_measurement([controls, target])} == {False}


def test_mcx_gidney_inverted_by_invert():
    """In an InversionEnvironment, gidney becomes its uncomputation."""
    controls, target = QuantumFloat(2), QuantumBool()
    mcx(controls, target, method="gidney")
    with invert():
        mcx(controls, target, method="gidney")
    assert [instruction.op.name for instruction in appended(controls)][-2:] == [
        "uncompiled_gidney_mcx",
        "uncompiled_gidney_mcx_inv",
    ]


@pytest.mark.parametrize("target", [lambda: QuantumVariable(2), lambda: [QuantumBool()[0]]])
def test_mcx_rejects_other_targets(target):
    """The target must be a Qubit or a QuantumBool."""
    with pytest.raises(TypeError, match="mcx target is not of type Qubit or QuantumBool"):
        mcx(QuantumVariable(2), target())


@pytest.mark.parametrize("ctrl_state", ["1", "1011"])
def test_mcx_rejects_a_control_state_of_the_wrong_length(ctrl_state):
    """A control state string must have one character per control."""
    message = f"Given control state {ctrl_state} does not match control qubit amount 3"
    with pytest.raises(ValueError, match=message):
        mcx(QuantumVariable(3), QuantumBool(), ctrl_state=ctrl_state)


def test_mcx_rejects_unknown_methods():
    """Unknown methods raise a ValueError."""
    with pytest.raises(ValueError, match="Don't know mcx method nonsense"):
        mcx(QuantumVariable(2), QuantumBool(), method="nonsense")


@pytest.mark.parametrize(
    "method, name",
    [
        ("gidney", "Gidney logical AND"),
        ("gidney_inv", "Gidney logical AND"),
        ("amy", "Amy MCX"),
        ("jones", "Jones MCX"),
    ],
)
def test_mcx_toffoli_methods_need_two_controls(method, name):
    """The Toffoli constructions raise for any number of controls but two."""
    with pytest.raises(ValueError, match=f"Tried to call {name} with 3 controls instead of two"):
        mcx(QuantumVariable(3), QuantumBool(), method=method)


# mcz and mcp

# Pairs of a control state and the basis state it selects, whose bit i is the state of qubit i.
PHASE_CONTROL_STATES = [(-1, 0b111), (3, 0b011), ("011", 0b110)]


@pytest.mark.parametrize("hadamard_qubit", [0, 2])
@pytest.mark.parametrize("ctrl_state, basis_state", PHASE_CONTROL_STATES)
@pytest.mark.parametrize("method", ["gray", "auto", "balauca", "hybrid", "khattar"])
def test_mcz_phase(method, ctrl_state, basis_state, hadamard_qubit):
    """Each method puts a phase of -1 exactly on the basis state given by ctrl_state."""

    def apply(qf):
        mcz(qf, method=method, ctrl_state=ctrl_state)

    measured = flipped_values(apply, hadamard_qubit)
    assert measured == expected_phase_flips(basis_state, hadamard_qubit)


@pytest.mark.parametrize("ctrl_state, expected", [(-1, "-|1>"), (0, "|1>")])
def test_mcz_one_qubit(ctrl_state, expected):
    """With one qubit, mcz is a Z gate on the control state."""
    qv = QuantumVariable(1)
    x(qv)
    mcz(qv, ctrl_state=ctrl_state)
    assert str(session_of(qv).statevector()) == expected


@pytest.mark.parametrize("ctrl_state", ["1", "1011"])
def test_mcz_rejects_a_control_state_of_the_wrong_length(ctrl_state):
    """The helper used by the ancilla-based methods checks the control state."""
    with pytest.raises(ValueError, match="does not matchcontrol qubit amount 3"):
        mcz(QuantumVariable(3), method="balauca", ctrl_state=ctrl_state)


@pytest.mark.xfail(strict=True, reason="mcz does not forward num_ancilla to mcx")
def test_mcz_hybrid_uses_num_ancilla():
    """The hybrid method uses num_ancilla ancillas, as it does for mcx."""

    def qubit_count(num_ancilla):
        qv = QuantumVariable(6)
        mcz(qv, method="hybrid", num_ancilla=num_ancilla)
        return session_of(qv).compile().num_qubits()

    assert qubit_count(3) != qubit_count(1)


@pytest.mark.parametrize("hadamard_qubit", [0, 2])
@pytest.mark.parametrize("ctrl_state, basis_state", PHASE_CONTROL_STATES)
@pytest.mark.parametrize("method", ["gray", "balauca", "khattar", "auto"])
def test_mcp_phase(method, ctrl_state, basis_state, hadamard_qubit):
    """With phase pi, each method puts a phase of -1 exactly on the basis state of ctrl_state."""

    def apply(qf):
        mcp(np.pi, qf, method=method, ctrl_state=ctrl_state)

    measured = flipped_values(apply, hadamard_qubit)
    assert measured == expected_phase_flips(basis_state, hadamard_qubit)


def test_mcp_auto_uses_balauca_from_four_qubits():
    """With method auto, mcp is a gray-code gate below four qubits and balauca from four on."""
    small, large = QuantumVariable(3), QuantumVariable(4)
    mcp(0.5, small)
    mcp(0.5, large)
    assert all(instruction.op.name != "anc supported mcp" for instruction in appended(small))
    assert any(instruction.op.name == "anc supported mcp" for instruction in appended(large))


@pytest.mark.parametrize("ctrl_state", [-1, "011"])
def test_mcp_phase_tolerant_gray(ctrl_state):
    """The gray_pt method appends a phase-tolerant controlled phase gate."""
    qv = QuantumVariable(3)
    mcp(0.5, qv, method="gray_pt", ctrl_state=ctrl_state)
    (instruction,) = appended(qv)
    assert instruction.op.base_operation.params == [0.5]
    assert instruction.op.method == "gray_pt"


def test_mcp_rejects_unknown_methods():
    """Unknown methods raise a ValueError."""
    with pytest.raises(ValueError, match="Don't know method nonsense"):
        mcp(0.5, QuantumVariable(2), method="nonsense")
