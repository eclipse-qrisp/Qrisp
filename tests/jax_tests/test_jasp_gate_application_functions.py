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

"""Tests for the gate application functions in Jasp mode."""

import numpy as np
import pytest

from qrisp import QuantumArray, QuantumBool, QuantumFloat, QuantumSession, QuantumVariable, measure
from qrisp.core.gate_application_functions import (
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
    measure_to_big_integer,
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
from qrisp.jasp import boolean_simulation, jaspify, make_jaspr, terminal_sampling

# Helpers


def bug(reason, raises=None):
    """Return a strict xfail mark for a known bug of the gate application functions."""
    return pytest.mark.xfail(strict=True, raises=raises, reason=reason)


def mcx_truth_table(apply, n):
    """Apply an mcx to a QuantumFloat in superposition and return the sampled pairs."""

    @terminal_sampling
    def main():
        controls, target = QuantumFloat(n), QuantumBool()
        h(controls)
        apply(controls, target)
        return controls, target

    return {(int(value), bool(flipped)) for value, flipped in main()}


def expected_truth_table(activating_value, n):
    """Return the pairs of an mcx on n controls that flips the target for activating_value."""
    return {(value, value == activating_value) for value in range(2**n)}


def jasp_and_static_values(apply, n=3):
    """Return the values measured after apply on a QuantumFloat in superposition.

    The values are sampled once in Jasp mode and once in static mode.
    """

    def prepare():
        qf = QuantumFloat(n)
        h(qf)
        apply(qf)
        return qf

    @terminal_sampling
    def main():
        return prepare()

    jasp_values = {int(value) for value in main()}
    static_values = {int(value) for value in prepare().get_measurement()}
    return jasp_values, static_values


# Gates

JASP_GATES = {
    "cx": lambda a, q: cx(q[1], q[0]),
    "cy": lambda a, q: cy(q[1], q[0]),
    "cz": lambda a, q: cz(q[1], q[0]),
    "swap": lambda a, q: swap(q[1], q[0]),
    "h": lambda a, q: h(q[0]),
    "x": lambda a, q: x(q[0]),
    "y": lambda a, q: y(q[0]),
    "z": lambda a, q: z(q[0]),
    "s": lambda a, q: s(q[0]),
    "t": lambda a, q: t(q[0]),
    "s_dg": lambda a, q: s_dg(q[0]),
    "t_dg": lambda a, q: t_dg(q[0]),
    "sx": lambda a, q: sx(q[0]),
    "sx_dg": lambda a, q: sx_dg(q[0]),
    "p": lambda a, q: p(a, q[0]),
    "rx": lambda a, q: rx(a, q[0]),
    "ry": lambda a, q: ry(a, q[0]),
    "rz": lambda a, q: rz(a, q[0]),
    "gphase": lambda a, q: gphase(a, q[0]),
    "cp": lambda a, q: cp(a, q[1], q[0]),
    "crz": lambda a, q: crz(a, q[1], q[0]),
    "rzz": lambda a, q: rzz(a, q[1], q[0]),
    "rxx": lambda a, q: rxx(a, q[1], q[0]),
    "ryy": lambda a, q: ryy(a, q[1], q[0]),
    "xxyy": lambda a, q: xxyy(a, a / 2, q[1], q[0]),
    "u3": lambda a, q: u3(a, a / 2, a / 3, q[0]),
}


@pytest.mark.parametrize("name", JASP_GATES)
def test_jasp_gates_match_static_gates(name):
    """With a traced angle, each gate gives the same circuit unitary as in static mode."""
    apply = JASP_GATES[name]

    def main(angle):
        qv = QuantumVariable(2)
        apply(angle, qv)
        return qv

    jasp_unitary = make_jaspr(main)(0.0).to_qc(0.3)[-1].get_unitary()
    qs = main(0.3).qs
    assert isinstance(qs, QuantumSession)
    assert np.allclose(jasp_unitary, qs.compile().get_unitary())


# Inputs without a static number of qubits: a QuantumVariable, a slice, a single qubit.
UNSIZED_BARRIER_ARGUMENTS = [lambda qf: qf, lambda qf: qf[0:2], lambda qf: qf[0]]


@pytest.mark.parametrize("argument", UNSIZED_BARRIER_ARGUMENTS)
def test_jasp_barrier_without_a_static_size_does_nothing(argument):
    """In Jasp mode, such a barrier adds nothing to the Jaspr and leaves the state unchanged."""

    def program(apply):
        def main():
            qf = QuantumFloat(3)
            x(qf[0])
            apply(argument(qf))
            return measure(qf)

        return main

    without_barrier = make_jaspr(program(lambda qubits: None))()
    assert str(make_jaspr(program(barrier))()) == str(without_barrier)

    @jaspify
    def simulate():
        return program(barrier)()

    assert simulate() == 1


@pytest.mark.parametrize("argument", [lambda qf: [qf[0], qf[2]], lambda qf: (qf[0], qf[2])])
def test_jasp_barrier_on_a_list_of_qubits(argument):
    """In Jasp mode, a barrier on a list of qubits is applied as in static mode."""

    def main():
        qf = QuantumFloat(3)
        h(qf[0])
        barrier(argument(qf))
        cx(qf[0], qf[2])
        return qf

    circuit = make_jaspr(main)().to_qc()[-1]
    gates = [
        (instruction.op.name, [circuit.qubits.index(qubit) for qubit in instruction.qubits])
        for instruction in circuit.data
        if instruction.op.name in ("h", "barrier", "cx")
    ]
    assert gates == [("h", [0]), ("barrier", [0, 2]), ("cx", [0, 2])]


def test_jasp_barrier_on_a_quantum_array():
    """In Jasp mode, a barrier on a QuantumArray is applied to its qubits."""

    def main():
        qa = QuantumArray(QuantumBool(), shape=2)
        barrier(qa)
        return qa

    circuit = make_jaspr(main)().to_qc()[-1]
    barrier_qubits = {
        circuit.qubits.index(qubit)
        for instruction in circuit.data
        if instruction.op.name == "barrier"
        for qubit in instruction.qubits
    }
    assert barrier_qubits == {0, 1}


def test_jasp_unitary():
    """An arbitrary unitary can be traced and simulated."""

    @jaspify
    def main():
        qb = QuantumBool()
        unitary(np.array([[0, 1], [1, 0]], dtype=complex), qb[0])
        return measure(qb)

    assert main()


def random_unitary(seed):
    """Return a random 2x2 unitary matrix."""
    rng = np.random.default_rng(seed)
    matrix, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    return matrix


UNITARIES = [random_unitary(seed) for seed in range(3)] + [
    np.eye(2, dtype=complex),
    np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
]


@pytest.mark.parametrize("traced", [False, True])
@pytest.mark.parametrize("matrix", UNITARIES)
def test_jasp_unitary_applies_the_matrix(matrix, traced):
    """The circuit of the Jaspr applies the matrix, given as a constant or traced."""

    def main(traced_matrix):
        qv = QuantumVariable(1)
        unitary(traced_matrix if traced else matrix, qv[0])
        return qv

    jasp_circuit = make_jaspr(main)(matrix).to_qc(matrix)[-1]
    assert np.allclose(jasp_circuit.get_unitary(), matrix)


# mcx

LIST_METHODS = [
    "gray",
    "gray_pt",
    "gray_pt_inv",
    "balauca",
    "yong",
    "khattar",
    "auto",
]


@pytest.mark.parametrize("method", LIST_METHODS)
@pytest.mark.parametrize("ctrl_state, activating_value", [(-1, 3), (1, 1)])
def test_jasp_mcx_list_of_controls(method, ctrl_state, activating_value):
    """Each method flips the target exactly for the controls matching ctrl_state."""

    def apply(controls, target):
        mcx([controls[0], controls[1]], target, method=method, ctrl_state=ctrl_state)

    assert mcx_truth_table(apply, 2) == expected_truth_table(activating_value, 2)


GIDNEY_CONTROL_STATES = [(-1, 3), pytest.param(1, 1, marks=bug("gidney ignores ctrl_state"))]


@pytest.mark.parametrize("ctrl_state, activating_value", GIDNEY_CONTROL_STATES)
def test_jasp_mcx_gidney(ctrl_state, activating_value):
    """The Gidney logical AND flips the target exactly for the controls matching ctrl_state."""
    table = mcx_truth_table(
        lambda c, t: mcx([c[0], c[1]], t, method="gidney", ctrl_state=ctrl_state),
        2,
    )
    assert table == expected_truth_table(activating_value, 2)


def test_jasp_mcx_gidney_inverse_uncomputes():
    """The gidney_inv method uncomputes the logical AND computed by gidney."""

    def apply(controls, target):
        mcx([controls[0], controls[1]], target, method="gidney")
        mcx([controls[0], controls[1]], target, method="gidney_inv")

    assert mcx_truth_table(apply, 2) == {(value, False) for value in range(4)}


@pytest.mark.parametrize("method", ["balauca", "khattar", "gray", "maslov"])
def test_jasp_mcx_quantum_variable_controls(method):
    """Each method gives the right result for a QuantumVariable of controls."""
    table = mcx_truth_table(lambda c, t: mcx(c, t, method=method, ctrl_state=5), 3)
    assert table == expected_truth_table(5, 3)


@pytest.mark.parametrize("ctrl_state, activating_value", [("100", 1), ("110", 3)])
@pytest.mark.parametrize(
    "method",
    ["gray", "khattar", pytest.param("balauca", marks=bug("string read as a decimal number"))],
)
def test_jasp_mcx_string_control_state(method, ctrl_state, activating_value):
    """A string control state gives the state of each control, as in static mode."""

    def apply(controls, target):
        mcx([controls[0], controls[1], controls[2]], target, method=method, ctrl_state=ctrl_state)

    assert mcx_truth_table(apply, 3) == expected_truth_table(activating_value, 3)


@pytest.mark.parametrize("method", ["maslov", "balauca_dirty", "hybrid", "amy", "jones"])
def test_jasp_static_mcx_methods_are_not_available(method):
    """The methods that build static circuits raise a ValueError in Jasp mode."""

    def main():
        qf, qb = QuantumFloat(2), QuantumBool()
        mcx([qf[0], qf[1]], qb, method=method)
        return measure(qb)

    message = f"The {method} method of mcx is not available in Jasp mode"
    with pytest.raises(ValueError, match=message):
        make_jaspr(main)()


def test_jasp_mcx_toffoli_methods_need_two_controls():
    """The Toffoli constructions raise for any number of controls but two."""

    def main():
        qf, qb = QuantumFloat(3), QuantumBool()
        mcx([qf[0], qf[1], qf[2]], qb, method="gidney")
        return measure(qb)

    message = "Tried to call Gidney logical AND with 3 controls instead of two"
    with pytest.raises(ValueError, match=message):
        make_jaspr(main)()


# mcz and mcp

STRING_STATE_BUG = bug("string control states are reversed in Jasp mode")
JASP_CONTROL_STATES = [-1, 3, pytest.param("011", marks=STRING_STATE_BUG)]


@pytest.mark.parametrize("hadamard_qubit", [0, 2])
@pytest.mark.parametrize("ctrl_state", JASP_CONTROL_STATES)
@pytest.mark.parametrize("method", ["balauca", "khattar", "auto"])
def test_jasp_mcz_matches_static_mcz(method, ctrl_state, hadamard_qubit):
    """The phase applied by each method, revealed by an H gate, is the one of static mode."""

    def apply(qf):
        mcz(qf, method=method, ctrl_state=ctrl_state)
        h(qf[hadamard_qubit])

    jasp_values, static_values = jasp_and_static_values(apply)
    assert jasp_values == static_values


@pytest.mark.parametrize("hadamard_qubit", [0, 2])
@pytest.mark.parametrize("ctrl_state", JASP_CONTROL_STATES)
@pytest.mark.parametrize("method", ["balauca", "khattar", "auto"])
def test_jasp_mcp_matches_static_mcp(method, ctrl_state, hadamard_qubit):
    """The phase pi applied by each method, revealed by an H gate, is the one of static mode."""

    def apply(qf):
        mcp(np.pi, qf, method=method, ctrl_state=ctrl_state)
        h(qf[hadamard_qubit])

    jasp_values, static_values = jasp_and_static_values(apply)
    assert jasp_values == static_values


@pytest.mark.parametrize("hadamard_qubit", [0, 2])
@pytest.mark.parametrize("ctrl_state", [-1, 3, "011"])
@pytest.mark.parametrize("method", ["gray", "gray_pt"])
def test_jasp_gray_mcp_on_a_list_of_qubits(method, ctrl_state, hadamard_qubit):
    """On a list of qubits, the gray-code methods apply the phase of static mode.

    Unlike the other methods in Jasp mode, they read a string control state as static
    mode does.
    """

    def apply(qf):
        mcp(np.pi, [qf[0], qf[1], qf[2]], method=method, ctrl_state=ctrl_state)
        h(qf[hadamard_qubit])

    jasp_values, static_values = jasp_and_static_values(apply)
    assert jasp_values == static_values


@pytest.mark.parametrize("method", ["gray", "gray_pt"])
def test_jasp_gray_mcp_needs_a_list_of_qubits(method):
    """The gray-code methods raise a TypeError for a QuantumVariable in Jasp mode."""

    def main():
        qf = QuantumFloat(3)
        mcp(0.5, qf, method=method)
        return measure(qf)

    with pytest.raises(TypeError, match=f"mcp method {method} needs a list of qubits"):
        make_jaspr(main)()


@pytest.mark.parametrize("gate", [mcx, mcz, mcp])
def test_jasp_unknown_methods(gate):
    """Unknown methods raise a ValueError in Jasp mode too."""

    def main():
        qf = QuantumFloat(3)
        if gate is mcx:
            mcx([qf[0], qf[1]], qf[2], method="nonsense")
        elif gate is mcz:
            mcz([qf[0], qf[1], qf[2]], method="nonsense")
        else:
            mcp(0.5, qf, method="nonsense")
        return measure(qf)

    with pytest.raises(ValueError, match="Don't know (mcx )?method nonsense"):
        make_jaspr(main)()


# measure_to_big_integer

BIG_VALUES = [(32, 2**31 + 5), (40, 2**39 + 12345), (70, 2**69 + 2**40 + 7)]


@pytest.mark.parametrize("n, value", BIG_VALUES)
def test_measure_to_big_integer(n, value):
    """The measured value is split into 32-bit digits, least significant first."""

    @boolean_simulation
    def main():
        qf = QuantumFloat(n)
        for i in range(n):
            if (value >> i) & 1:
                x(qf[i])
        return measure_to_big_integer(qf, 3).digits

    assert np.asarray(main()).tolist() == [(value >> (32 * i)) & 0xFFFFFFFF for i in range(3)]


# reset


def test_jasp_reset():
    """QuantumVariables, QuantumArrays and single qubits are reset to zero."""

    @jaspify
    def main():
        qf = QuantumFloat(3)
        x(qf)
        reset(qf)
        qa = QuantumArray(QuantumBool(), shape=2)
        x(qa)
        reset(qa)
        qb = QuantumBool()
        x(qb)
        reset(qb[0])
        return measure(qf), measure(qa), measure(qb)

    qf_value, qa_values, qb_value = main()
    assert float(qf_value) == 0
    assert not np.any(qa_values)
    assert not qb_value


QUBIT_ARRAYS = {
    "register": lambda qf: qf.reg,
    "register tracer": lambda qf: qf.reg.tracer,
    "slice": lambda qf: qf[0:2],
}


@pytest.mark.parametrize("name", QUBIT_ARRAYS)
def test_jasp_reset_qubit_array(name):
    """A qubit array, as a DynamicQubitArray or as its tracer, can be reset too."""

    @jaspify
    def main():
        qf = QuantumFloat(3)
        x(qf)
        reset(QUBIT_ARRAYS[name](qf))
        return measure(qf)

    # The slice holds qubits 0 and 1, so only qubit 2 stays in state 1.
    assert main() == (4 if name == "slice" else 0)


def test_jasp_reset_rejects_classical_values():
    """Traced classical values raise a TypeError."""

    def main(n):
        qf = QuantumFloat(2)
        reset(n)
        return measure(qf)

    with pytest.raises(TypeError, match="Tried to reset type"):
        make_jaspr(main)(1)
