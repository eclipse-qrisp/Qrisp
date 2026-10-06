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

"""Tests for control, which returns a ControlEnvironment or a ClControlEnvironment."""

import jax.numpy as jnp
import numpy as np
import pytest

from qrisp import (
    ClControlEnvironment,
    ControlEnvironment,
    QuantumBool,
    QuantumFloat,
    QuantumVariable,
    conjugate,
    control,
    h,
    invert,
    mcp,
    measure,
    rz,
    swap,
    x,
)
from qrisp.jasp import boolean_simulation, make_jaspr, qache, terminal_sampling

UNSUPPORTED_MESSAGE = "Don't know how to control from input type"


def bit(qbool):
    """Return the measured value of a QuantumBool in a deterministic state."""
    (value,) = qbool.get_measurement()
    return value


def quantum_env(*args, **kwargs):
    """Call control and check that it returned a ControlEnvironment."""
    env = control(*args, **kwargs)
    assert isinstance(env, ControlEnvironment)
    return env


def classical_env(*args, **kwargs):
    """Call control and check that it returned a ClControlEnvironment."""
    env = control(*args, **kwargs)
    assert isinstance(env, ClControlEnvironment)
    return env


# Static mode: dispatch


@pytest.mark.parametrize(
    "make_ctrl, expected_type",
    [
        (lambda qv: qv[0], ControlEnvironment),
        (lambda qv: [qv[0], qv[1]], ControlEnvironment),
        (lambda qv: (qv[0], qv[1]), ControlEnvironment),
        (lambda qv: qv, ControlEnvironment),
        (lambda qv: True, ClControlEnvironment),
        (lambda qv: False, ClControlEnvironment),
        (lambda qv: np.True_, ClControlEnvironment),
        (lambda qv: [True, np.False_], ClControlEnvironment),
        (lambda qv: (True, False), ClControlEnvironment),
        (lambda qv: jnp.array(True), ClControlEnvironment),
        (lambda qv: [jnp.array(True), jnp.array(False)], ClControlEnvironment),
    ],
)
def test_static_dispatch(make_ctrl, expected_type):
    """Qubits select ControlEnvironment and booleans select ClControlEnvironment."""
    qv = QuantumVariable(2)
    assert isinstance(control(make_ctrl(qv)), expected_type)


def test_static_quantum_controls_are_normalized():
    """A QuantumBool becomes its qubit, a QuantumVariable its qubits, a tuple a list."""
    qv = QuantumVariable(3)
    qb = QuantumBool()
    assert quantum_env(qb).ctrl_qubits == [qb[0]]
    assert quantum_env(qv).ctrl_qubits == list(qv)
    assert quantum_env((qv[0], qv[2])).ctrl_qubits == [qv[0], qv[2]]
    # QuantumBools inside a list are passed on as they are.
    assert quantum_env([qb, qb]).ctrl_qubits == [qb, qb]


def test_static_list_is_passed_through_unchanged():
    """A list of controls reaches the environment as the same object."""
    qubits = QuantumVariable(2)[:]
    booleans = [True, False]
    assert quantum_env(qubits).ctrl_qubits is qubits
    assert classical_env(booleans).ctrl_bls is booleans


def test_static_jax_arrays_become_python_booleans():
    """Concrete JAX arrays are converted to Python booleans for classical control."""
    env = classical_env([jnp.array(True), jnp.array(False)])
    assert env.ctrl_bls == [True, False]
    assert all(isinstance(value, bool) for value in env.ctrl_bls)


# Static mode: argument forwarding


def test_quantum_arguments_are_forwarded():
    """ctrl_state, ctrl_method and invert reach the ControlEnvironment."""
    qv = QuantumVariable(3)
    assert quantum_env(qv, "101").ctrl_state == "101"
    # An integer ctrl_state is stored as a bit string, bit i for qubit i.
    assert quantum_env(qv, ctrl_state=2).ctrl_state == "010"
    assert quantum_env(qv[0], 1, "gray").ctrl_method == "gray"
    assert quantum_env(qv[0], ctrl_method="gray").ctrl_method == "gray"
    assert quantum_env(qv[0], invert=True).invert is True


def test_classical_arguments_are_forwarded():
    """ctrl_state and invert reach the ClControlEnvironment."""
    # A string ctrl_state is read as a binary number.
    assert classical_env([True, False], "10").ctrl_state == int("10", 2)
    assert classical_env([True, False], ctrl_state=1).ctrl_state == 1
    # The third positional argument of ClControlEnvironment is invert.
    assert classical_env(True, -1, True).invert is True
    assert classical_env(True, invert=True).invert is True


def test_classical_control_rejects_ctrl_method():
    """ctrl_method is only accepted by quantum control."""
    with pytest.raises(TypeError, match="ctrl_method"):
        control(True, ctrl_method="gray")


# Static mode: errors


@pytest.mark.parametrize(
    "make_ctrl",
    [
        lambda qv: 1,
        lambda qv: 1.0,
        lambda qv: None,
        lambda qv: "11",
        lambda qv: [qv[0], True],
        lambda qv: [np.False_, jnp.array(True)],
    ],
)
def test_static_unsupported_input_raises_type_error(make_ctrl):
    """Unsupported types, and lists mixing kinds, raise a TypeError."""
    with pytest.raises(TypeError, match=UNSUPPORTED_MESSAGE):
        control(make_ctrl(QuantumVariable(1)))


def test_control_requires_an_argument():
    """Calling control without arguments raises a TypeError."""
    with pytest.raises(TypeError):
        control(*[])


# Static mode: semantics


@pytest.mark.parametrize("ctrl_state", ["10", 1])
@pytest.mark.parametrize("value", range(4))
def test_static_quantum_ctrl_state(value, ctrl_state):
    """String character i and integer bit i give the state of qubit i."""
    a = QuantumFloat(2)
    a[:] = value
    target = QuantumBool()
    with control([a[0], a[1]], ctrl_state=ctrl_state):
        x(target)
    # Both mean: qubit 0 in state 1 and qubit 1 in state 0, that is a == 1.
    assert bit(target) == (value == 1)


@pytest.mark.parametrize("value", range(4))
def test_static_quantum_invert(value):
    """invert=True activates the block for every state except ctrl_state."""
    a = QuantumFloat(2)
    a[:] = value
    target = QuantumBool()
    with control([a[0], a[1]], ctrl_state="10", invert=True):
        x(target)
    assert bit(target) == (value != 1)


@pytest.mark.parametrize(
    "ctrl, ctrl_state, applies",
    [
        (True, -1, True),
        (False, -1, False),
        ([np.True_, False], -1, False),
        # The string is a binary number: its last character belongs to the first boolean.
        ([False, True], "10", True),
        ([True, False], "10", False),
        ([jnp.array(False), jnp.array(True)], 2, True),
    ],
)
def test_static_classical_control(ctrl, ctrl_state, applies):
    """The block only takes effect if the booleans match ctrl_state."""
    target = QuantumBool()
    with control(ctrl, ctrl_state=ctrl_state):
        x(target)
    assert bit(target) == applies


@pytest.mark.parametrize("flag", [True, False])
def test_static_classical_invert(flag):
    """invert=True negates a classical condition."""
    target = QuantumBool()
    with control(flag, invert=True):
        x(target)
    assert bit(target) == (not flag)


@pytest.mark.parametrize("flag, inverted", [(False, False), (True, True)])
def test_static_classical_inactive_block_errors_are_discarded(flag, inverted):
    """An error raised by a block that does not take effect is discarded with the block."""
    target = QuantumBool()
    with control(flag, invert=inverted):
        x(target)
        raise IndexError("raised inside an inactive block")
    assert not bit(target)


def test_static_classical_active_block_errors_propagate():
    """An error raised by a block that takes effect propagates."""
    with pytest.raises(IndexError), control(False, invert=True):
        raise IndexError("raised inside an active block")


# Jasp mode: dispatch


@pytest.mark.parametrize(
    "make_ctrl, expected_type",
    [
        (lambda a, flag, m: a[0], ControlEnvironment),
        (lambda a, flag, m: [a[0], a[1]], ControlEnvironment),
        (lambda a, flag, m: (a[0], a[1]), ControlEnvironment),
        (lambda a, flag, m: flag, ControlEnvironment),
        (lambda a, flag, m: m, ClControlEnvironment),
        (lambda a, flag, m: measure(a) == 1, ClControlEnvironment),
        (lambda a, flag, m: [m, m], ClControlEnvironment),
        (lambda a, flag, m: True, ClControlEnvironment),
        (lambda a, flag, m: jnp.array(True), ClControlEnvironment),
    ],
)
def test_jasp_dispatch(make_ctrl, expected_type):
    """Traced qubits select ControlEnvironment, traced and Python booleans ClControlEnvironment."""
    environment_types = []

    def main():
        a = QuantumFloat(2)
        flag = QuantumBool()
        m = measure(flag)
        environment_types.append(type(control(make_ctrl(a, flag, m))))
        return m

    make_jaspr(main)()
    assert environment_types == [expected_type]


@pytest.mark.parametrize(
    "make_ctrl",
    [
        lambda a, flag, m: 1,
        lambda a, flag, m: np.True_,
        lambda a, flag, m: [a[0], m],
        lambda a, flag, m: [flag, flag],
    ],
)
def test_jasp_unsupported_input_raises_type_error(make_ctrl):
    """In Jasp mode, unsupported types and mixed lists raise a TypeError."""

    def main():
        a = QuantumFloat(2)
        flag = QuantumBool()
        m = measure(flag)
        control(make_ctrl(a, flag, m))
        return m

    with pytest.raises(TypeError, match=UNSUPPORTED_MESSAGE):
        make_jaspr(main)()


# Jasp mode: compilation


def flip_if_one(value):
    """Flip qubit 1 of a two-qubit QuantumFloat if its qubit 0 is measured in 1."""
    a = QuantumFloat(2)
    a[:] = value
    with control(measure(a[0])):
        x(a[1])
    return measure(a)


def controlled_flip(value):
    """Flip a QuantumBool, controlled on qubit 0 of a QuantumFloat."""
    a = QuantumFloat(2)
    a[:] = value
    target = QuantumBool()
    with control(a[0]):
        x(target)
    return measure(target)


def test_jasp_classical_control_becomes_cond():
    """Classical control is recorded as q_env markers and flattened into a cond primitive."""
    unflattened = make_jaspr(flip_if_one, flatten_envs=False)(1)
    assert "jasp.q_env" in [eqn.primitive.name for eqn in unflattened.eqns]
    assert "cond" in [eqn.primitive.name for eqn in make_jaspr(flip_if_one)(1).eqns]


def test_jasp_quantum_control_becomes_controlled_call():
    """Quantum control is flattened into a call of the controlled body."""
    jaspr = make_jaspr(controlled_flip)(1)
    jit_names = [eqn.params["name"] for eqn in jaspr.eqns if eqn.primitive.name == "jit"]
    assert "ctrl_env" in jit_names


def test_jasp_quantum_control_of_a_conjugation():
    """A conjugation inside a quantum control is compiled into the controlled body."""

    @terminal_sampling
    def main(phi, i):
        qv = QuantumFloat(i)
        x(qv[: qv.size - 1])
        qbl = QuantumBool()
        h(qbl)
        with control(qbl[0]):
            with conjugate(h)(qv[qv.size - 1]):
                mcp(phi, qv)
        return qv

    # Only the branch with qbl in |1> flips the last qubit of qv. The expected
    # distribution comes first, since type checkers see main as returning qv.
    assert {15.0: 0.5, 31.0: 0.5} == main(np.pi, 5)


def test_jasp_quantum_control_of_a_qached_function_with_a_closed_over_array():
    """A qached function that closes over a JAX array can be called in a controlled block."""
    # The closed-over array becomes a constant of the qached body.
    angles = jnp.array([jnp.pi], dtype=jnp.float64)

    @qache
    def apply_const_angle(qv):
        rz(angles[0], qv[0])

    def main():
        ctrl = QuantumBool()
        qv = QuantumVariable(1)
        h(ctrl[0])
        h(qv[0])
        with control(ctrl[0]):
            apply_const_angle(qv)
        return measure(qv)

    jaspr = make_jaspr(main)()
    jaspr.to_qc()
    assert int(jaspr()) in {0, 1}


# Jasp mode: semantics


def test_jasp_nested_quantum_control():
    """Nested quantum controls combine their conditions and control states."""

    @boolean_simulation
    def main(value):
        b = QuantumFloat(4)
        b[:] = value
        target = QuantumBool()
        with control([b[0], b[1]], 3):
            with control([b[2], b[3]], ctrl_state="10"):
                x(target)
        return measure(target)

    # Qubits 0 and 1 in 1 (ctrl_state 3), qubit 2 in 1 and qubit 3 in 0 ("10").
    activating_value = 0b0111
    expected = [value == activating_value for value in range(16)]
    assert [bool(main(value)) for value in range(16)] == expected


def test_jasp_classical_control():
    """The block runs only if the measured qubit is 1."""
    simulate = boolean_simulation(flip_if_one)
    # Qubit 1 is flipped for the odd values.
    assert [int(simulate(value)) for value in range(4)] == [0, 3, 2, 1]


def test_jasp_classical_ctrl_state_and_invert():
    """A string ctrl_state is a binary number, and invert negates the condition."""

    @boolean_simulation
    def main(n):
        a = QuantumFloat(2)
        a[:] = n
        bits = [measure(a[0]), measure(a[1])]
        matched = QuantumBool()
        with control(bits, ctrl_state="10"):
            x(matched)
        not_matched = QuantumBool()
        with control(bits, ctrl_state="10", invert=True):
            x(not_matched)
        return measure(matched), measure(not_matched)

    # "10" is the binary number 2: bit 0 (qubit 0) is 0 and bit 1 (qubit 1) is 1.
    activating_value = int("10", 2)
    for value in range(4):
        matched, not_matched = main(value)
        expected = (value == activating_value, value != activating_value)
        assert (bool(matched), bool(not_matched)) == expected


def test_jasp_nested_classical_control():
    """Classical controls nest, and a block can allocate, measure and delete a variable."""
    trigger = 4

    @boolean_simulation
    def main(value, c_value):
        a = QuantumFloat(3)
        a[:] = value
        with control(measure(a) == trigger):
            c = QuantumFloat(2)
            c[:] = c_value
            with control(measure(c) == 1):
                x(a[0])
                x(c[0])
            c.delete()
        return measure(a)

    # Only value 4 enters the outer block, and only c_value 1 the inner one.
    cases = [(3, 0), (3, 1), (4, 0), (4, 1)]
    assert [int(main(value, c_value)) for value, c_value in cases] == [3, 3, 4, 5]


def test_jasp_classical_control_carry_value_raises():
    """A variable created in a classically controlled block cannot be used after it."""

    def main(value):
        a = QuantumFloat(3)
        a[:] = value
        created = None
        with control(measure(a) == 0):
            created = QuantumFloat(2)
        return measure(created)

    with pytest.raises(Exception, match="Found ClControlEnvironment with carry value"):
        make_jaspr(main)(1)


def test_jasp_inverted_classical_control():
    """Inverting a classically controlled block inverts its body."""

    def cyclic_shift(qf):
        swap(qf[0], qf[1])
        swap(qf[1], qf[2])

    @boolean_simulation
    def main(flag_value):
        flag = QuantumBool()
        flag[:] = flag_value
        flag_result = measure(flag)
        forward, backward = QuantumFloat(3), QuantumFloat(3)
        x(forward[0])
        x(backward[0])
        with control(flag_result):
            cyclic_shift(forward)
        with invert(), control(flag_result):
            cyclic_shift(backward)
        return measure(forward), measure(backward)

    results = [[int(result) for result in main(flag_value)] for flag_value in (False, True)]
    # The shift moves qubit 0 to qubit 2, and its inverse moves it to qubit 1.
    assert results == [[0b001, 0b001], [0b100, 0b010]]


def test_jasp_python_bool_control():
    """A Python bool also controls a block in Jasp mode."""

    @boolean_simulation
    def main():
        target = QuantumBool()
        with control(True):
            x(target)
        return measure(target)

    assert bool(main())


@pytest.mark.parametrize("ctrl_state", [1, 0])
def test_jasp_quantum_invert_single_control(ctrl_state):
    """invert=True with a single control qubit activates the block for its other state."""

    @boolean_simulation
    def main(n):
        a = QuantumFloat(2)
        a[:] = n
        target = QuantumBool()
        with control(a[0], ctrl_state=ctrl_state, invert=True):
            x(target)
        return measure(target)

    expected = [(value & 1) != ctrl_state for value in range(4)]
    assert [bool(main(value)) for value in range(4)] == expected


@pytest.mark.parametrize("ctrl_state", ["10", 3])
def test_jasp_quantum_invert_several_controls(ctrl_state):
    """invert=True with several controls negates the whole condition and restores the controls."""

    @boolean_simulation
    def main(n):
        a = QuantumFloat(2)
        a[:] = n
        target = QuantumBool()
        with control([a[0], a[1]], ctrl_state=ctrl_state, invert=True):
            x(target)
        return measure(target), measure(a)

    # "10" means qubit 0 in 1 and qubit 1 in 0 (a == 1), the integer 3 means a == 3.
    activating_value = 1 if ctrl_state == "10" else ctrl_state
    for value in range(4):
        fired, controls = main(value)
        assert bool(fired) == (value != activating_value)
        assert int(controls) == value
