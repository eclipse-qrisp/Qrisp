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

"""Tests for control in Jasp mode."""

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


# Dispatch


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


# Compilation


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


# Semantics


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
