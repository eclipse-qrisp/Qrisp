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

"""Tests for control in static mode."""

import jax.numpy as jnp
import numpy as np
import pytest

from qrisp import (
    ClControlEnvironment,
    ControlEnvironment,
    QuantumBool,
    QuantumFloat,
    QuantumVariable,
    control,
    x,
)

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


# Dispatch


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


# Argument forwarding


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


# Errors


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


# Semantics


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
