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

"""Tests the qml expectation and parameter_shift functions on circuits with known values."""

import numpy as np
import pytest

from qrisp import QuantumFloat, cx, ry
from qrisp.jasp import jaspify
from qrisp.operators import X, Z
from qrisp.qml import Encoding, expectation, parameter_shift

# Coarse enough to keep the suite fast, tight enough that a wrong sign or a
# swapped entry cannot pass.
PRECISION = 0.02
TOLERANCE = 0.1


def state_prep(x):
    qf = QuantumFloat(2)
    Encoding.angle(n_features=2).apply(qf, x)
    return qf


@pytest.mark.parametrize(
    "x, observable, expected",
    [
        ([0.0, 0.0], Z(0), 1.0),
        ([np.pi, 0.0], Z(0), -1.0),
        ([np.pi, 0.0], Z(0) * Z(1), -1.0),
        ([0.0, 0.0], X(0), 0.0),
    ],
)
def test_expectation_of_a_single_observable_is_a_scalar(x, observable, expected):
    """
    A single observable is passed through to the operator, which yields one
    number rather than an array.
    """
    result = expectation(state_prep, observable, precision=PRECISION)(x)

    assert np.ndim(result) == 0
    assert np.isclose(result, expected, atol=TOLERANCE)


def test_expectation_of_a_sequence_keeps_the_entries_separate():
    """
    Each observable is reported on its own. The two qubits are in opposite
    states here, which a summed operator would report as a single 0.
    """
    result = expectation(state_prep, [Z(0), Z(1)], precision=PRECISION)([np.pi, 0.0])

    assert np.allclose(result, [-1.0, 1.0], atol=TOLERANCE)


def test_expectation_follows_the_order_of_the_observables():
    """
    The entries are aligned with the observables as given, and reversing the
    sequence reverses the result.
    """
    x = [np.pi, 0.0]

    forward = expectation(state_prep, [Z(0), Z(1)], precision=PRECISION)(x)
    reverse = expectation(state_prep, [Z(1), Z(0)], precision=PRECISION)(x)

    assert np.allclose(forward, reverse[::-1], atol=TOLERANCE)


def test_expectation_handles_non_commuting_observables():
    """
    Observables that cannot be measured in one circuit are estimated
    separately, which means they may be combined freely.
    """
    result = expectation(state_prep, [Z(0), X(0)], precision=PRECISION)([0.0, 0.0])

    assert np.allclose(result, [1.0, 0.0], atol=TOLERANCE)


def test_expectation_accepts_a_tuple_of_observables():
    """Any sequence is accepted, not only a list."""
    result = expectation(state_prep, (Z(0), Z(1)), precision=PRECISION)([np.pi, 0.0])

    assert np.allclose(result, [-1.0, 1.0], atol=TOLERANCE)


@pytest.mark.parametrize("container", [list, tuple, np.array, iter], ids=["list", "tuple", "ndarray", "iterator"])
def test_expectation_accepts_any_sequence_of_observables(container):
    """
    Observables were identified by isinstance against list and tuple, which
    let an ndarray or a generator fall through to the single operator branch
    and fail with an AttributeError naming a numpy internal.
    """
    observables = container([Z(0), Z(1)])

    def state_prep():
        return QuantumFloat(2)

    # The register stays in |00>, and both qubits are measured as +1.
    assert np.allclose(expectation(state_prep, observables)(), [1.0, 1.0])


def test_expectation_rejects_an_empty_sequence():
    """
    An empty sequence returned an empty array from a function that samples
    nothing, which reads as a successful measurement of no observables rather
    than as the mistake it is.
    """

    def state_prep():
        return QuantumFloat(2)

    with pytest.raises(ValueError, match="empty sequence"):
        expectation(state_prep, [])


def direct(function):
    return function


def traced(function):
    return jaspify(function, terminal_sampling=True)


def rotated_qubit(x, w):
    qf = QuantumFloat(1)
    ry(x[0], qf[0])
    ry(w[0], qf[0])
    return qf


@pytest.mark.parametrize("x, w", [(0.3, 0.4), (1.0, 2.0), (2.5, -0.4)])
def test_parameter_shift_gives_the_derivative_of_a_rotation(x, w):
    """
    Two rotations about the same axis give <Z> = cos(x + w), whose derivative
    in either angle is -sin(x + w). A wrong sign, a missing factor of one half
    or swapped shifts each move the result by more than the tolerance.
    """
    value, (dx, dw) = parameter_shift(rotated_qubit, Z(0), precision=PRECISION, argnums=(0, 1))(
        np.array([x]), np.array([w])
    )

    assert np.isclose(value, np.cos(x + w), atol=TOLERANCE)
    assert np.allclose([dx[0], dw[0]], -np.sin(x + w), atol=TOLERANCE)


def test_parameter_shift_accepts_an_integer_argument():
    """
    An integer argument shifted in place keeps its dtype, which truncates the
    shift of pi/2 to 1. At x + w = 2 that gives -sin(2) sin(1), about -0.765,
    instead of -sin(2), about -0.909, a gap that the tighter precision
    resolves.
    """
    _, (dx, dw) = parameter_shift(rotated_qubit, Z(0), precision=0.005, argnums=(0, 1))(np.array([2]), np.array([0]))

    assert np.allclose([dx[0], dw[0]], -np.sin(2), atol=0.05)


def entangled_pair(x):
    qf = QuantumFloat(2)
    ry(x[0], qf[0])
    ry(x[1], qf[1])
    cx(qf[0], qf[1])
    return qf


@pytest.mark.parametrize("run", [direct, traced])
def test_parameter_shift_puts_entries_in_rows_and_observables_in_columns(run):
    """
    After the CX, <Z0> = cos x0 and <Z1> = cos x0 cos x1. Every entry of the
    derivative array differs from its transposed partner. A transposed result
    therefore fails, on the traced path as well as without tracing.
    """
    x = np.array([0.6, 1.2])

    values, derivatives = run(parameter_shift(entangled_pair, [Z(0), Z(1)], precision=PRECISION))(x)

    expected = [
        [-np.sin(x[0]), -np.sin(x[0]) * np.cos(x[1])],
        [0.0, -np.cos(x[0]) * np.sin(x[1])],
    ]
    assert np.allclose(values, [np.cos(x[0]), np.cos(x[0]) * np.cos(x[1])], atol=TOLERANCE)
    assert np.shape(derivatives) == (2, 2)
    assert np.allclose(derivatives, expected, atol=TOLERANCE)


def separate_qubits(x, w):
    qf = QuantumFloat(2)
    ry(x[0], qf[0])
    ry(w[0], qf[1])
    return qf


@pytest.mark.parametrize("argnums, expected", [(0, 0.0), (1, -np.sin(0.8))])
def test_parameter_shift_differentiates_the_argument_at_argnums(argnums, expected):
    """
    The observable depends on w alone. Its derivative is therefore zero in x
    and -sin w in w. Differentiating the wrong argument swaps the two. A single
    position returns one array rather than a tuple of them.
    """
    _, derivatives = parameter_shift(separate_qubits, Z(1), precision=PRECISION, argnums=argnums)(
        np.array([0.3]), np.array([0.8])
    )

    assert not isinstance(derivatives, tuple)
    assert np.isclose(derivatives[0], expected, atol=TOLERANCE)


@pytest.mark.parametrize("run", [direct, traced])
def test_parameter_shift_rejects_an_argument_that_is_not_one_dimensional(run):
    """
    A matrix of angles has no single entry to shift per row of the result.
    The shape is static under tracing, and the check also runs there.
    """

    def state_prep(x):
        qf = QuantumFloat(2)
        ry(x[0, 0], qf[0])
        ry(x[1, 1], qf[1])
        return qf

    with pytest.raises(ValueError, match="one-dimensional"):
        run(parameter_shift(state_prep, Z(0), precision=PRECISION))(np.zeros((2, 2)))
