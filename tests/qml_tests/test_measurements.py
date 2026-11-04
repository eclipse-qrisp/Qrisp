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

import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from qrisp import QuantumFloat, cx, ry
from qrisp.jasp import jaspify
from qrisp.operators import X, Z
from qrisp.qml import Ansatz, Encoding, expectation, hardware_efficient_size, parameter_shift

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


@pytest.mark.parametrize("container", [list, tuple, np.array, iter], ids=["list", "tuple", "ndarray", "iterator"])
def test_expectation_of_a_sequence_keeps_the_entries_separate_and_in_order(container):
    """
    Each observable is reported on its own, in the order given, whatever kind
    of sequence holds them. The two qubits are in opposite states here, which
    makes the expected result [-1, +1]. A summed operator would report a single
    0, and a reversed order would give [+1, -1]. Observables were once
    identified by isinstance against list and tuple, which let an ndarray or a
    generator fall through to the single operator branch.
    """
    observables = container([Z(0), Z(1)])

    result = expectation(state_prep, observables, precision=PRECISION)([np.pi, 0.0])

    assert np.allclose(result, [-1.0, 1.0], atol=TOLERANCE)


def test_expectation_handles_non_commuting_observables():
    """
    Observables that cannot be measured in one circuit are estimated
    separately, which means they may be combined freely.
    """
    result = expectation(state_prep, [Z(0), X(0)], precision=PRECISION)([0.0, 0.0])

    assert np.allclose(result, [1.0, 0.0], atol=TOLERANCE)


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


@pytest.mark.parametrize("x, w", [(0.3, 0.4), (2.5, -0.4)])
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


def test_parameter_shift_differentiates_a_tuple_of_arguments_under_tracing():
    """
    With argnums=(0, 1) the result holds one derivative array per argument,
    also inside a traced program. Both angles enter the same rotation, which
    makes both derivatives -sin(x + w).
    """
    x, w = 0.3, 0.4

    value, (dx, dw) = traced(parameter_shift(rotated_qubit, Z(0), precision=PRECISION, argnums=(0, 1)))(
        np.array([x]), np.array([w])
    )

    assert np.isclose(value, np.cos(x + w), atol=TOLERANCE)
    assert np.allclose([dx[0], dw[0]], -np.sin(x + w), atol=TOLERANCE)


def test_parameter_shift_gives_the_same_derivatives_traced_and_untraced():
    """
    Check that the shifts mapped under tracing give the derivatives of the loop
    that runs outside of tracing, on generic signed inputs with two observables.
    """
    encoding = Encoding.angle(n_features=2)
    ansatz = Ansatz.hardware_efficient(num_qubits=2, layers=1, rotations=("y", "z"))

    def circuit(x, w):
        qf = QuantumFloat(2)
        encoding.apply(qf, x)
        ansatz.apply(qf, w)
        return qf

    x = np.array([0.7, -1.3])
    w = np.array([0.4, -2.1, 1.7, 0.3, -0.9, 2.6, -0.5, 1.1])
    shift = parameter_shift(circuit, [Z(0), Z(1)], precision=PRECISION, argnums=(0, 1))

    eager_values, (eager_dx, eager_dw) = direct(shift)(x, w)
    traced_values, (traced_dx, traced_dw) = traced(shift)(x, w)

    assert np.shape(traced_dx) == (2, 2)
    assert np.shape(traced_dw) == (8, 2)
    assert np.allclose(traced_values, eager_values, atol=TOLERANCE)
    assert np.allclose(traced_dx, eager_dx, atol=TOLERANCE)
    assert np.allclose(traced_dw, eager_dw, atol=TOLERANCE)


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


def test_expectation_feeds_classical_code_under_tracing():
    """
    Inside jaspify the readout is an ordinary Jax value, which a classical
    layer can use directly. Generic angles keep <Z_i> = cos(x_i) away from +1
    and -1, which makes the result depend on the estimated values themselves.
    """
    x = np.array([0.7, 2.1])
    w_out = np.array([1.0, 2.0])
    quantum_block = expectation(state_prep, [Z(0), Z(1)], precision=PRECISION)

    @jaspify(terminal_sampling=True)
    def model(x, w_out):
        return jnp.tanh(jnp.dot(quantum_block(x), w_out))

    result = model(jnp.array(x), jnp.array(w_out))

    assert np.ndim(result) == 0
    assert np.isclose(result, np.tanh(np.cos(x) @ w_out), atol=TOLERANCE)


def test_a_flat_vector_drives_a_hybrid_model_under_tracing():
    """
    The optimizers take one flat vector, while a hybrid model has named groups
    of parameters. The tutorials convert between the two with ravel_pytree and
    call the unravel inside the traced function, once per iteration. The
    groups are given distinct values, which makes a swapped group change the
    result.
    """
    encoding = Encoding.angle(n_features=2)
    ansatz = Ansatz.hardware_efficient(num_qubits=2, layers=1)

    def prepare(x, w):
        qv = QuantumFloat(2)
        encoding.apply(qv, x)
        ansatz.apply(qv, w)
        return qv

    # The parameters are zero, which makes both expectations exactly +1 or -1
    # and keeps the sampling noise far from the tolerance. A coarse precision
    # is therefore enough.
    quantum_block = expectation(prepare, [Z(0), Z(1)], precision=0.5)

    # w_in leaves the angles untouched, and the encoded row [0, pi] gives
    # [+1, -1] after the CX. w_out weights the two entries differently, which
    # makes the result depend on which group reached which layer.
    parameters = {
        "w_in": jnp.ones(2),
        "theta": jnp.zeros(hardware_efficient_size(2, 1)),
        "w_out": jnp.array([2.0, 3.0]),
        "b_out": jnp.array(0.5),
    }
    flat, unravel = ravel_pytree(parameters)

    @jaspify
    def model(flat_parameters):
        p = unravel(flat_parameters)
        q = quantum_block(jnp.array([0.0, np.pi]) * p["w_in"], p["theta"])
        return jnp.tanh(jnp.dot(q, p["w_out"]) + p["b_out"])

    result = model(flat)

    # [1, -1] . [2, 3] + 0.5 = -0.5, and the model returns tanh(-0.5).
    assert np.ndim(result) == 0
    assert np.isclose(result, np.tanh(-0.5), atol=TOLERANCE)
