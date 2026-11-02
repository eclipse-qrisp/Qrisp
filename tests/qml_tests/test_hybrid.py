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

"""Tests classical layers composed around a qml quantum block, with and without tracing."""

import numpy as np
import pytest

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from qrisp import QuantumFloat
from qrisp.operators import Z
from qrisp.qml import Ansatz, Encoding, expectation, hardware_efficient_size
from qrisp.jasp import jaspify

# Coarse enough to keep the suite fast, tight enough that a wrong sign or a
# swapped entry cannot pass.
PRECISION = 0.02
TOLERANCE = 0.1

NUM_QUBITS = 2
LAYERS = 1

# The parameters are held at zero throughout, which makes every rotation
# layer the identity and every expectation value exactly +1 or -1. The
# entangling CX is still applied, which means the quantum block is not the
# identity. It maps the encoded basis state |q0 q1> to |q0, q0 XOR q1>.
#
#   encoded angles     state       after CX     <Z0>  <Z1>
#   [0,  0]            |00>        |00>          +1    +1
#   [pi, 0]            |10>        |11>          -1    -1
#   [0,  pi]           |01>        |01>          +1    -1
#   [pi, pi]           |11>        |10>          -1    +1
#
# These were derived by running the block, not by inspection. The CX makes the
# naive guess ([pi,0] -> [-1,+1]) wrong.
QUANTUM_TRUTH_TABLE = [
    ([0.0, 0.0], [1.0, 1.0]),
    ([np.pi, 0.0], [-1.0, -1.0]),
    ([0.0, np.pi], [1.0, -1.0]),
    ([np.pi, np.pi], [-1.0, 1.0]),
]


def _quantum_block():
    """A two-observable readout, which is what a classical head consumes."""
    encoding = Encoding.angle(n_features=NUM_QUBITS)
    ansatz = Ansatz.hardware_efficient(num_qubits=NUM_QUBITS, layers=LAYERS)

    def state_prep(x, w):
        qv = QuantumFloat(NUM_QUBITS)
        encoding.apply(qv, x)
        ansatz.apply(qv, w)
        return qv

    return expectation(state_prep, [Z(0), Z(1)], precision=PRECISION)


def _zero_parameters():
    return jnp.zeros(hardware_efficient_size(NUM_QUBITS, LAYERS))


@pytest.mark.parametrize("x, expected", QUANTUM_TRUTH_TABLE)
def test_quantum_block_returns_one_entry_per_observable(x, expected):
    """
    The readout a classical head consumes is a vector, not a scalar. A summed
    operator would collapse [pi, pi] to a single 0 and lose both entries.
    """
    result = _quantum_block()(jnp.array(x), _zero_parameters())

    assert np.shape(result) == (NUM_QUBITS,)
    assert np.allclose(result, expected, atol=TOLERANCE)


def test_classical_layer_before_the_quantum_block():
    """
    A classical layer feeding the encoding lets a wide feature vector reach a
    narrow register. The scaling it applies has to survive into the encoded
    angles.
    """
    quantum_block = _quantum_block()

    def model(raw, w_in, w):
        return quantum_block(raw * w_in, w)

    # raw * w_in puts pi on both qubits, which is the [pi, pi] row above.
    result = model(jnp.array([1.0, 1.0]), jnp.array([np.pi, np.pi]), _zero_parameters())

    assert np.allclose(result, [-1.0, 1.0], atol=TOLERANCE)


def test_classical_layer_after_the_quantum_block():
    """
    The vector readout feeds an ordinary jnp expression. A model can therefore
    put a trainable head on top of the quantum block.
    """
    quantum_block = _quantum_block()

    def model(x, w, w_out, b_out):
        return jnp.tanh(jnp.dot(quantum_block(x, w), w_out) + b_out)

    # The [0, 0] row gives [+1, +1], and the head computes tanh(1 + 1 + 0).
    result = model(
        jnp.array([0.0, 0.0]),
        _zero_parameters(),
        jnp.array([1.0, 1.0]),
        0.0,
    )

    assert np.isclose(result, np.tanh(2.0), atol=TOLERANCE)


def test_classical_quantum_classical_composes_end_to_end():
    """
    The full sandwich, which is the composition a hybrid model is built from.
    Each stage is checked above on its own. This pins that they chain.
    """
    quantum_block = _quantum_block()

    def model(raw, w_in, w, w_out, b_out):
        h = raw * w_in
        q = quantum_block(h, w)
        return jnp.tanh(jnp.dot(q, w_out) + b_out)

    # raw * w_in gives [pi, 0] -> [-1, -1] -> tanh(-1 - 1 + 0).
    result = model(
        jnp.array([1.0, 0.0]),
        jnp.array([np.pi, np.pi]),
        _zero_parameters(),
        jnp.array([1.0, 1.0]),
        0.0,
    )

    assert np.isclose(result, np.tanh(-2.0), atol=TOLERANCE)


def test_quantum_block_output_is_a_jax_array():
    """
    A hybrid model is only composable if the readout is an ordinary Jax value.
    A raw ``measure`` returns a Clbit, which is not a Jax type and breaks any
    transformation applied to the surrounding function.
    """
    result = _quantum_block()(jnp.array([0.0, 0.0]), _zero_parameters())

    assert isinstance(result, jnp.ndarray)
    assert jnp.issubdtype(result.dtype, jnp.floating)


def test_hybrid_composition_traces():
    """
    Composition has to survive tracing, because a training run uses it. The
    classical parts are jnp expressions either side of the quantum block, and
    the whole model goes through one ``jaspify``.
    """
    quantum_block = _quantum_block()

    @jaspify
    def model(raw, w_in, w, w_out):
        q = quantum_block(raw * w_in, w)
        return jnp.tanh(jnp.dot(q, w_out))

    result = model(
        jnp.array([1.0, 1.0]),
        jnp.array([np.pi, np.pi]),
        _zero_parameters(),
        jnp.array([1.0, 1.0]),
    )

    # The [pi, pi] row gives [-1, +1], which sums to 0.
    assert np.isclose(result, 0.0, atol=TOLERANCE)


def _hybrid_parameters():
    """
    The parameters of a hybrid model in named groups, with a classical layer
    either side of the quantum block. Since ``b_out`` is a scalar, the
    roundtrip has to carry a 0-d entry as well as vectors.
    """
    ansatz = Ansatz.hardware_efficient(num_qubits=NUM_QUBITS, layers=LAYERS)
    return {
        "w_in": jnp.ones(NUM_QUBITS),
        "theta": jnp.zeros(ansatz.num_parameters),
        "w_out": jnp.ones(NUM_QUBITS),
        "b_out": jnp.array(0.0),
    }


def test_named_parameter_groups_survive_the_flat_roundtrip():
    """
    The optimizers take one flat vector while a hybrid model has named groups
    of parameters, and something has to convert between the two. Jax already
    does it, and slicing the flat vector by hand instead is the mistake this
    pins against.
    An off-by-one there trains the model on the wrong numbers and raises
    nothing.
    """
    parameters = _hybrid_parameters()
    ansatz = Ansatz.hardware_efficient(num_qubits=NUM_QUBITS, layers=LAYERS)

    flat, unravel = ravel_pytree(parameters)

    # 2 in + num_parameters + 2 out + 1 bias.
    assert flat.shape == (2 * NUM_QUBITS + ansatz.num_parameters + 1,)

    restored = unravel(flat)

    assert set(restored) == set(parameters)
    for name, value in parameters.items():
        assert restored[name].shape == np.shape(value)
        assert np.allclose(restored[name], value)


def test_a_flat_vector_drives_a_hybrid_model_under_tracing():
    """
    The optimizer calls the unravel inside the traced function, once per
    iteration, which is why the unravel has to work there.

    With every group set to ones, the model returns the same number whichever
    way two groups are exchanged, and the test would pass against the very
    mistake it is meant to catch. The groups are therefore given distinct
    values, which makes a swapped group change the result.
    """
    encoding = Encoding.angle(n_features=NUM_QUBITS)
    ansatz = Ansatz.hardware_efficient(num_qubits=NUM_QUBITS, layers=LAYERS)

    def state_prep(x, w):
        qv = QuantumFloat(NUM_QUBITS)
        encoding.apply(qv, x)
        ansatz.apply(qv, w)
        return qv

    # The parameters are zero, which makes both expectations exactly +-1 and
    # keeps the sampling noise far away from the tolerance. A coarse precision
    # is enough, and tightening it only makes the suite slower.
    quantum_block = expectation(state_prep, [Z(0), Z(1)], precision=0.5)

    # w_in leaves the angles untouched, and the encoded row is [0, pi] -> [1, -1].
    # w_out weights those two entries differently, which makes the result
    # depend on which group reached which layer.
    parameters = {
        "w_in": jnp.ones(NUM_QUBITS),
        "theta": jnp.zeros(hardware_efficient_size(NUM_QUBITS, LAYERS)),
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

    # [1, -1] . [2, 3] + 0.5 = -0.5, and the model returns tanh(-0.5). Exchanging
    # any two groups moves this number, which an all-ones fixture would not.
    assert np.ndim(result) == 0
    assert np.isclose(result, np.tanh(-0.5), atol=TOLERANCE)
