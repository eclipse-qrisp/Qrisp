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

"""Prepares an arbitrary quantum state via a binary-tree decomposition using the q_switch primitive."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.typing import ArrayLike

from qrisp.misc.utility import EPSILON

if TYPE_CHECKING:
    from qrisp.core import QuantumVariable
    from qrisp.typing import NDArrayLike


# This is required in the qswitch-based state preparation, where it is called
# inside jrange loops, because DynamicQubitArray does not support reverse iteration.
def bit_reverse(i, width):
    """Jasp-compatible bit-reversal function.

    Interprets ``i`` as a ``width``-bit binary integer
    and returns the decimal integer corresponding to the bit-reversal of ``i``.
    The maximum supported width is 64 bits.

    This function can be used in Jasp-mode and within a `jrange` loop.
    It does not use any Python or Jax control flow, but only Jax array operations.

    Parameters
    ----------
    i : jnp.ndarray
        Index to be bit-reversed.
    width : jnp.ndarray
        Bit-width for the reversal (scalar array).

    Returns
    -------
    jnp.ndarray
        Bit-reversed index.


    Examples
    --------
    For ``i=5`` and ``width=3``, the binary representation
    of ``5`` is ``101``, and its bit-reversal is (again) ``101``, which is ``5`` in decimal.

    >>> from qrisp.alg_primitives.state_preparation.qswitch_state_preparation import bit_reverse
    >>> bit_reverse(5, 3)
    5

    For ``i=3`` and ``width=4``, the binary representation
    of ``3`` is ``0011``, and its bit-reversal is ``1100``, which is ``12`` in decimal.

    >>> bit_reverse(3, 4)
    12

    """
    i = jnp.asarray(i, dtype=jnp.uint64)
    width = jnp.asarray(width, dtype=jnp.uint64)

    m1 = jnp.uint64(0x5555555555555555)
    m2 = jnp.uint64(0x3333333333333333)
    m3 = jnp.uint64(0x0F0F0F0F0F0F0F0F)
    m4 = jnp.uint64(0x00FF00FF00FF00FF)
    m5 = jnp.uint64(0x0000FFFF0000FFFF)
    m6 = jnp.uint64(0x00000000FFFFFFFF)

    i = ((i >> 1) & m1) | ((i & m1) << 1)
    i = ((i >> 2) & m2) | ((i & m2) << 2)
    i = ((i >> 4) & m3) | ((i & m3) << 4)
    i = ((i >> 8) & m4) | ((i & m4) << 8)
    i = ((i >> 16) & m5) | ((i & m5) << 16)
    i = ((i >> 32) & m6) | ((i & m6) << 32)

    return i >> jnp.asarray(64, jnp.uint64) - width


def _bitrev_indices(n: ArrayLike) -> jax.Array:
    """Return array r where r[j] = bitreverse(j) over n bits."""
    idx = jnp.arange(1 << n, dtype=jnp.uint32)
    rev = jnp.zeros_like(idx)
    for k in range(n):
        rev = (rev << 1) | ((idx >> k) & 1)
    return rev


def swap_endianness(vec: ArrayLike, n: ArrayLike) -> jax.Array:
    """Convert between big-endian and little-endian qubit ordering.

    This transformation is its own inverse, so it works in both directions.

    Parameters
    ----------
    vec : ArrayLike
        The state vector to convert.
    n : ArrayLike
        The number of qubits.

    Returns
    -------
    jax.Array
        The state vector with reversed qubit ordering.

    """
    r = _bitrev_indices(n)
    return vec[r]


def _rot_params_from_state(
    vec: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes the rotation angles to prepare a single qubit state,
    where the amplitude of the |0> basis state is real and non-negative.

    Specifically, it computes the angles ``theta``, ``phi``, and ``lambda``
    such that applying the U3 gate with these angles to the |0> state results in the desired state:

    |0> → a|0> + b|1>, with a real ≥ 0.

    Parameters
    ----------
    vec : jax.Array
        A 2-dimensional complex vector representing a qubit state.

    Returns
    -------
    theta : float
        The rotation angle theta.

    phi : float
        The rotation angle phi.

    lam : float
        The rotation angle lambda.

    """
    a, b = vec
    # We know that a is real (and non-negative).
    # This step avoids warning about casting complex to real.
    a = jnp.clip(jnp.real(a), -1.0, 1.0)
    theta = 2.0 * jnp.arccos(a)
    phi = jnp.where(jnp.abs(b) > EPSILON, jnp.angle(b), 0.0)
    lam = jnp.float64(0.0)
    return theta, phi, lam


def _normalize_with_phase(v: jax.Array, acc: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Normalizes a given vector and adjusts its phase.

    The phase of the first element of the vector is removed and added to the accumulated phase.
    The vector is normalized to have a unit norm and the first element is ensured to be real and non-negative.

    Parameters
    ----------
    v : jax.Array
        The child vector to normalize.

    acc : jax.Array
        The accumulated phase from previous operations.

    Returns
    -------
    norm : jax.Array
        The norm of the input vector.

    v_normalized : jax.Array
        The normalized vector with adjusted phase.

    updated_acc : jax.Array
        The updated accumulated phase.

    """
    norm = jnp.linalg.norm(v)

    def branch_nonzero(_):
        alpha = jnp.angle(v[0])
        v_normalized = v / (norm * jnp.exp(1j * alpha))
        return norm, v_normalized, acc + alpha

    def branch_zero(_):
        # If the norm is zero, we return a default normalized vector
        # with the first element real and non-negative.
        v0 = jnp.where(jnp.real(v[0]) < 0, -v[0], v[0])
        v_adj = v.at[0].set(v0)
        return norm, v_adj, acc

    return lax.cond(
        norm > EPSILON,
        lambda _: branch_nonzero(None),
        lambda _: branch_zero(None),
        operand=None,
    )


def _compute_thetas(vec: jax.Array, acc: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """For a given input vector, this function computes the rotation angles
    needed for the uniformly controlled RY at this tree layer, normalizes its child vectors,
    and updates the accumulated phases for each child vector.

    Parameters
    ----------
    vec : jax.Array
        A complex vector representing the current vector to process.

    acc : jax.Array
        The accumulated phase from previous operations.


    Returns
    -------
    theta : jax.Array
        The angle (scalar array) for the ry rotation gate.

    subvecs : jax.Array
        A 2D array where each row corresponds to a normalized subvector.

    acc_phases : jax.Array
        A 1D array containing the updated accumulated phases for each subvector.

    """
    len_vec = vec.shape[0]
    half = len_vec // 2

    v0 = vec[:half]
    v1 = vec[half:]

    n0, v0n, acc0 = _normalize_with_phase(v0, acc)
    _, v1n, acc1 = _normalize_with_phase(v1, acc)

    theta = 2.0 * jnp.arccos(jnp.minimum(1.0, n0))  # shape ()
    subvecs = jnp.stack([v0n, v1n], axis=0)  # shape (2, half)
    acc_phases = jnp.stack([acc0, acc1], axis=0)  # shape (2,)

    return theta, subvecs, acc_phases


def _compute_u3_params(qubit_vec: jax.Array, acc: jax.Array) -> tuple[jax.Array, jax.Array]:
    """For a given length-2 vector, this function computes the U3 gate parameters needed
    to prepare the corresponding state, normalizes the vector, and updates the accumulated phase.

    Parameters
    ----------
    qubit_vec : jax.Array
        A complex vector representing a one-qubit state.

    acc : jax.Array
        The accumulated phase from previous operations.

    Returns
    -------
    u_params : jax.Array
        A 1D array containing the rotation angles (theta, phi, lambda) for the U3 gate.

    total_phase : jax.Array
        The updated accumulated phase after processing the leaf subvector.

    """
    _, vec_n, total_phase = _normalize_with_phase(qubit_vec, acc)
    theta, phi, lam = _rot_params_from_state(vec_n)
    return jnp.array([theta, phi, lam]), total_phase


#  Here is the explanation of the data structures used in the state preparation algorithm:
#
# - `thetas`` has shape (n - 1, 2^(n-1)), and contains the ry rotation angles for each layer:
#
#    thetas = Array[[theta_0_0, 0, 0, 0, ..., 0],                                     # layer 0
#                   [theta_1_0, theta_1_1, 0, 0, ..., 0],                             # layer 1
#                   [theta_2_0, theta_2_1, theta_2_2, theta_2_3, ..., 0],             # layer 2
#                   ...
#                   [theta_{n-2}_0, theta_{n-2}_1, ..., theta_{n-2}_{2^(n-2)-1}, 0]]  # layer n-2
#
# - `u_params` has shape (2^(n-1), 3), and contains the U3 parameters for each leaf node.
#
#    u_params = Array[[theta_leaf0, phi_leaf0, lam_leaf0],                                 # leaf 0
#                   [theta_leaf1, phi_leaf1, lam_leaf1],                                   # leaf 1
#                   ...,
#                   [theta_leaf_{2^(n-1)-1}, phi_leaf_{2^(n-1)-1}, lam_leaf_{2^(n-1)-1}]]  # leaf 2^(n-1)-1
#
# - `phases` has shape (2^(n-1),), and contains the global phase for each leaf node.
#
#    phases = Array[phase_leaf0, phase_leaf1, ..., phase_leaf_{2^(n-1)-1}]
#
def _preprocess(
    target_array: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """This preprocessing function returns three data structures needed for state preparation.

    Parameters
    ----------
    target_array : jax.Array
        A complex vector representing the target state to prepare.

    Returns
    -------
    thetas : jax.Array
        A 2D array containing the ry rotation angles for each layer.
    u_params : jax.Array
        A 2D array containing the U3 parameters for each leaf node.
    phases : jax.Array
        A 1D array containing the global phase for each leaf node.

    """
    n = int(np.log2(target_array.shape[0]))
    max_nodes = 1 << (n - 1)

    # Data structures to return
    thetas = jnp.zeros((n - 1, max_nodes), dtype=jnp.float64)
    u_params = jnp.zeros((max_nodes, 3), dtype=jnp.float64)
    phases = jnp.zeros(max_nodes, dtype=jnp.float64)

    # Data structures used during the computation (reshaped at each layer)
    subvecs = target_array[jnp.newaxis, :]
    acc_phases = jnp.zeros((1,), dtype=jnp.float64)
    for l in range(n):
        num_nodes = 1 << l
        sub_len = 1 << (n - l)

        if sub_len == 2:
            u_params_vec, phases_vec = jax.vmap(_compute_u3_params)(subvecs, acc_phases)
            u_params = u_params.at[:num_nodes, :].set(u_params_vec)
            phases = phases.at[:num_nodes].set(phases_vec)
            break

        theta_vec, subvecs, acc_phases = jax.vmap(_compute_thetas)(subvecs, acc_phases)
        thetas = thetas.at[l, :num_nodes].set(theta_vec)
        subvecs = subvecs.reshape((2 * num_nodes, sub_len // 2))
        acc_phases = acc_phases.reshape((2 * num_nodes,))

    return thetas, u_params, phases


def prepare_qswitch(qv: QuantumVariable, target_array: NDArrayLike, big_endianness: bool = False) -> None:
    """Prepare the quantum state encoded in ``qv`` so that it matches the given
    ``target_array`` by constructing a binary-tree decomposition of the target
    amplitudes and applying a sequence of uniformly controlled rotations via
    the ``q_switch`` primitive.

    This routine implements a standard state-preparation algorithm based on
    recursively splitting the target statevector.
    The classical preprocessing stage extracts RY angles for internal tree nodes
    and U3 parameters for the leaf nodes.
    The quantum stage applies them using ``q_switch``, which replaces
    explicit multiplexers and conditionals in both static execution and Jasp mode.

    Parameters
    ----------
    qv : QuantumVariable
        The quantum variable representing the qubits to be prepared.

    target_array : NDArrayLike
        A normalized complex vector representing the target state to prepare.

    big_endianness : bool, optional
        If ``True``, indicates that the state preparation should use big-endian
        convention for the computational basis ordering.
        Default is ``False``, meaning little-endian convention is used.

    """
    # These imports are here to avoid circular dependencies
    from qrisp import gphase, ry, u3
    from qrisp.jasp.program_control.jrange_iterator import jrange
    from qrisp.jasp.program_control.prefix_control import q_switch
    from qrisp.jasp.tracing_logic import check_for_tracing_mode

    target_array_jax: jax.Array = jnp.asarray(target_array, dtype=jnp.complex128)
    target_array_jax = target_array_jax / jnp.linalg.norm(target_array_jax)
    # n is static (known at compile time), so we can use normal numpy here
    n = int(np.log2(target_array_jax.shape[0]))

    # The binary-tree preprocessing (_preprocess) and the q_switch traversal in this
    # function were originally implemented for a big-endian interpretation of the
    # target statevector indices (i.e. MSB-first splitting).
    #
    # However, we later on decided to consider little-endianness as the default
    # convention in Qrisp. That is, qv[0] is the least significant bit in the basis-state index.”
    #
    # Therefore, `big_endianness=False` indicates that we want to use little-endianness,
    # so we need to swap the endianness of the target_array before proceeding.
    if big_endianness is False:
        target_array_jax = swap_endianness(target_array_jax, n)

    # We could use jrange even in static mode, but this would add overhead.
    xrange = jrange if check_for_tracing_mode() else range

    thetas, u_params, phases = _preprocess(target_array_jax)

    def make_case_fn(layer_size, is_final: bool = False) -> Callable:
        """Create a case function for q_switch at a given layer."""

        def case_fn(i, qb):
            # NOTE: This bit-reversal is not about Qrisp's endianness.
            # It compensates for the order in which q_switch enumerates control patterns
            rev_idx = bit_reverse(i, layer_size)
            if is_final:
                theta_i, phi_i, lam_i = u_params[rev_idx]
                u3(theta_i, phi_i, lam_i, qb)
                gphase(phases[rev_idx], qb)
            else:
                ry(thetas[layer_size][rev_idx], qb)

        return case_fn

    if n == 1:
        theta, phi, lam = u_params[0]
        u3(theta, phi, lam, qv[0])
        gphase(phases[0], qv[0])
        return

    ry(thetas[0][0], qv[0])

    for layer_size in xrange(1, qv.size - 1):
        q_switch(
            qv[:layer_size],
            make_case_fn(layer_size),
            qv[layer_size],
        )

    q_switch(
        qv[: qv.size - 1],
        make_case_fn(qv.size - 1, is_final=True),
        qv[qv.size - 1],
    )
