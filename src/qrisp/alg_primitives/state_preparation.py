"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


EPSILON = 1e-12


def _rot_params_from_state(vec: jnp.ndarray) -> tuple:
    """
    Computes the rotation angles to prepare a single qubit state,
    where the amplitude of the |0> basis state is real and non-negative.

    Specifically, it computes the angles ``theta``, ``phi``, and ``lambda``
    such that applying the U3 gate with these angles to the |0> state results in the desired state:

    |0> → a|0> + b|1>, with a real ≥ 0.

    Parameters
    ----------
    vec : jnp.ndarray
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
    theta = 2.0 * jnp.arccos(a)
    phi = jnp.where(jnp.abs(b) > EPSILON, jnp.angle(b), 0.0)
    lam = 0.0
    return theta, phi, lam


def _normalize_with_phase(
    v: jnp.ndarray, acc: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Normalizes a given vector and adjusts its phase.

    The phase of the first element of the vector is removed and added to the accumulated phase.
    The vector is normalized to have a unit norm and the first element is ensured to be real and non-negative.

    Parameters
    ----------
    v : jnp.ndarray
        The child vector to normalize.
    acc : jnp.ndarray
        The accumulated phase from previous operations.

    Returns
    -------
    norm : jnp.ndarray
        The norm of the input vector.
    v_normalized : jnp.ndarray
        The normalized vector with adjusted phase.
    updated_acc : jnp.ndarray
        The updated accumulated phase.
    """

    norm = jnp.linalg.norm(v)
    alpha = jnp.where(norm > EPSILON, jnp.angle(v[0]), 0.0)
    v_normalized = jnp.where(
        norm > EPSILON,
        v / (norm * jnp.exp(1j * alpha)),
        v,
    )
    updated_acc = acc + alpha
    return norm, v_normalized, updated_acc


def _compute_thetas(
    vec: jnp.ndarray, acc: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    For a given input vector, this function computes the rotation angles
    needed to prepare the corresponding two-qubit state, normalizes its child vectors,
    and updates the accumulated phases for each child vector.

    Parameters
    ----------
    vec : jnp.ndarray
        A complex vector representing the current vector to process.
    acc : jnp.ndarray
        The accumulated phase from previous operations.


    Returns
    -------
    theta : jnp.ndarray
        The angle (scalar array) for the ry rotation gate.
    subvecs : jnp.ndarray
        A 2D array where each row corresponds to a normalized subvector.
    acc_phases : jnp.ndarray
        A 1D array containing the updated accumulated phases for each subvector.

    """

    len = vec.shape[0]
    half = len // 2

    v0 = vec[:half]
    v1 = vec[half:]

    n0, v0n, acc0 = _normalize_with_phase(v0, acc)
    _, v1n, acc1 = _normalize_with_phase(v1, acc)

    theta = 2.0 * jnp.arccos(jnp.minimum(1.0, n0))  # shape ()
    subvecs = jnp.stack([v0n, v1n], axis=0)  # shape (2, half)
    acc_phases = jnp.stack([acc0, acc1], axis=0)  # shape (2,)

    return theta, subvecs, acc_phases


def _compute_u3_params(
    qubit_vec: jnp.ndarray, acc: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    For a given one-qubit vector, this function computes the U3 gate parameters needed
    to prepare the corresponding state, normalizes the vector, and updates the accumulated phase.

    Parameters
    ----------
    qubit_vec : jnp.ndarray
        A complex vector representing a one-qubit state.
    acc : jnp.ndarray
        The accumulated phase from previous operations.

    Returns
    -------
    u_params : jnp.ndarray
        A 1D array containing the rotation angles (theta, phi, lambda) for the U3 gate.
    total_phase : jnp.ndarray
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
# - `glob_phases` has shape (2^(n-1),), and contains the global phase for each leaf node.
#
#    glob_phases = Array[phase_leaf0, phase_leaf1, ..., phase_leaf_{2^(n-1)-1}]
#
def _preprocess(
    target_array: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    This preprocessing function returns three data structures needed for state preparation.

    Parameters
    ----------
    target_array : jnp.ndarray
        A complex vector representing the target state to prepare.

    Returns
    -------
    thetas : jnp.ndarray
        A 2D array containing the ry rotation angles for each layer.
    u_params : jnp.ndarray
        A 2D array containing the U3 parameters for each leaf node.
    glob_phases : jnp.ndarray
        A 1D array containing the global phase for each leaf node.

    """

    n = int(np.log2(target_array.shape[0]))

    if n == 1:
        _, vec_n, a0_phase = _normalize_with_phase(target_array, 0.0)
        theta, phi, lam = _rot_params_from_state(vec_n)

        thetas = jnp.zeros((0, 1))
        u_params = jnp.stack([theta, phi, lam])[None, :]
        glob_phases = (a0_phase)[None]
        return thetas, u_params, glob_phases

    # Data structures to return
    thetas = jnp.zeros((n - 1, 1 << (n - 1)))
    u_params = jnp.zeros((1 << (n - 1), 3))
    glob_phases = jnp.zeros((1 << (n - 1),))

    # Data structures used during the computation (reshaped at each layer)
    subvecs = target_array[jnp.newaxis, :]
    acc_phases = jnp.zeros((1,))

    for l in range(n):

        num_nodes = 1 << l
        sub_len = 1 << (n - l)

        if sub_len == 2:
            u_params_vec, glob_phases_vec = jax.vmap(_compute_u3_params)(
                subvecs, acc_phases
            )
            u_params = u_params.at[:num_nodes, :].set(u_params_vec)
            glob_phases = glob_phases.at[:num_nodes].set(glob_phases_vec)
            break

        theta_vec, subvecs, acc_phases = jax.vmap(_compute_thetas)(subvecs, acc_phases)

        thetas = thetas.at[l, :num_nodes].set(theta_vec)

        subvecs = subvecs.reshape((2 * num_nodes, sub_len // 2))
        acc_phases = acc_phases.reshape((2 * num_nodes,))

    return thetas, u_params, glob_phases


def state_preparation(qv, target_array, method: str = "auto") -> None:
    """
    TODO: add docstring
    """

    # These imports are here to avoid circular dependencies
    from qrisp import gphase, qswitch, ry, u3
    from qrisp.misc.utility import jasp_bit_reverse

    target_array = jnp.asarray(target_array, dtype=jnp.complex128)
    # n is static, so we can use normal numpy here
    n = int(np.log2(target_array.shape[0]))

    thetas, u_params, glob_phases = _preprocess(target_array)

    def make_case_fn(layer_size: int, is_final: bool = False) -> Callable:
        """Create a case function for qswitch at a given layer."""

        def case_fn(i, qb):
            rev_idx = jasp_bit_reverse(i, layer_size)
            if is_final:
                theta_i, phi_i, lam_i = u_params[rev_idx]
                u3(theta_i, phi_i, lam_i, qb)
                gphase(glob_phases[rev_idx], qb)
            else:
                ry(thetas[layer_size][rev_idx], qb)

        return case_fn

    if n == 1:
        theta, phi, lam = u_params[0]
        u3(theta, phi, lam, qv[0])
        gphase(glob_phases[0], qv[0])
        return

    ry(thetas[0][0], qv[0])

    for layer_size in range(1, n - 1):

        qswitch(
            operand=qv[layer_size],
            case=qv[:layer_size],
            case_function=make_case_fn(layer_size),
            method=method,
        )

    qswitch(
        operand=qv[n - 1],
        case=qv[: n - 1],
        case_function=make_case_fn(n - 1, is_final=True),
        method=method,
    )
