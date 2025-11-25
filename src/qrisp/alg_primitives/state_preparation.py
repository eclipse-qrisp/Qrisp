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

from qrisp import gphase, qswitch, ry, u3
from qrisp.misc.utility import jasp_bit_reverse


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
    phi = jnp.where(jnp.abs(b) > 1e-12, jnp.angle(b), 0.0)
    lam = 0.0
    return theta, phi, lam


def _normalize(v: jnp.ndarray, n: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize a vector by its norm and apply a phase shift.

    If the norm is very small (below 1e-12), return the original vector.
    """
    return jnp.where(
        n > 1e-12,
        v / (n * jnp.exp(1j * a)),
        v,
    )


# ---------- Internal operation (split into v0, v1) ----------
def internal_op(subvec, acc):

    sub_len = subvec.shape[0]
    half = sub_len // 2

    v0 = subvec[:half]
    v1 = subvec[half:]

    n0 = jnp.linalg.norm(v0)
    n1 = jnp.linalg.norm(v1)

    theta_l = 2.0 * jnp.arccos(jnp.minimum(1.0, n0))

    alpha0 = jnp.where(n0 > 1e-12, jnp.angle(v0[0]), 0.0)
    alpha1 = jnp.where(n1 > 1e-12, jnp.angle(v1[0]), 0.0)

    v0n = _normalize(v0, n0, alpha0)
    v1n = _normalize(v1, n1, alpha1)

    acc0 = acc + alpha0
    acc1 = acc + alpha1

    children = jnp.stack([v0n, v1n], axis=0)  # shape (2, half)
    child_phases = jnp.stack([acc0, acc1], axis=0)

    return theta_l, children, child_phases


# ---------- Leaf operation (sub_len == 2) ----------
def leaf_op(subvec, acc):
    a0 = subvec[0]
    mag0 = jnp.abs(a0)
    a0_phase = jnp.where(mag0 > 1e-12, jnp.angle(a0), 0.0)

    vec_n = subvec * jnp.exp(-1j * a0_phase)
    theta, phi, lam = _rot_params_from_state(vec_n)

    total_phase = acc + a0_phase
    return jnp.array([theta, phi, lam]), total_phase


def _preprocess(target_array) -> tuple:
    """Preprocess the target statevector for state preparation."""

    n = int(np.log2(target_array.shape[0]))

    if n == 1:
        a0 = target_array[0]
        mag0 = jnp.abs(a0)
        a0_phase = jnp.where(mag0 > 1e-12, jnp.angle(a0), 0.0)
        vec_n = target_array * jnp.exp(-1j * a0_phase)

        theta, phi, lam = _rot_params_from_state(vec_n)

        thetas = jnp.zeros((0, 1))
        leaf_u = jnp.stack([theta, phi, lam])[None, :]
        leaf_phase = (a0_phase)[None]
        return thetas, leaf_u, leaf_phase

    num_theta_layers = n - 1
    num_leaves = 1 << (n - 1)

    # We store thetas in a "rectangular" array: (layer, index),
    # and only use the first 2^l entries at layer l.
    thetas = jnp.zeros((num_theta_layers, num_leaves))
    leaf_u = jnp.zeros((num_leaves, 3))
    leaf_phase = jnp.zeros((num_leaves,))

    level_vecs = target_array[jnp.newaxis, :]
    acc_phases = jnp.zeros((1,))

    # ---------- BFS over tree levels ----------
    for l in range(n):

        num_nodes = 1 << l
        sub_len = 1 << (n - l)

        level_vecs_l = level_vecs[:, :sub_len]

        if sub_len == 2:
            leaf_u_all, leaf_phase_all = jax.vmap(leaf_op)(level_vecs_l, acc_phases)
            leaf_u = leaf_u.at[:num_nodes, :].set(leaf_u_all)
            leaf_phase = leaf_phase.at[:num_nodes].set(leaf_phase_all)
            break

        theta_vec, children_all, child_accs_all = jax.vmap(internal_op)(
            level_vecs_l, acc_phases
        )

        thetas = thetas.at[l, :num_nodes].set(theta_vec)
        level_vecs = children_all.reshape((2 * num_nodes, sub_len // 2))
        acc_phases = child_accs_all.reshape((2 * num_nodes,))

    return thetas, leaf_u, leaf_phase


def state_preparation(qv, target_array, method: str = "auto") -> None:
    """
    TODO: add docstring
    """

    target_array = jnp.asarray(target_array, dtype=jnp.complex128)
    # n is static, so we can use normal numpy here
    n = int(np.log2(target_array.shape[0]))

    thetas, leaf_u, leaf_phase = _preprocess(target_array)

    def make_case_fn(layer_size: int, is_final: bool = False) -> Callable:
        """Create a case function for qswitch at a given layer."""

        def _case_fn(i, qb):
            rev_idx = jasp_bit_reverse(i, layer_size)
            if is_final:
                theta_i, phi_i, lam_i = leaf_u[rev_idx]
                u3(theta_i, phi_i, lam_i, qb)
                gphase(leaf_phase[rev_idx], qb)
            else:
                ry(thetas[layer_size][rev_idx], qb)

        return _case_fn

    if n == 1:
        theta, phi, lam = leaf_u[0]
        u3(theta, phi, lam, qv[0])
        gphase(leaf_phase[0], qv[0])
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
