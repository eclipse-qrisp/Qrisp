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
from jax import lax

from qrisp.misc.utility import _EPSILON, swap_endianness


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
    # We know that a is real (and non-negative).
    # This step avoids warning about casting complex to real.
    a = jnp.clip(jnp.real(a), -1.0, 1.0)
    theta = 2.0 * jnp.arccos(a)
    phi = jnp.where(jnp.abs(b) > _EPSILON, jnp.angle(b), 0.0)
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
        norm > _EPSILON,
        lambda _: branch_nonzero(None),
        lambda _: branch_zero(None),
        operand=None,
    )


def _compute_thetas(
    vec: jnp.ndarray, acc: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    For a given input vector, this function computes the rotation angles
    needed for the uniformly controlled RY at this tree layer, normalizes its child vectors,
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


def _compute_u3_params(
    qubit_vec: jnp.ndarray, acc: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    For a given length-2 vector, this function computes the U3 gate parameters needed
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
# - `phases` has shape (2^(n-1),), and contains the global phase for each leaf node.
#
#    phases = Array[phase_leaf0, phase_leaf1, ..., phase_leaf_{2^(n-1)-1}]
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
    phases : jnp.ndarray
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


def init_state_qswitch(qv, target_array: jnp.ndarray) -> None:
    """
    Prepare the quantum state encoded in ``qv`` so that it matches the given
    ``target_array`` by constructing a binary-tree decomposition of the target
    amplitudes and applying a sequence of uniformly controlled rotations via
    the ``qswitch`` primitive.

    This routine implements a standard state-preparation algorithm based on
    recursively splitting the target statevector.
    The classical preprocessing stage extracts RY angles for internal tree nodes
    and U3 parameters for the leaf nodes.
    The quantum stage applies them using ``qswitch``, which replaces
    explicit multiplexers and conditionals in both static execution and Jasp mode.

    .. note::

        During the quantum stage, ``qswitch`` enumerates control patterns in
        little-endian order, so each index is bit-reversed before accessing
        the parameters computed in the classical preprocessing stage.

    Parameters
    ----------
    qv : QuantumVariable
        The quantum variable representing the qubits to be prepared.
    target_array : jnp.ndarray
        A normalized complex vector representing the target state to prepare.

    """

    # These imports are here to avoid circular dependencies
    from qrisp import gphase, qswitch, ry, u3
    from qrisp.jasp.program_control.jrange_iterator import jrange
    from qrisp.jasp.tracing_logic import check_for_tracing_mode
    from qrisp.misc.utility import bit_reverse

    target_array = jnp.asarray(target_array, dtype=jnp.complex128)
    # n is static (known at compile time), so we can use normal numpy here
    n = int(np.log2(target_array.shape[0]))

    # We could use jrange even in static mode, but this would add overhead.
    xrange = jrange if check_for_tracing_mode() else range

    thetas, u_params, phases = _preprocess(target_array)

    def make_case_fn(layer_size: int, is_final: bool = False) -> Callable:
        """Create a case function for qswitch at a given layer."""

        def case_fn(i, qb):
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

        qswitch(
            operand=qv[layer_size],
            case=qv[:layer_size],
            case_function=make_case_fn(layer_size),
        )

    qswitch(
        operand=qv[qv.size - 1],
        case=qv[: qv.size - 1],
        case_function=make_case_fn(qv.size - 1, is_final=True),
    )


def init_state_qiskit(qv, target_array, from_dict: bool = False) -> None:
    """
    Initialize the quantum state of ``qv`` to match ``target_array`` using Qiskit's
    StatePreparation circuit.

    ``target_array`` is assumed to be in big-endian order by default, but if ``from_dict`` is ``True``,
    it is assumed to be in little-endian order.

    Parameters
    ----------
    qv : QuantumVariable
        The quantum variable representing the qubits to be prepared.
    target_array : array-like
        A normalized complex vector representing the target state to prepare.
    from_dict : bool, optional
        If ``True``, indicates that ``target_array`` is in little-endian order. By default, ``False``.

    """
    from qiskit.circuit.library.data_preparation import StatePreparation

    from qrisp import QuantumCircuit
    from qrisp.simulator import statevector_sim

    target_array = np.asarray(target_array, dtype=np.complex128)

    n = qv.size

    target_little_end = target_array if from_dict else swap_endianness(target_array, n)

    qiskit_qc = StatePreparation(target_little_end).definition
    init_qc = QuantumCircuit.from_qiskit(qiskit_qc)

    init_qc.qubits.reverse()
    sim_little_end = statevector_sim(init_qc)
    init_qc.qubits.reverse()

    sim_big_end = sim_little_end if from_dict else swap_endianness(sim_little_end, n)

    # Apply global phase correction
    arg_max = np.argmax(np.abs(sim_big_end))
    gphase = (np.angle(target_array[arg_max] / sim_big_end[arg_max])) % (2 * np.pi)
    init_qc.gphase(gphase, 0)

    init_gate = init_qc.to_gate()
    init_gate.name = "state_init"
    qv.qs.append(init_gate, qv)
