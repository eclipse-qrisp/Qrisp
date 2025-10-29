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

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import fori_loop, while_loop

from qrisp.core import QuantumVariable, measure
from qrisp.jasp import sample


def get_jasp_measurement(
    hamiltonian,
    state_prep,
    state_args=(),
    precision=0.01,
    diagonalisation_method="commuting_qw",
):
    r"""
    This method returns the expected value of a Hamiltonian for the state of a quantum argument.

    Parameters
    ----------
    hamiltonian : QubitOperator
        The Hamiltonian for which the exptectaion value is measuered.
    state_prep : callable
        A function returning a QuantumVariable.
        The expectation of the Hamiltonian for the state from this QuantumVariable will be measured.
        The state preparation function can only take classical values as arguments.
        This is because a quantum value would need to be copied for each sampling iteration, which is prohibited by the no-cloning theorem.
    state_args : tuple
        A tuple of arguments of the ``state_prep`` function.
    precision: float, optional
        The precision with which the expectation of the Hamiltonian is to be evaluated.
        The default is 0.01. The number of shots scales quadratically with the inverse precision.
    diagonalisation_method : str, optional
        Specifies the method for grouping and diagonalizing the QubitOperator.
        Available are ``commuting_qw``, i.e., the operator is grouped based on qubit-wise commutativity of terms,
        and ``commuting``, i.e., the operator is grouped based on commutativity of terms.
        The default is ``commuting_qw``.

    Returns
    -------
    float
        The expected value of the Hamiltonian.

    """

    hamiltonian = hamiltonian.hermitize()
    hamiltonian = hamiltonian.eliminate_ladder_conjugates()
    hamiltonian = hamiltonian.apply_threshold(0)
    if len(hamiltonian.terms_dict) == 0:
        return 0

    if diagonalisation_method == "commuting_qw":
        temp_groups = hamiltonian.commuting_qw_groups()
        groups = []
        # In order for the change of basis function (below) to work properly,
        # the ladder terms either need to completely agree or completely disagree
        for group in temp_groups:
            groups.extend(
                group.group_up(
                    lambda a, b: a.ladders_agree(b) or not a.ladders_intersect(b)
                )
            )

    elif diagonalisation_method == "commuting":
        temp_groups = hamiltonian.group_up(lambda a, b: a.commute_pauli(b))
        groups = []
        # In order for the change of basis function (below) to work properly,
        # the ladder terms either need to completely agree or completely disagree
        for group in temp_groups:
            groups.extend(
                group.group_up(
                    lambda a, b: a.ladders_agree(b) or not a.ladders_intersect(b)
                )
            )

    else:
        raise Exception(f"Unknown diagonalisation method: {diagonalisation_method}.")

    samples = []
    meas_ops = []
    meas_coeffs = []
    stds = []

    # Compute amounts of shots
    for group in groups:
        # Collect standard deviation
        n = hamiltonian.find_minimal_qubit_amount()
        stds.append(np.sqrt(group.get_operator_variance(n=n)))

    N = sum(stds)
    shots_list = [N * s for s in stds]

    for index, group in enumerate(groups):

        # Calculate the new measurement operators (after change of basis)
        meas_op = group.change_of_basis(method=diagonalisation_method)

        def new_state_prep(state_args):
            qv = state_prep(*state_args)
            group.change_of_basis(qv, method=diagonalisation_method)
            return qv

        shots = int(shots_list[index] / precision**2)
        res = sample(new_state_prep, shots=shots)(state_args)

        samples.append(jnp.int64(res))

        temp_meas_ops = []
        temp_coeff = []
        for term, coeff in meas_op.terms_dict.items():
            temp_meas_ops.append(jnp.array(term.serialize(), dtype=jnp.int64))
            temp_coeff.append(jnp.real(coeff))

        meas_coeffs.append(jnp.array(temp_coeff, dtype=jnp.float64))
        meas_ops.append(jnp.array(temp_meas_ops))

    expectation = jasp_evaluate_expectation_jitted(samples, meas_ops, meas_coeffs)
    return expectation


@jax.jit
def jasp_evaluate_expectation_jitted(samples, operators, coefficients):
    """
    Evaluate the expectation.

    """

    expectation = 0

    # Evaluate and sum intermediate results for each measurement setting
    for index, ops in enumerate(operators):
        expectation += sum_over_observables_and_samples(
            ops, samples[index], coefficients[index]
        ) / len(samples[index])

    return expectation


@jax.jit
def jasp_evaluate_observable_jitted(observable: tuple, x: int):
    # This function evaluates how to compute the energy of a measurement sample x.
    # Since we are also considering ladder operators, this energy can either be
    # 0, -1 or 1. For more details check out the comments of QubitOperator.get_conjugation_circuit

    # The observable is given as tuple, containing four integers.
    # To understand the meaning of these integers check QubitTerm.serialize.
    # print(observable)
    # Unwrap the tuple
    z_int, AND_bits, AND_ctrl_state, contains_ladder = observable

    # Compute whether the sign should be sign flipped based on the Z operators
    sign_flip_int = z_int & x
    sign_flip = jax.lax.population_count(sign_flip_int)

    # If there is a ladder operator in the term, we need to half the energy
    # because we want to measure (|110><110| - |111><111|)/2
    prefactor = 1 - 0.5 * contains_ladder

    # If there are no AND bits, we return the result
    # Otherwise we apply the AND_ctrl_state to flip the appropriate bits.
    corrected_x = x ^ AND_ctrl_state

    # If all bits are in the 0 state the AND is true.
    return (
        prefactor
        * jnp.where(sign_flip % 2 == 0, 1, -1)
        * jnp.int64((AND_bits == 0) | (corrected_x & AND_bits == 0))
    )


@jax.jit
def sum_over_observables_and_samples(observables, x_values, coefficients):

    def body_fun(i, val):
        sum_val = val
        obs = observables[i]
        c = coefficients[i]
        results = jax.vmap(jasp_evaluate_observable_jitted, in_axes=(None, 0))(
            obs, x_values
        )
        return sum_val + c * jnp.sum(results)

    total_sum = jax.lax.fori_loop(
        0,
        observables.shape[0],
        body_fun,
        jnp.float64(0),
    )
    return total_sum
