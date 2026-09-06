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

"""Jasp/JAX-traceable expectation-value measurement of QubitOperators via sampling."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

from qrisp.jasp import sample
from qrisp.operators.qubit.measurement_plan import _create_measurement_plan

if TYPE_CHECKING:
    from qrisp.core import QuantumVariable
    from qrisp.operators.hamiltonian import Hamiltonian


def _jasp_expectation_value_helper(  # noqa: PLR0913
    hamiltonian: Hamiltonian,
    state_prep: Callable[..., QuantumVariable],
    *,
    precision: float = 0.01,
    shots: int | None = None,
    max_shots: int | None = None,
    state_args: tuple = (),
    diagonalization_method: Literal["commuting", "commuting_qw"] = "commuting_qw",
) -> jax.Array:
    """Evaluate a the expectation value of a Hamiltonian using Jasp sampling."""
    plan = _create_measurement_plan(hamiltonian, diagonalization_method, construct_basis_gates=False)
    if len(plan.hamiltonian.terms_dict) == 0:
        return jnp.array(0)

    samples = []
    meas_ops = []
    meas_coeffs = []
    shots_list = plan.allocate_shots(precision, shots=shots, max_shots=max_shots)

    for index, group in enumerate(plan.groups):
        # Calculate the new measurement operators (after change of basis)
        meas_op = plan.measurement_operators[index]

        def new_state_prep(state_args):
            qv = state_prep(*state_args)
            group.change_of_basis(qv, method=diagonalization_method)
            return qv

        res = sample(new_state_prep, shots=shots_list[index])(state_args)

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
    """Evaluate the expectation."""
    expectation = 0

    # Evaluate and sum intermediate results for each measurement setting
    for index, ops in enumerate(operators):
        expectation += sum_over_observables_and_samples(ops, samples[index], coefficients[index]) / len(samples[index])

    return expectation


@jax.jit
def jasp_evaluate_observable_jitted(observable: tuple, x: int):
    """Evaluate one serialized observable for a sampled bitstring."""
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
    return prefactor * jnp.where(sign_flip % 2 == 0, 1, -1) * jnp.int64((AND_bits == 0) | (corrected_x & AND_bits == 0))


@jax.jit
def sum_over_observables_and_samples(observables, x_values, coefficients):
    """Sum serialized observable values over all samples in one group."""

    def body_fun(i, val):
        sum_val = val
        obs = observables[i]
        c = coefficients[i]
        results = jax.vmap(jasp_evaluate_observable_jitted, in_axes=(None, 0))(obs, x_values)
        return sum_val + c * jnp.sum(results)

    total_sum = jax.lax.fori_loop(
        0,
        observables.shape[0],
        body_fun,
        jnp.float64(0),
    )
    return total_sum
