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
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from qrisp.jasp import sample

if TYPE_CHECKING:
    from qrisp.core import QuantumVariable
    from qrisp.operators.qubit.measurement_plan import MeasurementPlan


def _jasp_expectation_value_helper(
    plan: MeasurementPlan,
    state_prep: Callable[..., QuantumVariable],
    shots: list[int],
    state_args: tuple = (),
) -> jax.Array:
    """Measure the expectation value of a measurement plan on the state of ``state_prep`` with Jasp sampling."""
    if not plan.groups:
        return jnp.array(0)

    samples = []
    for group, group_shots in zip(plan.groups, shots):

        def new_state_prep(state_args):
            qv = state_prep(*state_args)
            group.change_of_basis(qv, method=plan.diagonalization_method)
            return qv

        res = sample(new_state_prep, shots=group_shots)(state_args)
        samples.append(jnp.int64(res))

    return jasp_evaluate_expectation_jitted(samples, plan.observables, plan.coefficients)


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
