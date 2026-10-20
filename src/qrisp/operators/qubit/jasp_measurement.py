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
import numpy as np

from qrisp.core import QuantumVariable
from qrisp.jasp import QuantumVariableTemplate, sample
from qrisp.operators.qubit.measurement_plan import _CHUNK_SIZE

if TYPE_CHECKING:
    from qrisp.operators.qubit.measurement_plan import MeasurementPlan


def _check_jasp_arguments(static_only_arguments: list[str]) -> None:
    """Raise a ``ValueError`` if arguments that only apply to static execution are used in Jasp mode."""
    if static_only_arguments:
        raise ValueError(
            f"{', '.join(static_only_arguments)} cannot be used in Jasp mode. In Jasp mode, the state is always "
            "prepared by state_prep and simulated by the Jasp interpreter; backend, compile=False, "
            "compilation_kwargs, precompiled_qc and subs_dic only apply to static execution."
        )


def _jasp_expectation_value_helper(
    plan: MeasurementPlan,
    state_prep: Callable[..., QuantumVariable],
    shots: list[int],
    state_args: tuple = (),
) -> jax.Array:
    """Measure the expectation value of a measurement plan on the state of ``state_prep`` with Jasp sampling."""
    if plan.num_qubits == 0:
        # A multiple of the identity has the same value for every state.
        return jnp.array(np.sum(plan.coefficients))

    num_chunks = plan.observables.shape[-1]
    samples = []
    for group, group_shots in zip(plan.groups, shots):

        def raw_state_prep(state_args):
            qv = state_prep(*state_args)
            group.change_of_basis(qv, method=plan.diagonalization_method)
            return _raw_register_chunks(qv, plan.num_qubits)

        res = sample(raw_state_prep, shots=group_shots)(state_args)
        samples.append(_as_chunk_array(res, group_shots, num_chunks))

    # All groups are evaluated together. Each sample is weighted by 1/shots of its group.
    group_indices = np.repeat(np.arange(len(shots)), shots)
    sample_weights = np.repeat(1 / np.array(shots), shots)
    return _evaluate_samples(
        jnp.concatenate(samples), group_indices, sample_weights, plan.observables, plan.coefficients
    )


# Template of a QuantumVariable without decoder. Views of a register constructed from it
# are sampled as the raw measured integers, whatever the type of the register. It is
# created once at import, because creating a QuantumVariable while tracing would
# allocate qubits.
_RAW_REGISTER = QuantumVariableTemplate(QuantumVariable(0), size_tracked=False)


def _raw_register_chunks(qv: QuantumVariable, num_qubits: int) -> QuantumVariable | tuple[QuantumVariable, ...]:
    """Return undecoded views of the first ``num_qubits`` qubits of ``qv``, in chunks of ``_CHUNK_SIZE`` qubits."""
    chunks = tuple(
        _RAW_REGISTER.construct(reg=qv.reg[start : min(start + _CHUNK_SIZE, num_qubits)])
        for start in range(0, num_qubits, _CHUNK_SIZE)
    )
    return chunks[0] if len(chunks) == 1 else chunks


def _as_chunk_array(res: jax.Array | tuple[jax.Array, ...], shots: int, num_chunks: int) -> jax.Array:
    """Return the samples of the chunks of a register as an integer array of shape ``(shots, chunks)``.

    ``sample`` returns one array, or one array per chunk if its kernel returns several values per shot.
    """
    if isinstance(res, tuple):
        res = jnp.stack(res, axis=-1)
    return jnp.reshape(res, (shots, num_chunks)).astype(jnp.int64)


@jax.jit
def _evaluate_samples(samples, group_indices, sample_weights, observables, coefficients):
    r"""Sum the weighted values of all terms over the samples of all groups.

    A term contributes its coefficient times :math:`(-1)^{|z \wedge x|}` for a
    sample :math:`x` if :math:`(x \oplus c) \wedge a = 0` for its Z mask
    :math:`z`, AND mask :math:`a` and control mask :math:`c`, and zero
    otherwise. The masks and samples are split into chunks of equal size, see
    ``MeasurementPlan.observables``.
    """

    def add_term(term, total):
        masks = observables[group_indices, term]
        coefficient = coefficients[group_indices, term]
        satisfied = jnp.all(((samples ^ masks[:, 2]) & masks[:, 1]) == 0, axis=1)
        parity = jnp.sum(jax.lax.population_count(samples & masks[:, 0]), axis=1) % 2
        values = jnp.where(satisfied, 1 - 2 * parity, 0)
        return total + jnp.sum(sample_weights * coefficient * values)

    return jax.lax.fori_loop(0, observables.shape[1], add_term, jnp.float64(0))
