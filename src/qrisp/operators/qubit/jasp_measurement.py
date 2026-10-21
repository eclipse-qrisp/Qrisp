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

"""Jasp/JAX-traceable expectation-value measurement of QubitOperators via sampling.

The expectation value is estimated with the ``MeasurementPlan`` of the operator (see
``measurement_plan.py`` for the grouping, the bit masks and the chunks). The shot counts
are fixed when ``expectation_value`` is called, so all arrays have shapes that are known
while tracing:

1. Sampling: for each group, a sampling kernel prepares the state with ``state_prep``,
   applies the group's change of basis, and returns the first ``plan.num_qubits`` qubits,
   the ones the operator acts on, as views of the register in chunks of ``_CHUNK_SIZE``
   qubits. The views are constructed from a template of a plain ``QuantumVariable``,
   whose decoder returns the measured integer, so ``sample`` yields the measured bits
   whatever the type of the prepared variable. All chunks of a shot come from the same
   measurement. The result is an array of shape (shots of the group, chunks).
2. Evaluation: right after a group is sampled, a jitted function computes the group's
   mean. It converts the samples to integers, loops over the group's terms, evaluates
   each term on all samples of the group at once, chunk by chunk, and adds coefficient
   times value. The sum is divided by the group's number of shots, and the means of the
   groups are added. Each group is evaluated in its own call, so the intermediate arrays
   of only one group exist at a time.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from qrisp.core import QuantumVariable
from qrisp.jasp import QuantumVariableTemplate, sample
from qrisp.operators.qubit.measurement_plan import _CHUNK_SIZE, MeasurementPlan


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
    expectation = jnp.float64(0)
    for group, group_shots, measurement_operator, observables, group_coefficients in zip(
        plan.groups, shots, plan.measurement_operators, plan.observables, plan.coefficients
    ):

        def raw_state_prep(state_args):
            qv = state_prep(*state_args)
            if not isinstance(qv, QuantumVariable):
                raise TypeError(f"state_prep must return a QuantumVariable, got {type(qv).__name__}.")
            group.change_of_basis(qv, method=plan.diagonalization_method)
            return _raw_register_chunks(qv, plan.num_qubits)

        res = sample(raw_state_prep, shots=group_shots)(state_args)
        # Each group is evaluated in its own jitted call to bound the memory. Within one
        # call, XLA allocates the temporary arrays of the evaluation, which are as large as
        # the group's samples, and frees them when the call returns. A single jitted
        # function over all groups would keep the temporaries of all groups alive at once,
        # i.e., about as much memory again as all samples together. For the same reason, the
        # samples are converted to integers inside the call, not in a full copy beforehand.
        # Only the group's own terms are passed, without the padding of the plan.
        num_terms = len(measurement_operator.terms_dict)
        expectation = expectation + _group_mean(
            _as_chunk_array(res, group_shots, num_chunks),
            observables[:num_terms],
            group_coefficients[:num_terms],
        )

    return expectation


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
    """Return the samples of the chunks of a register as an array of shape ``(shots, chunks)``.

    ``sample`` returns one array, or one array per chunk if its kernel returns several values per shot.
    """
    if isinstance(res, tuple):
        res = jnp.stack(res, axis=-1)
    return jnp.reshape(res, (shots, num_chunks))


@jax.jit
def _group_mean(samples, masks, coefficients):
    r"""Return the mean value of the terms of one group over its samples.

    ``samples`` has the shape (shots, chunks) and is converted to integers here, ``masks``
    holds the masks of the group's terms with the shape (terms, 3, chunks), and
    ``coefficients`` their coefficients. A term contributes its coefficient times
    :math:`(-1)^{|z \wedge x|}` for a sample :math:`x` if :math:`(x \oplus c) \wedge a = 0`
    for its Z mask :math:`z`, AND mask :math:`a` and control mask :math:`c`, and zero
    otherwise.
    """
    x = samples.astype(jnp.int64)

    def add_term(term, total):
        satisfied = jnp.all(((x ^ masks[term, 2]) & masks[term, 1]) == 0, axis=1)
        parity = jnp.sum(jax.lax.population_count(x & masks[term, 0]), axis=1) % 2
        return total + coefficients[term] * jnp.sum(jnp.where(satisfied, 1 - 2 * parity, 0))

    return jax.lax.fori_loop(0, masks.shape[0], add_term, jnp.float64(0)) / samples.shape[0]
