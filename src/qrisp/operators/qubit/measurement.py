# ********************************************************************************
# * Copyright (c) 2024 the Qrisp authors
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

"""Backend-sampling measurement of QubitOperator expectation values.

The expectation value is estimated with the ``MeasurementPlan`` of the operator (see
``measurement_plan.py`` for the grouping, the bit masks and the chunks):

1. Circuit: the state preparation is compiled once per evaluation, or ``precompiled_qc``
   is used, and the abstract parameters are bound with ``subs_dic``.
2. Measurement: for each group, the group's change-of-basis gate is appended to a copy of
   the circuit, and the first ``plan.num_qubits`` qubits, the ones the operator acts on,
   are measured with the group's number of shots. The backend returns the distinct
   outcomes with their probabilities. A ``BatchedBackend`` receives all circuits at once.
3. Outcomes: the outcomes are integers with one bit per measured qubit. They are split
   into chunks of ``_CHUNK_SIZE`` qubits, like the masks of the plan, and stored with their
   probabilities in arrays of shape (groups, outcomes, chunks) and (groups, outcomes).
   Groups with fewer distinct outcomes are padded with zero probabilities.
4. Evaluation: a numba kernel loops over the groups, their outcomes and their terms, and
   sums probability * coefficient * value of the term for the outcome, with the value
   computed chunk by chunk. The probabilities of each group sum to one, so this is the
   sum over the groups of their mean values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numba import njit

from qrisp.circuit import QuantumCircuit, Qubit
from qrisp.core import QuantumArray, QuantumSession, QuantumVariable, merge
from qrisp.core.compilation import qompiler
from qrisp.interface import BackendLike, BatchedBackend
from qrisp.operators.qubit.measurement_plan import _split_bits

if TYPE_CHECKING:
    from qrisp.operators.qubit.measurement_plan import MeasurementPlan


def _expectation_value_helper(  # noqa: PLR0912, PLR0913
    plan: MeasurementPlan,
    qarg: QuantumVariable | QuantumArray | list[Qubit],
    shots: list[int],
    *,
    backend: BackendLike | None = None,
    compile: bool = True,
    compilation_kwargs: dict[Any, Any] | None = None,
    subs_dic: dict[Any, Any] | None = None,
    precompiled_qc: QuantumCircuit | None = None,
) -> float:
    """Measure the expectation value of a measurement plan on the state of ``qarg`` with a backend."""
    from qrisp.misc import get_measurement_from_qc

    if compilation_kwargs is None:
        compilation_kwargs = {}

    if isinstance(qarg, QuantumVariable):
        if qarg.is_deleted():
            raise Exception("Tried to get measurement from deleted QuantumVariable")
        qs = qarg.qs

    elif isinstance(qarg, QuantumArray):
        for qv in qarg.flatten():
            if qv.is_deleted():
                raise Exception("Tried to measure QuantumArray containing deleted QuantumVariables")
        qs = qarg.qs
    elif isinstance(qarg, list):
        qs = QuantumSession()
        for arg in qarg:
            if isinstance(arg, QuantumVariable) and qv.is_deleted():
                raise Exception("Tried to measure QuantumArray containing deleted QuantumVariables")
            merge(qs, arg)

    if backend is None:
        if qs.backend is None:
            from qrisp.default_backend import def_backend

            backend = def_backend
        else:
            backend = qarg.qs.backend

    if len(qs.env_stack) != 0:
        raise Exception("Tried to get measurement within open environment")

    if plan.num_qubits == 0:
        # A multiple of the identity has the same value for every state.
        return float(np.sum(plan.coefficients))

    # Copy circuit in over to prevent modification
    if precompiled_qc is None:
        if compile:
            qc = qompiler(qs, **compilation_kwargs)
        else:
            qc = qs.copy()
        qubit_list = qarg
    else:
        qc = precompiled_qc.copy()
        qubit_list = qc.qubits[: len(qarg.reg)]

    # Bind parameters
    if subs_dic:
        qc = qc.bind_parameters(subs_dic)
        from qrisp.circuit.pass_management.passes.combine_single_qubit_gates import combine_single_qubit_gates

        qc = combine_single_qubit_gates(qc)

    qc = qc.transpile()

    # Only the qubits the operator acts on are measured.
    qubits = [qubit_list[j] for j in range(plan.num_qubits)]

    results = []
    for gate, group_shots in zip(plan.change_of_basis_gates(), shots):
        curr = qc.copy()
        curr.append(gate, qubits)

        results.append(get_measurement_from_qc(curr, qubits, backend, group_shots))

    if isinstance(backend, BatchedBackend):
        backend.dispatch()

    outcomes, probabilities = _outcome_arrays(results, plan.observables.shape[-1])

    return _evaluate_expectation(outcomes, probabilities, plan.observables, plan.coefficients)


def _outcome_arrays(results: list, num_chunks: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the measured outcomes of all groups split into chunks, and their probabilities.

    The arrays have the shapes ``(groups, outcomes, chunks)`` and
    ``(groups, outcomes)``. Groups with fewer distinct outcomes are padded with
    zero probabilities.
    """
    results = [list(result.items()) for result in results]
    num_outcomes = max((len(items) for items in results), default=0)
    outcomes = np.zeros((len(results), num_outcomes, num_chunks), dtype=np.int64)
    probabilities = np.zeros((len(results), num_outcomes), dtype=np.float64)
    for i, items in enumerate(results):
        if not items:
            continue
        keys, values = zip(*items)
        probabilities[i, : len(values)] = values
        if num_chunks == 1:
            outcomes[i, : len(keys), 0] = keys
        else:
            outcomes[i, : len(keys)] = [_split_bits(key, num_chunks) for key in keys]
    return outcomes, probabilities


@njit(cache=True)
def _parity(value: int) -> int:
    """Return the parity of the set bits of a non-negative 64-bit integer."""
    value ^= value >> 32
    value ^= value >> 16
    value ^= value >> 8
    value ^= value >> 4
    value ^= value >> 2
    value ^= value >> 1
    return value & 1


@njit(cache=True)
def _evaluate_expectation(outcomes, probabilities, observables, coefficients):
    r"""Sum the values of all terms over the measured outcomes of all groups.

    A term contributes its coefficient times :math:`(-1)^{|z \wedge x|}` for an
    outcome :math:`x` if :math:`(x \oplus c) \wedge a = 0` for its Z mask
    :math:`z`, AND mask :math:`a` and control mask :math:`c`, and zero
    otherwise. The masks and outcomes are split into chunks of equal size, see
    ``MeasurementPlan.observables``.
    """
    expectation = 0.0
    for group in range(outcomes.shape[0]):
        for outcome_index in range(outcomes.shape[1]):
            probability = probabilities[group, outcome_index]
            if probability == 0:
                continue
            outcome = outcomes[group, outcome_index]
            for term in range(observables.shape[1]):
                coefficient = coefficients[group, term]
                if coefficient == 0:
                    continue
                masks = observables[group, term]
                sign_bits = 0
                satisfied = True
                for chunk in range(outcome.shape[0]):
                    if (outcome[chunk] ^ masks[2, chunk]) & masks[1, chunk]:
                        satisfied = False
                        break
                    sign_bits ^= outcome[chunk] & masks[0, chunk]
                if satisfied:
                    expectation += probability * coefficient * (1 - 2 * _parity(sign_bits))
    return expectation
