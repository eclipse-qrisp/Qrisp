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
   into chunks of ``_CHUNK_SIZE`` qubits, like the masks of the plan. The outcomes of all
   groups are stored one after the other in an array of shape (outcomes, chunks), with
   their probabilities, and the outcomes of group g are at ``offsets[g]:offsets[g + 1]``.
   So the arrays hold exactly the distinct outcomes that were measured.
4. Evaluation: a numba kernel loops over the groups, their outcomes and their terms, and
   sums probability * coefficient * value of the term for the outcome, with the value
   computed chunk by chunk. The probabilities of each group sum to one, so this is the
   sum over the groups of their mean values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from qrisp import default_backend
from qrisp.circuit import QuantumCircuit
from qrisp.circuit.pass_management.passes.combine_single_qubit_gates import combine_single_qubit_gates
from qrisp.core import QuantumVariable
from qrisp.core.compilation import qompiler
from qrisp.interface import BackendLike, BatchedBackend
from qrisp.misc import get_measurement_from_qc
from qrisp.operators.qubit.measurement_plan import _CHUNK_SIZE, MeasurementPlan


def _expectation_value_helper(  # noqa: PLR0912, PLR0913
    plan: MeasurementPlan,
    qarg: QuantumVariable,
    shots: list[int],
    *,
    backend: BackendLike | None = None,
    compile: bool = True,
    compilation_kwargs: dict[Any, Any] | None = None,
    subs_dic: dict[Any, Any] | None = None,
    precompiled_qc: QuantumCircuit | None = None,
) -> float:
    """Measure the expectation value of a measurement plan on the state of ``qarg`` with a backend."""
    if compilation_kwargs is None:
        compilation_kwargs = {}

    if not isinstance(qarg, QuantumVariable):
        raise TypeError(f"state_prep must return a QuantumVariable, got {type(qarg).__name__}.")
    if qarg.is_deleted():
        raise Exception("Tried to get measurement from deleted QuantumVariable")
    qs = qarg.qs

    if backend is None:
        if qs.backend is None:
            # Read at call time, because the default backend can be replaced at runtime.
            backend = default_backend.def_backend
        else:
            backend = qs.backend

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

    outcomes, probabilities, offsets = _outcome_arrays(results, plan.observables.shape[-1])

    return _evaluate_expectation(outcomes, probabilities, offsets, plan.observables, plan.coefficients)


def _outcome_arrays(results: list, num_chunks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the measured outcomes of all groups split into chunks, their probabilities and the group offsets.

    The outcomes of all groups are stored one after the other, with the shapes
    ``(outcomes, chunks)`` and ``(outcomes,)``. The outcomes of group ``g`` are at
    ``offsets[g]:offsets[g + 1]``.
    """
    offsets = np.zeros(len(results) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(result) for result in results])
    outcomes = np.empty((offsets[-1], num_chunks), dtype=np.int64)
    probabilities = np.empty(offsets[-1], dtype=np.float64)
    mask = (1 << _CHUNK_SIZE) - 1
    for result, start, stop in zip(results, offsets[:-1], offsets[1:]):
        count = stop - start
        probabilities[start:stop] = np.fromiter(result.values(), dtype=np.float64, count=count)
        for chunk in range(num_chunks):
            shift = _CHUNK_SIZE * chunk
            keys = ((key >> shift) & mask for key in result.keys())
            outcomes[start:stop, chunk] = np.fromiter(keys, dtype=np.int64, count=count)
    return outcomes, probabilities, offsets


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
def _evaluate_expectation(outcomes, probabilities, offsets, observables, coefficients):
    r"""Sum the values of all terms over the measured outcomes of all groups.

    A term contributes its coefficient times :math:`(-1)^{|z \wedge x|}` for an
    outcome :math:`x` if :math:`(x \oplus c) \wedge a = 0` for its Z mask
    :math:`z`, AND mask :math:`a` and control mask :math:`c`, and zero
    otherwise. The masks and outcomes are split into chunks of equal size, see
    ``MeasurementPlan.observables``.
    """
    expectation = 0.0
    for group in range(offsets.shape[0] - 1):
        for outcome_index in range(offsets[group], offsets[group + 1]):
            probability = probabilities[outcome_index]
            if probability == 0:
                continue
            outcome = outcomes[outcome_index]
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
