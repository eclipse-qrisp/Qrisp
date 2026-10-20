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

"""Backend-sampling measurement of QubitOperator expectation values."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from numba import njit

from qrisp.circuit import QuantumCircuit, Qubit
from qrisp.core import QuantumArray, QuantumSession, QuantumVariable, merge
from qrisp.core.compilation import qompiler
from qrisp.interface import BackendLike, BatchedBackend

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

    if not plan.groups:
        return 0.0

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

    results = []
    for gate, group_shots in zip(plan.change_of_basis_gates(), shots):
        qubits = [qubit_list[j] for j in range(gate.num_qubits)]

        curr = qc.copy()
        curr.append(gate, qubits)

        results.append(get_measurement_from_qc(curr, list(qubit_list), backend, group_shots))

    if isinstance(backend, BatchedBackend):
        backend.dispatch()

    samples = create_padded_array([list(res.keys()) for res in results]).astype(np.int64)
    probs = create_padded_array([list(res.values()) for res in results])

    return evaluate_expectation_jitted(samples, probs, plan.observables, plan.coefficients)


def create_padded_array(list_of_lists):
    """Create a zero-padded numpy array from a list of lists with varying lengths.

    Parameters
    ----------
    list_of_lists : list
        A list of lists with potentially different lengths.

    Returns
    -------
    numpy.ndarray: A 2D numpy array with padded rows.

    """
    # Find the maximum length of any list in the input
    max_length = max(len(lst) for lst in list_of_lists)

    # Create a padded list of lists
    padded_lists = [lst + [0] * (max_length - len(lst)) for lst in list_of_lists]

    # Convert to numpy array
    return np.array(padded_lists)


#
# Evaluate expectation
#


def evaluate_observable(observable: tuple, x: int):
    """Evaluate one serialized observable for a sampled bitstring."""
    # This function evaluates how to compute the energy of a measurement sample x.
    # Since we are also considering ladder operators, this energy can either be
    # 0, -1 or 1. For more details check out the comments of QubitOperator.get_conjugation_circuit

    # The observable is given as tuple, containing for integers and a boolean.
    # To understand the meaning of these integers check QubitTerm.serialize.

    # Unwrap the tuple
    z_int, AND_bits, AND_ctrl_state, contains_ladder = observable

    # Compute whether the sign should be sign flipped based on the Z operators
    sign_flip_int = z_int & x
    sign_flip = 0
    while sign_flip_int:
        sign_flip += sign_flip_int & 1
        sign_flip_int >>= 1

    # If there is a ladder operator in the term, we need to half the energy
    # because we want to measure (|110><110| - |111><111|)/2
    if contains_ladder:
        prefactor = 0.5
    else:
        prefactor = 1

    # If there are no and bits, we return the result
    if AND_bits == 0:
        return prefactor * (-1) ** sign_flip

    # Otherwise we apply the AND_ctrl_state to flip the appropriate bits.
    corrected_x = x ^ AND_ctrl_state

    # If all bits are in the 0 state the AND is true.
    if corrected_x & AND_bits == 0:
        return prefactor * (-1) ** sign_flip
    else:
        return 0


evaluate_observable_jitted = njit(cache=True)(evaluate_observable)


@njit(cache=True)
def evaluate_expectation_jitted(samples, probs, operators, coefficients):
    """Evaluate the expectation."""
    expectation = 0

    for index1, ops in enumerate(operators):
        for index2, op in enumerate(ops):
            for i in range(len(samples[index1])):
                outcome, probability = samples[index1, i], probs[index1, i]
                expectation += probability * evaluate_observable_jitted(op, outcome) * coefficients[index1][index2]

    return expectation


def partition(values, num_qubits):
    """Partitions a list of integers into a list of lists of integers with size 64 bit.

    Parameters
    ----------
    values : list[int]
        A list of integers.
    num_qubits : int
        The maximal number of bits.

    Returns
    -------
    partition : list[np.array]
        A list of NumPy numpy.uint64 arrays.

    """
    M = math.ceil(num_qubits / 64)
    N = len(values)

    zeros = [[0] * N for k in range(M)]
    partition = [values]
    partition.extend(zeros)

    lower_mask = (1 << 64) - 1

    for j in range(1, M):
        for i in range(N):
            partition[j][i] = partition[j - 1][i] & lower_mask
            partition[j + 1][i] = partition[j - 1][i] >> 64

    if M == 1:
        return [np.array(partition[0], dtype=np.uint64)]
    else:
        return [np.array(part, dtype=np.uint64) for part in partition[1:]]
