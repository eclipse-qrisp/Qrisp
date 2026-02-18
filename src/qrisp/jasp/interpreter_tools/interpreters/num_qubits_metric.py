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

import types
from functools import lru_cache
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.random import key
from jax.typing import ArrayLike

from qrisp.jasp.interpreter_tools.interpreters.profiling_interpreter import (
    BaseMetric,
    eval_jaxpr,
    make_profiling_eqn_evaluator,
)
from qrisp.jasp.interpreter_tools.interpreters.utilities import (
    get_quantum_operations,
)
from qrisp.jasp.jasp_expression import Jaspr
from qrisp.jasp.primitives import (
    AbstractQubitArray,
)


# The computation will fail anyway if this is triggered,
# but this way we can print an informative message before the failure happens.
def _warn(n_allocations, max_allocations):
    """Helper function to print a warning when the number of qubits exceeds the maximum supported."""

    jax.debug.print(
        (
            "ERROR: `num_qubits` computation overflowed: tried to (de)allocate qubits "
            "{n_allocations} times, but the maximum supported is {max_allocations}. "
            "Consider increasing the `max_allocations` parameter for `num_qubits`."
        ),
        n_allocations=n_allocations,
        max_allocations=max_allocations,
    )


class NumQubitsMetric(BaseMetric):
    """
    A metric implementation that computes the number of qubits in a Jaspr.

    Parameters
    ----------
    meas_behavior : Callable
        The measurement behavior function.

    profiling_dic : dict
        The profiling dictionary mapping quantum operations to indices.

    max_allocations : int
        The maximum number of qubit allocations/deallocations supported by the profiler.

    """

    def __init__(
        self,
        meas_behavior: Callable,
        profiling_dic: dict,
        max_allocations: int = 1000,
    ):
        """Initialize the NumQubitsMetric."""

        super().__init__(meas_behavior=meas_behavior, profiling_dic=profiling_dic)

        self._max_allocations = max_allocations
        allocations_array = jnp.zeros(self._max_allocations, dtype=jnp.int64)
        self.allocations_counter_index = jnp.int64(0)
        invalid = jnp.bool_(False)

        self._initial_metric = (
            allocations_array,
            self.allocations_counter_index,
            invalid,
        )

    @property
    def initial_metric(self) -> Tuple[ArrayLike, int, bool]:
        """Return the initial metric value."""
        return self._initial_metric

    @property
    def max_allocations(self) -> int:
        """Return the maximum number of qubit allocations/deallocations supported by the profiler."""
        return self._max_allocations

    def handle_create_qubits(self, invalues, eqn, context_dic):

        size, metric_data = invalues

        allocations_array, allocations_counter_index, invalid = metric_data

        allocations_array = allocations_array.at[allocations_counter_index].set(size)
        allocations_counter_index += 1

        overflow = allocations_counter_index > jnp.int64(self._max_allocations)
        invalid = jnp.logical_or(invalid, overflow)
        jax.lax.cond(
            invalid,
            _warn,
            lambda *_: None,
            allocations_counter_index,
            self._max_allocations,
        )

        # Associate the following in context_dic:
        # QubitArray -> size
        # QuantumState -> metric_data
        return (size, (allocations_array, allocations_counter_index, invalid))

    def handle_get_qubit(self, invalues, eqn, context_dic):

        # Associate the following in context_dic:
        # Qubit -> None (we don't need to track individual qubits for this metric)
        return None

    def handle_get_size(self, invalues, eqn, context_dic):

        # Associate the following in context_dic:
        # size -> size
        return invalues[0]

    def handle_quantum_gate(self, invalues, eqn, context_dic):

        # Associate the following in context_dic:
        # QuantumState -> metric_data
        return invalues[-1]

    def handle_measure(self, invalues, eqn, context_dic):

        target, metric_data = invalues

        # We keep a measurement counter only to generate unique keys.
        meas_number = context_dic.get("_meas_number", jnp.int32(0))

        if isinstance(eqn.invars[0].aval, AbstractQubitArray):

            _, size = target

            def body_fun(i, acc):
                return self._measurement_body_fun(meas_number, i, acc)

            meas_res = jax.lax.fori_loop(0, size, body_fun, jnp.int64(0))
            context_dic["_meas_number"] = meas_number + size

        else:  # measuring a single qubit

            meas_res = self.meas_behavior(key(meas_number))
            self._validate_measurement_result(meas_res)
            context_dic["_meas_number"] = meas_number + jnp.int32(1)

        # Associate the following in context_dic:
        # meas_result -> meas_res (int64 scalar)
        # QuantumState -> metric_data
        return (meas_res, metric_data)

    def handle_fuse(self, invalues, eqn, context_dic):

        # Associate the following in context_dic:
        # QubitArray -> size1 + size2
        return invalues[0] + invalues[1]

    def handle_slice(self, invalues, eqn, context_dic):

        start = jnp.max(jnp.array([invalues[1], 0]))
        stop = jnp.min(jnp.array([invalues[2], invalues[0]]))

        # Associate the following in context_dic:
        # QubitArray -> size
        return stop - start

    def handle_reset(self, invalues, eqn, context_dic):

        # Associate the following in context_dic:
        # QuantumState -> metric_data
        return invalues[-1]

    def handle_delete_qubits(self, invalues, eqn, context_dic):

        size, metric_data = invalues

        allocations_array, allocations_counter_index, invalid = metric_data
        allocations_array = allocations_array.at[allocations_counter_index].set(-size)
        allocations_counter_index += 1

        overflow = allocations_counter_index > jnp.int64(self._max_allocations)
        invalid = jnp.logical_or(invalid, overflow)
        jax.lax.cond(
            invalid,
            _warn,
            lambda *_: None,
            allocations_counter_index,
            self._max_allocations,
        )

        metric_data = (allocations_array, allocations_counter_index, invalid)

        # Associate the following in context_dic:
        # QubitArray -> size
        # QuantumState -> metric_data
        return metric_data

    def handle_parity(self, invalues, eqn, context_dic):
        """Handle the `jasp.parity` primitive"""

        # Parity is a classical operation on measurement results
        # Compute XOR and handle expectation
        expectation = eqn.params.get("expectation", 0)
        result = jnp.bitwise_xor(sum(invalues) % 2, expectation)

        return jnp.array(result, dtype=bool)


def _peak_allocated_qubits(dict_values) -> int:
    """Helper function to compute the peak number of allocated qubits from the allocation dictionary."""

    current, peak = 0, 0
    for delta in dict_values:
        current += delta
        peak = max(peak, current)
    return peak


def extract_num_qubits(res: Tuple, jaspr: Jaspr, _) -> dict:
    """Extract the number of allocated and deallocated qubits from the metric result."""

    metric = res[-1] if len(jaspr.outvars) > 1 else res

    allocations_array, allocations_counter_index, overflowed = metric

    if overflowed:
        raise ValueError(
            "The ``num_qubits`` metric computation overflowed "
            "the maximum number of allocations supported. "
        )

    alloc_dict = {
        f"alloc{i+1}": int(allocations_array[i])
        for i in range(int(allocations_counter_index))
    }
    dict_values = alloc_dict.values()

    total_allocated = sum(v for v in dict_values if v > 0)
    total_deallocated = -sum(v for v in dict_values if v < 0)
    peak_allocations = _peak_allocated_qubits(dict_values)
    finally_allocated = sum(dict_values)

    return {
        "total_allocated": total_allocated,
        "total_deallocated": total_deallocated,
        "peak_allocations": peak_allocations,
        "finally_allocated": finally_allocated,
    }


@lru_cache(int(1e5))
def get_num_qubits_profiler(
    jaspr: Jaspr, meas_behavior: Callable, max_allocations: int = 1000
) -> Tuple[Callable, None]:
    """
    Build a num qubits profiling computer for a given Jaspr.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr expression to profile.

    meas_behavior : Callable
        The measurement behavior function.

    max_allocations : int
        The maximum number of qubit allocations/deallocations supported by the profiler.
        Default is 1000.

    Returns
    -------
    Tuple[Callable, None]
        A num qubits profiler function and None as auxiliary data.

    """
    quantum_operations = get_quantum_operations(jaspr)
    profiling_dic = {quantum_operations[i]: i for i in range(len(quantum_operations))}

    if "measure" not in profiling_dic:
        profiling_dic["measure"] = -1

    num_qubits_metric = NumQubitsMetric(meas_behavior, profiling_dic, max_allocations)
    profiling_eqn_evaluator = make_profiling_eqn_evaluator(num_qubits_metric)
    jitted_evaluator = jax.jit(eval_jaxpr(jaspr, eqn_evaluator=profiling_eqn_evaluator))

    def num_qubits_profiler(*args):
        # Filter out types that are known to be static (https://github.com/eclipse-qrisp/Qrisp/issues/258)
        # Import here to avoid circular import issues
        from qrisp.operators import FermionicOperator, QubitOperator

        STATIC_TYPES = (str, QubitOperator, FermionicOperator, types.FunctionType)

        initial_metric_value = num_qubits_metric.initial_metric
        filtered_args = [
            x for x in args + (initial_metric_value,) if type(x) not in STATIC_TYPES
        ]
        return jitted_evaluator(*filtered_args)

    return num_qubits_profiler, None


def simulate_num_qubits(jaspr: Jaspr, *_, **__) -> dict:
    """Simulate num_qubits metric via actual simulation."""

    raise NotImplementedError(
        "Num qubits metric via simulation is not implemented yet."
    )
