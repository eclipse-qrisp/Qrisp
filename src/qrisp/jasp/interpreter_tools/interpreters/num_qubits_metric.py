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

"""Defines NumQubitsMetric, a profiling metric that tracks qubit allocations and deallocations in a Jaspr."""

from collections.abc import Callable
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.random import key

from qrisp._cache_config import qrisp_lru_compilation_cache
from qrisp.jasp.interpreter_tools.interpreters.profiling_interpreter import (
    BaseMetric,
    build_metric_profiler,
)
from qrisp.jasp.jasp_expression import Jaspr
from qrisp.jasp.primitives import (
    AbstractQubitArray,
)


class NumQubitsMetric(BaseMetric):
    """A metric implementation that tracks the qubit allocations in a Jaspr.

    The metric only keeps running counters, so its cost does not depend on how
    many allocations and deallocations the program performs.

    Parameters
    ----------
    meas_behavior : Callable
        The measurement behavior function.

    """

    def __init__(self, meas_behavior: Callable):
        """Initialize the NumQubitsMetric."""
        super().__init__(meas_behavior=meas_behavior)

    def cache_key(self) -> Tuple[Callable]:
        return (self.meas_behavior,)

    @classmethod
    def from_cache_key(cls, cache_key) -> "NumQubitsMetric":
        (meas_behavior,) = cache_key
        return cls(meas_behavior)

    def initial_metric(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        # Here is the explanation of the metric data structure:

        # - currently_allocated: the number of qubits allocated at this point of the computation.
        #
        # - peak_allocations: the largest value `currently_allocated` has reached so far.
        #
        # - total_allocated: the summed size of all allocations performed so far.
        #
        # - total_deallocated: the summed size of all deallocations performed so far.
        currently_allocated = jnp.int64(0)
        peak_allocations = jnp.int64(0)
        total_allocated = jnp.int64(0)
        total_deallocated = jnp.int64(0)

        return currently_allocated, peak_allocations, total_allocated, total_deallocated

    ##############################################################
    ### Quantum primitive handlers
    ##############################################################

    def handle_create_qubits(self, invalues, eqn, context_dic):

        size, metric_data = invalues

        currently_allocated, peak_allocations, total_allocated, total_deallocated = metric_data

        currently_allocated = currently_allocated + size
        peak_allocations = jnp.maximum(peak_allocations, currently_allocated)
        total_allocated = total_allocated + size

        metric_data = (currently_allocated, peak_allocations, total_allocated, total_deallocated)

        # Associate the following in context_dic:
        # QubitArray -> size
        # QuantumState -> metric_data
        return (size, metric_data)

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

        _, metric_data = invalues

        # We keep a measurement counter only to generate unique keys.
        meas_number = context_dic.get("_meas_number", jnp.int32(0))

        if isinstance(eqn.invars[0].aval, AbstractQubitArray):

            def body_fun(i, acc):
                return self._measurement_body_fun(meas_number, i, acc)

            meas_res = jax.lax.fori_loop(0, invalues[0], body_fun, jnp.int64(0))
            context_dic["_meas_number"] = meas_number + invalues[0]

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

        currently_allocated, peak_allocations, total_allocated, total_deallocated = metric_data

        currently_allocated = currently_allocated - size
        total_deallocated = total_deallocated + size

        metric_data = (currently_allocated, peak_allocations, total_allocated, total_deallocated)

        # Associate the following in context_dic:
        # QuantumState -> metric_data
        return metric_data

    def handle_parity(self, invalues, eqn, context_dic):

        # Parity is a classical operation on measurement results
        # Compute XOR and handle expectation
        expectation = eqn.params.get("expectation", 0)
        result = jnp.bitwise_xor(sum(invalues) % 2, expectation)

        return jnp.array(result, dtype=bool)


def extract_num_qubits(res: Tuple, jaspr: Jaspr, _) -> dict:
    """Extract the number of allocated and deallocated qubits from the metric result."""
    metric = res[-1] if len(jaspr.outvars) > 1 else res

    currently_allocated, peak_allocations, total_allocated, total_deallocated = metric

    return {
        "total_allocated": int(total_allocated),
        "total_deallocated": int(total_deallocated),
        "peak_allocations": int(peak_allocations),
        "finally_allocated": int(currently_allocated),
    }


# LRU cache controlled by QRISP_COMPILATION_CACHE_SIZE env var
@qrisp_lru_compilation_cache
def get_num_qubits_profiler(
    jaspr: Jaspr,
    meas_behavior: Callable,
    max_allocations: int | None = None,
    callback_threshold: int | None = None,
) -> tuple[Callable, None]:
    """Build a num qubits profiling computer for a given Jaspr.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr expression to profile.

    meas_behavior : Callable
        The measurement behavior function.

    max_allocations : int | None, optional
        Deprecated and ignored. Kept in this position so that existing positional
        and keyword calls keep their meaning. The public entry points warn about it.

    callback_threshold : int | None, optional
        Minimum value of ``call_count * inlined_eqn_count`` required to
        trigger ``jax.pure_callback`` wrapping.  ``None`` (default)
        disables callbacks entirely (fastest execution).  ``0`` wraps
        every reused sub-jaxpr (fastest compilation).

    Returns
    -------
    Tuple[Callable, None]
        A num qubits profiler function and None as auxiliary data.

    """
    num_qubits_metric = NumQubitsMetric(meas_behavior)

    return build_metric_profiler(jaspr, num_qubits_metric, callback_threshold), None


def simulate_num_qubits(jaspr: Jaspr, *_, **__) -> dict:
    """Simulate num_qubits metric via actual simulation."""
    raise NotImplementedError("Num qubits metric via simulation is not implemented yet.")
