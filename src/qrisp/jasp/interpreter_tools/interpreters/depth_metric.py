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

from functools import lru_cache
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.random import key

from qrisp.jasp.interpreter_tools import (
    ContextDict,
    eval_jaxpr,
    make_profiling_eqn_evaluator,
)
from qrisp.jasp.interpreter_tools.interpreters.utilities import filter_static_types
from qrisp.jasp.jasp_expression import Jaspr
from qrisp.jasp.primitives import (
    AbstractQubitArray,
)

MAX_QUBITS = 1024


class DepthMetric:
    """
    A simple depth counting metric implementation.

    Parameters
    ----------
    meas_behavior : Callable
        The measurement behavior function.

    """

    def __init__(self, meas_behavior: Callable):
        """Initialize the DepthMetric."""

        self.meas_behavior = meas_behavior

        # As data structure in the context dictionary to keep track of depth,
        # we use an array of size MAX_QUBITS where each entry corresponds to the depth of that qubit.
        # The final depth is the maximum value in this array (retrieved at the end).
        #
        # Note 1: to speed up compilation time, it will be probably necessary to
        # use the same incrementation constants trick used for gate counting.
        #
        # Note 2: in case the algorithm uses more than MAX_QUBITS qubits,
        # this will obviously fail. We can later think of dynamic resizing strategies
        depth_array = jnp.zeros(MAX_QUBITS, dtype=jnp.int64)
        global_depth = jnp.int64(0)
        self.initial_metric = (depth_array, global_depth)

    def handle_create_qubits(self, invalues, context_dic: ContextDict):
        """Handle the `jasp.create_qubits` primitive."""

        size, metric_data = invalues

        # We create a fresh qubit array with `size` qubits in a new region of the depth_array.
        # This keeps into account the fact that `create_qubits` can be called multiple times.
        index_start = context_dic.get("_previous_qubit_array_end", jnp.int64(0))
        context_dic["_previous_qubit_array_end"] = index_start + size
        qubit_array_handle = (index_start, size)

        # Associate the following in context_dic:
        # QubitArray -> qubit_array_handle (index_start, size)
        # QuantumCircuit -> metric_data (depth_array, global_depth)
        return qubit_array_handle, metric_data

    def handle_get_qubit(self, invalues):
        """Handle the `jasp.get_qubit` primitive."""

        qubit_array_handle, get_qubit_index = invalues

        index_start, _ = qubit_array_handle

        # Associate the following in context_dic:
        # Qubit -> qubit_id (integer)
        return index_start + get_qubit_index

    def handle_get_size(self, invalues):
        """Handle the `jasp.get_size` primitive."""

        (qubit_array_handle,) = invalues

        _, size = qubit_array_handle

        # Associate the following in context_dic:
        # size -> size_value (integer)
        return size

    def handle_fuse(self, invalues):
        """Handle the `jasp.fuse` primitive."""

        raise NotImplementedError(
            "Fusing qubit arrays not yet supported in depth metric."
        )

    def handle_slice(self, invalues):
        """Handle the `jasp.slice` primitive."""

        raise NotImplementedError(
            "Slicing qubit arrays not yet supported in depth metric."
        )

    def handle_quantum_gate(self, invalues, _):
        """Handle the `jasp.quantum_gate` primitive."""

        *qubits, metric_data = invalues
        depth_array, global_depth = metric_data

        # Note that qubits have been already converted to their integer IDs
        # by the `jasp.get_qubit` primitive handler, which also keeps track
        # of the qubit array base index in `depth_array`.

        qubit_ids = jnp.asarray(qubits, dtype=jnp.int64)
        touched = jnp.take(depth_array, qubit_ids)
        start = jnp.max(touched)

        # TODO: I am assuming here that each gate has depth 1
        # This can be changed later to accommodate gates with different depths if needed
        # (in the same way as in gate counting metric)
        end = start + 1

        # update all involved qubits to `end`
        depth_array = depth_array.at[qubit_ids].set(end)
        global_depth = jnp.maximum(global_depth, end)

        # Associate the following in context_dic:
        # QuantumCircuit -> metric_data (depth_array, global_depth)
        return depth_array, global_depth

    def _validate_measurement_result(self, meas_res):
        """Validate that measurement result is a boolean."""

        if not isinstance(meas_res, bool) and meas_res.dtype != jnp.bool:
            raise ValueError(
                f"Measurement behavior must return a boolean, got {meas_res.dtype}"
            )

    def _measurement_body_fun(self, meas_number, i, acc):
        """Helper function for measuring qubit arrays."""

        meas_key = key(meas_number + i)
        meas_res = self.meas_behavior(meas_key)
        self._validate_measurement_result(meas_res)
        return acc + (1 << i) * meas_res

    def handle_measure(self, invalues, _):
        """Handle the `jasp.measure` primitive."""

        _, metric_data = invalues

        # TODO: At this stage we ignore measurement impact on depth.
        # We just fabricate a valid measurement result.
        # In qrisp/jasp it looks like measurement result is an int64 scalar.
        meas_res = jnp.int64(0)

        # Associate the following in context_dic:
        # meas_result -> meas_res (int64 scalar)
        # QuantumCircuit -> metric_data (depth_array, global_depth)
        return meas_res, metric_data

    def handle_reset(self, invalues):
        """Handle the `jasp.reset` primitive."""

        _, metric_data = invalues

        # Associate the following in context_dic:
        # QuantumCircuit -> metric_data (depth_array, global_depth)
        return metric_data

    def handle_delete_qubits(self, invalues):
        """Handle the `jasp.delete_qubits` primitive."""

        _, metric_data = invalues

        # Associate the following in context_dic:
        # QuantumCircuit -> metric_data (depth_array, global_depth)
        return metric_data


def extract_depth(res: Tuple, jaspr: Jaspr, _) -> int:
    """Extract depth from the profiling result."""

    if len(jaspr.outvars) > 1:
        depth = res[-1][1]
    else:
        depth = res[1]

    return int(depth) if hasattr(depth, "__int__") else depth


@lru_cache(int(1e5))
def get_depth_profiler(jaspr: Jaspr, meas_behavior: Callable) -> Tuple[Callable, None]:
    """
    Build a depth profiling computer for a given Jaspr.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr expression to profile.

    meas_behavior : Callable
        The measurement behavior function.

    Returns
    -------
    Tuple[Callable, None]
        A depth profiler function and None as auxiliary data.

    """

    depth_metric = DepthMetric(meas_behavior)
    profiling_eqn_evaluator = make_profiling_eqn_evaluator(depth_metric)
    jitted_evaluator = jax.jit(eval_jaxpr(jaspr, eqn_evaluator=profiling_eqn_evaluator))

    def depth_profiler(*args):
        filtered_args = filter_static_types(args, metric_instance=depth_metric)()
        return jitted_evaluator(*filtered_args)

    return depth_profiler, None


def simulate_depth(jaspr: Jaspr, *_, **__) -> int:
    """Simulate depth metric via actual simulation."""

    raise NotImplementedError("Depth metric via simulation is not implemented yet.")
