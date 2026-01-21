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

from qrisp.jasp.interpreter_tools import (
    eval_jaxpr,
    make_profiling_eqn_evaluator_new,
)
from qrisp.jasp.interpreter_tools.interpreters.utilities import get_quantum_operations
from qrisp.jasp.jasp_expression import Jaspr

MAX_QUBITS = 1024


class DepthMetric:
    """
    A simple depth counting metric implementation.

    Parameters
    ----------
    profiling_dic : dict
        A dictionary mapping quantum operation names to their profiling indices.

    meas_behavior : Callable
        The measurement behavior function.

    """

    def __init__(self, profiling_dic: dict, meas_behavior: Callable):
        """Initialize the DepthMetric."""

        self.profiling_dic = profiling_dic
        self.meas_behavior = meas_behavior

        depth_array = jnp.zeros((MAX_QUBITS,), dtype=jnp.int64)
        global_depth = jnp.int64(0)
        self.metric_data = (depth_array, global_depth)

    def handle_measure(self, invalues, _):
        """Handle the `jasp.measure` primitive."""

        _, metric_data = invalues

        # At this stage we ignore measurement impact on depth.
        # We just fabricate a valid measurement result.
        # In qrisp/jasp it looks like measurement result is an int64 scalar.
        meas_res = jnp.int64(0)

        # Associate the following in context_dic:
        # meas_result -> meas_res (int64 scalar)
        # QuantumCircuit -> metric_data (depth_array, global_depth)
        return meas_res, metric_data

    def handle_get_qubit(self, invalues):
        """Handle the `jasp.get_qubit` primitive."""

        qubit_array_handle, index = invalues

        base_id, _ = qubit_array_handle
        base_id = jnp.asarray(base_id, dtype=jnp.int64)
        index = jnp.asarray(index, dtype=jnp.int64)

        # Associate the following in context_dic:
        # Qubit -> qubit_id (integer)
        return base_id + index

    def handle_get_size(self, invalues):
        """Handle the `jasp.get_size` primitive."""

        (qubit_array_handle,) = invalues

        _, size = qubit_array_handle
        size = jnp.asarray(size, dtype=jnp.int64)

        # Associate the following in context_dic:
        # size -> size_value (integer)
        return size

    def handle_create_qubits(self, invalues, context_dic):
        """Handle the `jasp.create_qubits` primitive."""

        size, metric_data = invalues

        # Allocate a fresh id range [size] starting from next_base_id
        base_id = context_dic.get("_depth_next_base_id", jnp.int64(0))
        base_id = jnp.asarray(base_id, dtype=jnp.int64)
        size = jnp.asarray(size, dtype=jnp.int64)
        context_dic["_depth_next_base_id"] = base_id + size
        qubit_array_handle = (base_id, size)

        # Associate the following in context_dic:
        # QubitArray -> qubit_array_handle (base_id, size)
        # QuantumCircuit -> metric_data (depth_array, global_depth)
        return qubit_array_handle, metric_data

    def handle_quantum_gate(self, invalues, _):
        """Handle the `jasp.quantum_gate` primitive."""

        *qubits, metric_data = invalues

        depth_array, global_depth = metric_data
        qubit_ids = jnp.asarray(qubits, dtype=jnp.int64)
        touched = jnp.take(depth_array, qubit_ids)
        start = jnp.max(touched)

        # TODO: determine actual duration of the gate
        # duration: 1 for now (or duration = op_depth if op.definition exists)
        end = start + 1

        # update all involved qubits to `end`
        depth_array = depth_array.at[qubit_ids].set(end)
        global_depth = jnp.maximum(global_depth, end)

        # Associate the following in context_dic:
        # QuantumCircuit -> metric_data (depth_array, global_depth)
        return depth_array, global_depth

    def handle_delete_qubits(self, invalues):
        """Handle the `jasp.delete_qubits` primitive."""

        _, metric_data = invalues

        # Associate the following in context_dic:
        # QuantumCircuit -> metric_data (depth_array, global_depth)
        return metric_data

    def handle_reset(self, invalues):
        """Handle the `jasp.reset` primitive."""

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

    quantum_operations = get_quantum_operations(jaspr)
    profiling_dic = {quantum_operations[i]: i for i in range(len(quantum_operations))}

    if "measure" not in profiling_dic:
        profiling_dic["measure"] = -1

    depth_metric = DepthMetric(profiling_dic, meas_behavior)
    profiling_eqn_evaluator = make_profiling_eqn_evaluator_new(depth_metric)
    jitted_evaluator = jax.jit(eval_jaxpr(jaspr, eqn_evaluator=profiling_eqn_evaluator))

    def depth_profiler(*args):

        # Import here to avoid circular imports
        from qrisp.operators import FermionicOperator, QubitOperator

        static_types = (str, QubitOperator, FermionicOperator, types.FunctionType)
        filtered_args = [
            x
            for x in list(args) + [depth_metric.metric_data]
            if type(x) not in static_types
        ]

        return jitted_evaluator(*filtered_args)

    return depth_profiler, None
