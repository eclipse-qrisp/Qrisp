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

MAX_QUBITS = 5


# TODO: The final architecture for depth counting might differ from this.
class DepthMetric:
    """A simple depth counting metric implementation."""

    def __init__(self, profiling_dic: dict, meas_behavior: Callable):
        """Initialize the DepthMetric."""

        depth_vec = jnp.zeros((MAX_QUBITS,), dtype=jnp.int32)
        global_depth = jnp.int32(0)
        self.final_arg = (depth_vec, global_depth)

        self.profiling_dic = profiling_dic
        self.meas_behavior = meas_behavior

    def handle_measure(self, invalues):
        """Handle the `jasp.measure` primitive."""

        measured_obj, depth_state = invalues
        print(f"measured_obj: {measured_obj}")
        print(f"depth_state: {depth_state}")

        # At this stage we ignore measurement impact on depth.
        # We just fabricate a valid measurement result.
        # In qrisp/jasp it looks like measurement result is an int64 scalar.
        meas_res = jnp.int64(0)

        return (meas_res, depth_state)

    def handle_get_qubit(self, invalues):
        """Handle the `jasp.get_qubit` primitive."""

        qubit_array_handle, index = invalues
        print(f"qubit_array_handle: {qubit_array_handle}")
        print(f"index: {index}")

        base_id, _ = qubit_array_handle

        # A Qubit is represented by its (virtual) integer id.
        qubit_id = jnp.asarray(base_id + index, dtype=jnp.int32)
        print(f"qubit_id: {qubit_id}")
        return qubit_id

    def handle_create_qubits(self, invalues, context_dic):
        """Handle the `jasp.create_qubits` primitive."""

        size, depth_state = invalues
        print(f"size: {size}")
        print(f"depth_state: {depth_state}")

        # Allocate a fresh id range [size] starting from next_base_id
        base_id = context_dic.get("_depth_next_base_id", jnp.int32(0))
        base_id = jnp.asarray(base_id, dtype=jnp.int32)
        print(f"base_id: {base_id}")

        size = jnp.asarray(size, dtype=jnp.int32)
        print(f"size after asarray: {size}")

        context_dic["_depth_next_base_id"] = base_id + size

        qubit_array_handle = (base_id, size)
        print(f"qubit_array_handle: {qubit_array_handle}")

        return (qubit_array_handle, depth_state)

    def handle_quantum_gate(self, invalues):
        """Handle the `jasp.quantum_gate` primitive."""

        *qubits, depth_state = invalues
        print(f"qubits: {qubits}")
        print(f"depth_state: {depth_state}")

        depth_vec, global_depth = depth_state
        print(f"depth_vec: {depth_vec}")
        print(f"global_depth: {global_depth}")

        qubit_ids = jnp.array(qubits, dtype=jnp.int32)
        print(f"qubit_ids: {qubit_ids}")
        touched = jnp.take(depth_vec, qubit_ids)
        print(f"touched: {touched}")
        start = jnp.max(touched)
        print(f"start: {start}")

        # TODO: determine actual duration of the gate
        # duration: 1 for now (or duration = op_depth if op.definition exists)
        end = start + 1

        # update all involved qubits to `end`
        depth_vec = depth_vec.at[qubit_ids].set(end)
        print(f"updated depth_vec: {depth_vec}")

        global_depth = jnp.maximum(global_depth, end)
        print(f"updated global_depth: {global_depth}")

        return (depth_vec, global_depth)

    def handle_delete_qubits(self, invalues):
        """Handle the `jasp.delete_qubits` primitive."""

        _qubitarray_handle, depth_state = invalues
        print(f"_qubitarray_handle: {_qubitarray_handle}")
        print(f"depth_state: {depth_state}")
        return depth_state

    def handle_reset(self, invalues):
        """Handle the `jasp.reset` primitive."""

        _qubitarray_handle, depth_state = invalues
        print(f"_qubitarray_handle: {_qubitarray_handle}")
        print(f"depth_state: {depth_state}")
        return depth_state


def extract_depth(res: Tuple, jaspr: Jaspr) -> int:
    """Extract depth from the profiling result."""

    if len(jaspr.outvars) > 1:
        depth = res[-1][1]
    else:
        depth = res[1]

    return int(depth) if hasattr(depth, "__int__") else depth


@lru_cache(int(1e5))
def get_depth_profiler(jaspr: Jaspr, meas_behavior: Callable) -> Callable:
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
    Callable
        A depth profiler function.

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
        # TODO: is this needed for depth metric?
        from qrisp.operators import FermionicOperator, QubitOperator

        static_types = (str, QubitOperator, FermionicOperator, types.FunctionType)
        filtered_args = [
            x
            for x in list(args) + [depth_metric.final_arg]
            if type(x) not in static_types
        ]

        return jitted_evaluator(*filtered_args)

    return depth_profiler
