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
from typing import Any, Callable, Iterator, List, Tuple

import jax
import jax.numpy as jnp
from jax.random import key
from jax.typing import ArrayLike

from qrisp.circuit.instruction import Instruction
from qrisp.jasp.interpreter_tools import (
    BaseMetric,
    eval_jaxpr,
    make_profiling_eqn_evaluator,
)
from qrisp.jasp.interpreter_tools.interpreters.utilities import (
    get_quantum_operations,
    is_abstract,
)
from qrisp.jasp.jasp_expression import Jaspr
from qrisp.jasp.primitives import (
    AbstractQubitArray,
)


def _apply_duration_on_qubits(
    depth_array: jnp.ndarray,
    current_depth: jnp.ndarray,
    qubit_ids: List,
    duration: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply a gate of given duration on specified qubits,
    updating the depth array and current depth accordingly.
    """
    qubit_ids_arr = jnp.asarray(qubit_ids, dtype=jnp.int64)
    touched = jnp.take(depth_array, qubit_ids_arr)
    start = jnp.max(touched)
    end = start + jnp.int64(duration)

    depth_array = depth_array.at[qubit_ids_arr].set(end)
    current_depth = jnp.maximum(current_depth, end)
    return depth_array, current_depth


def _iter_definition_ops(definition) -> Iterator[Tuple[Instruction, List[Any]]]:
    """Iterate over operations in a quantum circuit definition."""

    for instruction in definition.data:
        qubits = getattr(instruction, "qubits", None)
        if qubits is None:
            raise TypeError(
                f"Unsupported definition.data entry type {type(instruction)}: {instruction}"
            )
        yield instruction, list(qubits)


# This function creates a lookup table (mapping table) with length `table_size`
# (this must be a concrete integer because of JAX and <= MAX_QUBITS)
# that maps logical indices to global qubit ids in the depth array.
def _create_lookup_table(idx_start: ArrayLike, table_size: ArrayLike) -> ArrayLike:
    """Create a lookup table for qubit IDs starting from idx_start."""

    return idx_start + jnp.arange(table_size, dtype=jnp.int64)


def _as_qubit_array_handle(x) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert input to a qubit array handle (array, size) tuple."""

    if isinstance(x, tuple) and len(x) == 2:
        return jnp.asarray(x[0], dtype=jnp.int64), jnp.asarray(x[1], dtype=jnp.int64)

    return jnp.asarray([x], dtype=jnp.int64), jnp.asarray(jnp.int64(1))


# The computation will fail anyway if this is triggered,
# but this way we can print an informative message before the failure happens.
def _warn_overflow(idx_end, max_qubits):
    """Helper function to print a warning when the depth metric computation overflows."""
    jax.debug.print(
        (
            "ERROR: Depth metric computation overflowed: tried to create qubits with "
            "global indices up to {idx_end}, but the maximum supported is {max_qubits}. "
            "Consider increasing the `max_qubits` parameter for depth profiling."
        ),
        idx_end=idx_end,
        max_qubits=max_qubits,
    )


# The computation currently does not fail if the slice operation goes out of bounds,
# but this way we can print an informative message.
def _warn_slice(idx_slice1, idx_slice2):
    """Helper function to print a warning when the slice operation goes out of bounds."""
    jax.debug.print(
        (
            "ERROR: Slice operation with start index {idx_slice1} and stop index {idx_slice2} "
            "is out of bounds or is currently not implemented. Please adjust the slice "
            "indices to provide a positive size and ensure they are within bounds."
        ),
        idx_slice1=idx_slice1,
        idx_slice2=idx_slice2,
    )


# The computation currently does not fail if the delete/reset operation is called,
# but this way we can print an informative message.
def _warn_not_implemented(primitive_name):
    """Helper function to print a warning when the delete/reset operation is called."""
    jax.debug.print(
        (
            "Warning: {primitive_name} primitive for "
            "depth metric is currently not implemented. "
        ),
        primitive_name=primitive_name,
    )


class DepthMetric(BaseMetric):
    """
    A metric implementation that computes the circuit depth of a Jaspr.

    Parameters
    ----------
    meas_behavior : Callable
        The measurement behavior function.

    profiling_dic : dict
        The profiling dictionary mapping quantum operations to indices.

    max_qubits : int, optional
        The maximum number of qubits supported for depth computation. Default is 1024.

    """

    def __init__(
        self, meas_behavior: Callable, profiling_dic: dict, max_qubits: int = 1024
    ):
        """Initialize the DepthMetric."""

        super().__init__(meas_behavior=meas_behavior, profiling_dic=profiling_dic)

        # Define a maximum number of qubits to track depth for
        # This can be adjusted as needed (trade-off between memory and flexibility).
        # Unfortunately, JAX does not support dynamic arrays (yet).
        # Ideally, we would implement dynamic resizing in the future
        self._max_qubits = max_qubits

    @property
    def max_qubits(self) -> int:
        """Return the maximum number of qubits supported."""
        return self._max_qubits

    def handle_create_qubits(self, invalues, eqn, context_dic):

        size, metric_data = invalues
        depth_array, global_depth, previous_size, invalid = metric_data

        idx_start = previous_size
        idx_end = idx_start + size

        overflow = idx_end > jnp.int64(self.max_qubits)
        invalid = jnp.logical_or(invalid, overflow)
        jax.lax.cond(invalid, _warn_overflow, lambda *_: None, idx_end, self.max_qubits)

        previous_size = idx_end

        table_size = size if not is_abstract(size) else self.max_qubits
        qubit_ids_table = _create_lookup_table(idx_start, table_size)

        qubit_table_handle = (qubit_ids_table, size)

        # Associate the following in context_dic:
        # QubitArray -> qubit_table_handle (qubit_ids_table, size)
        # QuantumCircuit -> metric_data
        return qubit_table_handle, (
            depth_array,
            global_depth,
            previous_size,
            invalid,
        )

    def handle_get_qubit(self, invalues, eqn, context_dic):

        qubit_table_handle, qubit_index = invalues

        qubit_ids_table, _ = qubit_table_handle

        # Associate the following in context_dic:
        # Qubit -> qubit_id (integer tracer/array scalar)
        return qubit_ids_table[qubit_index]

    def handle_get_size(self, invalues, eqn, context_dic):

        (qubit_table_handle,) = invalues

        _, size = qubit_table_handle

        # Associate the following in context_dic:
        # size -> size_value (integer tracer/scalar)
        return size

    def handle_quantum_gate(self, invalues, eqn, context_dic):

        depth_array, current_depth, previous_size, invalid = invalues[-1]

        op = eqn.params["gate"]
        # Qubits have been already converted to their integer (possibly traced)
        # indices by the `jasp.get_qubit` primitive handler
        qubits = list(invalues[: op.num_qubits])

        if not getattr(op, "definition", None):
            depth_array, current_depth = _apply_duration_on_qubits(
                depth_array=depth_array,
                current_depth=current_depth,
                qubit_ids=qubits,
                duration=jnp.int64(1),
            )
            return depth_array, current_depth, previous_size, invalid

        transpiled_definition = op.definition.transpile()
        concrete_qubits = list(transpiled_definition.qubits)

        # Here, `concrete_qubits` are Qubit objects, while `qubits` are retrieved
        # from the invalues and are integer tracers/array scalars.
        # With this `qubit_map` below, we can map the concrete qubits in the
        # op definition to their corresponding indices in the depth array.
        qubit_map = dict(zip(concrete_qubits, qubits, strict=True))

        for _, inner_def_qubits in _iter_definition_ops(transpiled_definition):

            if not inner_def_qubits:
                continue

            inner_ids = [qubit_map[qb] for qb in inner_def_qubits]

            depth_array, current_depth = _apply_duration_on_qubits(
                depth_array=depth_array,
                current_depth=current_depth,
                qubit_ids=inner_ids,
                duration=jnp.int64(1),
            )

        # Associate the following in context_dic:
        # QuantumCircuit -> metric_data
        return depth_array, current_depth, previous_size, invalid

    def handle_measure(self, invalues, eqn, context_dic):

        target, metric_data = invalues

        depth_array, current_depth, previous_size, invalid = metric_data
        # We keep a measurement counter only to generate unique keys.
        meas_number = context_dic.get("_depth_meas_number", jnp.int32(0))

        if isinstance(eqn.invars[0].aval, AbstractQubitArray):

            _, size = target

            def body_fun(i, acc):
                return self._measurement_body_fun(meas_number, i, acc)

            meas_res = jax.lax.fori_loop(0, size, body_fun, jnp.int64(0))
            context_dic["_depth_meas_number"] = meas_number + size

        else:  # measuring a single qubit

            meas_res = self.meas_behavior(key(meas_number))
            self._validate_measurement_result(meas_res)
            context_dic["_depth_meas_number"] = meas_number + jnp.int32(1)

        # Associate the following in context_dic:
        # meas_result -> meas_res (int64 scalar)
        # QuantumCircuit -> metric_data
        return (
            meas_res,
            (depth_array, current_depth, previous_size, invalid),
        )

    def handle_fuse(self, invalues, eqn, context_dic):

        invalue1, invalue2 = invalues

        arr_a, size_a = _as_qubit_array_handle(invalue1)
        arr_b, size_b = _as_qubit_array_handle(invalue2)

        size_out = jnp.minimum(size_a + size_b, jnp.int64(self.max_qubits))

        indices_size = size_out if not is_abstract(size_out) else self.max_qubits
        idxs = jnp.arange(indices_size, dtype=jnp.int64)

        # positions in the fused array that correspond to A
        a_pos = jnp.clip(idxs, 0, jnp.maximum(size_a - 1, 0))
        a_vals = arr_a[a_pos]

        # positions in the fused array that correspond to B
        b_pos = idxs - size_a
        b_pos = jnp.clip(b_pos, 0, jnp.maximum(size_b - 1, 0))
        b_vals = arr_b[b_pos]

        mask_a = idxs < size_a
        qubit_array_out = jnp.where(mask_a, a_vals, b_vals)

        # Associate the following in context_dic:
        # QubitArray -> (qubit_array_out, size_out)
        return (qubit_array_out, size_out)

    def handle_slice(self, invalues, eqn, context_dic):

        (qubit_array, size), start_inv, stop_inv = invalues

        start = jnp.maximum(start_inv, jnp.int64(0))
        stop = jnp.minimum(stop_inv, size)
        stop = jnp.maximum(stop, start)

        size_out = stop - start
        jax.lax.cond(size_out <= 0, _warn_slice, lambda *_: None, start_inv, stop_inv)

        table_size = size_out if not is_abstract(size_out) else self.max_qubits

        array_idx_mapping = _create_lookup_table(start, table_size)
        qubit_array_out = qubit_array[array_idx_mapping]

        # Associate the following in context_dic:
        # QubitArray -> (qubit_array_out, size_out)
        return (qubit_array_out, size_out)

    def handle_reset(self, invalues, eqn, context_dic):

        _, metric_data = invalues

        _warn_not_implemented("reset")

        # Associate the following in context_dic:
        # QuantumCircuit -> metric_data
        return metric_data

    def handle_delete_qubits(self, invalues, eqn, context_dic):

        _, metric_data = invalues

        _warn_not_implemented("delete_qubits")

        # Associate the following in context_dic:
        # QuantumCircuit -> metric_data
        return metric_data

    def handle_parity(self, invalues, eqn, context_dic):
        """Handle the `jasp.parity` primitive"""

        # Parity is a classical operation on measurement results
        # Compute XOR and handle expectation
        expectation = eqn.params.get("expectation", 0)
        result = jnp.bitwise_xor(sum(invalues) % 2, expectation)

        return jnp.array(result, dtype=bool)


def extract_depth(res: Tuple, jaspr: Jaspr, _) -> int:
    """Extract depth from the profiling result."""

    metric = res[-1] if len(jaspr.outvars) > 1 else res
    _, depth, _, overflowed = metric

    if overflowed:
        raise ValueError(
            "The depth metric computation overflowed the maximum number of qubits supported."
        )

    return int(depth)


@lru_cache(int(1e5))
def get_depth_profiler(
    jaspr: Jaspr, meas_behavior: Callable, max_qubits: int = 1024
) -> Tuple[Callable, None]:
    """
    Build a depth profiling computer for a given Jaspr.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr expression to profile.

    meas_behavior : Callable
        The measurement behavior function.

    max_qubits : int, optional
        The maximum number of qubits supported for depth computation. Default is 1024.

    Returns
    -------
    Tuple[Callable, None]
        A depth profiler function and None as auxiliary data.

    """
    quantum_operations = get_quantum_operations(jaspr)
    profiling_dic = {quantum_operations[i]: i for i in range(len(quantum_operations))}

    if "measure" not in profiling_dic:
        profiling_dic["measure"] = -1

    depth_metric = DepthMetric(meas_behavior, profiling_dic, max_qubits)
    profiling_eqn_evaluator = make_profiling_eqn_evaluator(depth_metric)
    jitted_evaluator = jax.jit(eval_jaxpr(jaspr, eqn_evaluator=profiling_eqn_evaluator))

    def depth_profiler(*args):
        # Filter out types that are known to be static (https://github.com/eclipse-qrisp/Qrisp/issues/258)
        # Import here to avoid circular import issues
        from qrisp.operators import FermionicOperator, QubitOperator

        STATIC_TYPES = (str, QubitOperator, FermionicOperator, types.FunctionType)

        # To speed up compilation time, it will be probably necessary to
        # use the same incrementation constants trick used for gate counting.

        # Here is the explanation of the metric data structure:
        #
        # - depth_array: jnp.ndarray of shape (max_qubits,) that keeps track
        #   of the depth of each qubit.
        #
        # - current_depth: jnp.ndarray scalar that keeps track of the current
        #   maximum depth of the circuit.
        #
        # - previous_size: jnp.ndarray scalar that keeps track of the
        #   next available index in the depth_array for newly created qubits.
        #   When we create new qubits, we assign them global indices starting from this value,
        #   and then we update it by the number of qubits created.
        #
        # - invalid: jnp.ndarray boolean that indicates whether the depth computation
        #   has overflowed the maximum number of qubits supported.
        depth_array = jnp.zeros(max_qubits, dtype=jnp.int64)
        current_depth = jnp.int64(0)
        previous_size = jnp.int64(0)
        invalid = jnp.bool_(False)

        initial_metric_value = (
            depth_array,
            current_depth,
            previous_size,
            invalid,
        )

        filtered_args = [
            x for x in args + (initial_metric_value,) if type(x) not in STATIC_TYPES
        ]
        return jitted_evaluator(*filtered_args)

    return depth_profiler, None


def simulate_depth(jaspr: Jaspr, *_, **__) -> int:
    """Simulate depth metric via actual simulation."""

    raise NotImplementedError("Depth metric via simulation is not implemented yet.")
