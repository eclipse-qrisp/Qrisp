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

"""Defines DepthMetric, a profiling metric that computes the circuit depth of a Jaspr."""

from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from qrisp._cache_config import qrisp_lru_compilation_cache
from qrisp.circuit.instruction import Instruction
from qrisp.jasp.interpreter_tools.interpreters.profiling_interpreter import (
    BaseMetric,
    build_metric_profiler,
    normalize_slice_bounds,
)
from qrisp.jasp.interpreter_tools.interpreters.utilities import (
    is_abstract,
)
from qrisp.jasp.jasp_expression import Jaspr
from qrisp.jasp.primitives import (
    AbstractQubitArray,
)


def _apply_duration_on_qubits(
    depth_array: jnp.ndarray,
    current_depth: jnp.ndarray,
    qubit_ids: list,
    duration: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply a gate of given duration on the given qubits.

    The gate starts once all its qubits are free, and ends ``duration`` later.

    Parameters
    ----------
    depth_array : jnp.ndarray
        The time at which each qubit becomes free.
    current_depth : jnp.ndarray
        The depth of the circuit so far.
    qubit_ids : list
        The global ids of the qubits the gate acts on.
    duration : jnp.ndarray
        The duration of the gate.

    Returns
    -------
    depth_array : jnp.ndarray
        The updated time at which each qubit becomes free.
    current_depth : jnp.ndarray
        The updated depth of the circuit.

    """
    qubit_ids_arr = jnp.asarray(qubit_ids, dtype=jnp.int64)
    touched = jnp.take(depth_array, qubit_ids_arr)
    start = jnp.max(touched)
    end = start + jnp.int64(duration)

    depth_array = depth_array.at[qubit_ids_arr].set(end)
    current_depth = jnp.maximum(current_depth, end)
    return depth_array, current_depth


def _iter_definition_ops(definition) -> Iterator[tuple[Instruction, list[Any]]]:
    """Iterate over operations in a quantum circuit definition.

    Parameters
    ----------
    definition : QuantumCircuit
        The definition of a gate.

    Yields
    ------
    tuple[Instruction, list[Any]]
        Each instruction of the definition with the qubits it acts on.

    Raises
    ------
    TypeError
        If an entry of the definition has no qubits.

    """
    for instruction in definition.data:
        qubits = getattr(instruction, "qubits", None)
        if qubits is None:
            raise TypeError(f"Unsupported definition.data entry type {type(instruction)}: {instruction}")
        yield instruction, list(qubits)


# This function creates a lookup table (mapping table) with length `table_size`
# (this must be a concrete integer because of JAX and <= MAX_QUBITS)
# that maps logical indices to global qubit ids in the depth array.
def _create_lookup_table(idx_start: jax.Array | int, table_size: ArrayLike) -> jax.Array:
    """Create a lookup table for qubit IDs starting from idx_start.

    Parameters
    ----------
    idx_start : jax.Array or int
        The global id of the first qubit.
    table_size : ArrayLike
        The number of entries, which must be concrete.

    Returns
    -------
    jax.Array
        The global ids ``idx_start, idx_start + 1, ...``.

    """
    return idx_start + jnp.arange(table_size, dtype=jnp.int64)


def _as_qubit_array_handle(x) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert a QubitArray handle or a single qubit id to a ``(table, size)`` handle.

    Parameters
    ----------
    x : tuple or ArrayLike
        A QubitArray handle ``(table, size)``, or the global id of a single qubit.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        The lookup table and the size of the qubit array.

    """
    if isinstance(x, tuple):
        table, size = x
        return jnp.asarray(table, dtype=jnp.int64), jnp.asarray(size, dtype=jnp.int64)

    return jnp.asarray([x], dtype=jnp.int64), jnp.asarray(jnp.int64(1))


def _fuse_qubit_tables(handle_a: tuple, handle_b: tuple, max_qubits: int) -> tuple[jax.Array, jax.Array]:
    """Concatenate the lookup tables of two qubit arrays.

    Parameters
    ----------
    handle_a : tuple
        The ``(table, size)`` handle of the first qubit array.
    handle_b : tuple
        The ``(table, size)`` handle of the second qubit array.
    max_qubits : int
        The maximum number of qubits supported, which bounds the fused size.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        The ``(table, size)`` handle of the fused qubit array.

    """
    arr_a, size_a = handle_a
    arr_b, size_b = handle_b

    size_out = jnp.minimum(size_a + size_b, jnp.int64(max_qubits))

    indices_size = size_out if not is_abstract(size_out) else max_qubits
    idxs = jnp.arange(indices_size, dtype=jnp.int64)

    # Entries before size_a come from A, the following ones from B
    a_vals = arr_a[jnp.clip(idxs, 0, jnp.maximum(size_a - 1, 0))]
    b_vals = arr_b[jnp.clip(idxs - size_a, 0, jnp.maximum(size_b - 1, 0))]

    return jnp.where(idxs < size_a, a_vals, b_vals), size_out


def _gate_layers(op, qubits: list) -> Iterator[list]:
    """Yield the qubit ids of each operation that a gate occupies in time.

    A gate without a definition occupies its qubits once. A gate with a definition
    occupies them as the operations of its transpiled definition do.

    Parameters
    ----------
    op : Operation
        The applied gate.
    qubits : list
        The global ids of the qubits the gate acts on.

    Yields
    ------
    list
        The global ids of the qubits of each operation.

    """
    if not getattr(op, "definition", None):
        yield qubits
        return

    transpiled_definition = op.definition.transpile()

    # The definition refers to its own Qubit objects: map them to the global
    # ids (integer tracers/array scalars) of the qubits the gate acts on.
    qubit_map = dict(zip(transpiled_definition.qubits, qubits, strict=True))

    for _, inner_def_qubits in _iter_definition_ops(transpiled_definition):
        if inner_def_qubits:
            yield [qubit_map[qb] for qb in inner_def_qubits]


# The computation will fail anyway if this is triggered,
# but this way we can print an informative message before the failure happens.
def _warn_overflow(idx_end, max_qubits):
    """Print an error message when the depth metric computation overflows.

    Parameters
    ----------
    idx_end : ArrayLike
        The first free global id after the allocation that overflowed.
    max_qubits : int
        The maximum number of qubits supported.

    """
    jax.debug.print(
        (
            "ERROR: Depth metric computation overflowed: tried to create qubits with "
            "global indices up to {idx_end}, but the maximum supported is {max_qubits}. "
            "Consider increasing the `max_qubits` parameter for depth profiling."
        ),
        idx_end=idx_end,
        max_qubits=max_qubits,
    )


class DepthMetric(BaseMetric):
    """A metric implementation that computes the circuit depth of a Jaspr.

    Parameters
    ----------
    meas_behavior : Callable
        The measurement behavior function.

    max_qubits : int, optional
        The maximum number of qubits supported for depth computation. Default is 1024.

    """

    def __init__(self, meas_behavior: Callable, max_qubits: int = 1024):
        """Initialize the DepthMetric."""
        super().__init__(meas_behavior=meas_behavior)

        # Define a maximum number of qubits to track depth for
        # This can be adjusted as needed (trade-off between memory and flexibility).
        # Unfortunately, JAX does not support dynamic arrays (yet).
        # Ideally, we would implement dynamic resizing in the future
        self._max_qubits: int = max_qubits

    @property
    def max_qubits(self) -> int:
        """Return the maximum number of qubits supported."""
        return self._max_qubits

    @classmethod
    def from_cache_key(cls, cache_key) -> "DepthMetric":
        """Rebuild the metric from its measurement behavior and maximum number of qubits."""
        meas_behavior, max_qubits = cache_key
        return cls(meas_behavior, max_qubits)

    def cache_key(self) -> tuple[Callable, int]:
        """Return the measurement behavior and the maximum number of qubits as a hashable key."""
        return (self.meas_behavior, self.max_qubits)

    def initial_metric(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return a circuit of depth zero with no qubits allocated."""
        # To speed up compilation time, it will be probably necessary to
        # use the same incrementation constants trick used for gate counting.

        # Here is the explanation of the metric data structure:
        #
        # - depth_array: jnp.ndarray of shape (max_qubits,) that keeps track
        #   of the depth of each qubit.
        #
        # - current_depth: integer scalar that keeps track of the current
        #   maximum depth of the circuit.
        #
        # - previous_size: integer scalar that keeps track of the
        #   next available index in the depth_array for newly created qubits.
        #   When we create new qubits, we assign them global indices starting from this value,
        #   and then we update it by the number of qubits created.
        #
        # - invalid: boolean scalar that indicates whether the depth computation
        #   has overflowed the maximum number of qubits supported.
        #
        # - meas_count: integer scalar with the number of qubits measured so far,
        #   used to number the measurements (see BaseMetric._sample_measurement).
        depth_array = jnp.zeros(self.max_qubits, dtype=jnp.int64)
        current_depth = jnp.int64(0)
        previous_size = jnp.int64(0)
        invalid = jnp.bool_(False)
        meas_count = jnp.int64(0)

        return depth_array, current_depth, previous_size, invalid, meas_count

    ##############################################################
    ### Quantum primitive handlers
    ##############################################################

    def _reserve_qubits(self, previous_size: jax.Array, size: ArrayLike, invalid: jax.Array) -> tuple:
        """Reserve global ids for ``size`` new qubits and flag an overflow.

        Parameters
        ----------
        previous_size : jax.Array
            The first free global id.
        size : ArrayLike
            The number of new qubits.
        invalid : jax.Array
            Whether the computation has already overflowed.

        Returns
        -------
        idx_start : jax.Array
            The global id of the first new qubit.
        idx_end : jax.Array
            The first free global id after the new qubits.
        invalid : jax.Array
            Whether the computation has overflowed ``max_qubits``.

        """
        idx_start = previous_size
        idx_end = idx_start + size

        invalid = jnp.logical_or(invalid, idx_end > jnp.int64(self.max_qubits))
        jax.lax.cond(invalid, _warn_overflow, lambda *_: None, idx_end, self.max_qubits)

        return idx_start, idx_end, invalid

    def handle_create_qubits(self, invalues, eqn, context_dic):
        """Assign global ids to the new qubits, flagging an overflow of ``max_qubits``."""
        size, metric_data = invalues
        depth_array, global_depth, previous_size, invalid, meas_count = metric_data

        idx_start, previous_size, invalid = self._reserve_qubits(previous_size, size, invalid)

        table_size = size if not is_abstract(size) else self.max_qubits
        qubit_table_handle = (_create_lookup_table(idx_start, table_size), size)
        metric_data = (depth_array, global_depth, previous_size, invalid, meas_count)

        # Associate the following in context_dic:
        # QubitArray -> qubit_table_handle (qubit_ids_table, size)
        # QuantumState -> metric_data
        return qubit_table_handle, metric_data

    def handle_get_qubit(self, invalues, eqn, context_dic):
        """Look up the global id of the qubit, counting negative indices from the end."""
        qubit_table_handle, qubit_index = invalues

        qubit_ids_table, size = qubit_table_handle

        # Negative indices count from the end of the qubit array. This has to be
        # done here: for a dynamic size the lookup table has max_qubits entries,
        # so indexing it with a negative index would pick the wrong qubit.
        qubit_index = qubit_index + (qubit_index < 0) * size

        # Associate the following in context_dic:
        # Qubit -> qubit_id (integer tracer/array scalar)
        return qubit_ids_table[qubit_index]

    def handle_get_size(self, invalues, eqn, context_dic):
        """Return the size stored in the QubitArray handle."""
        (qubit_table_handle,) = invalues

        _, size = qubit_table_handle

        # Associate the following in context_dic:
        # size -> size_value (integer tracer/scalar)
        return size

    def handle_quantum_gate(self, invalues, eqn, context_dic):
        """Advance the depth of the qubits the gate acts on, gate by gate of its definition."""
        depth_array, current_depth, previous_size, invalid, meas_count = invalues[-1]

        op = eqn.params["gate"]
        # Qubits have been already converted to their integer (possibly traced)
        # indices by the `jasp.get_qubit` primitive handler
        qubits = list(invalues[: op.num_qubits])

        for qubit_ids in _gate_layers(op, qubits):
            depth_array, current_depth = _apply_duration_on_qubits(
                depth_array=depth_array,
                current_depth=current_depth,
                qubit_ids=qubit_ids,
                duration=jnp.int64(1),
            )

        # Associate the following in context_dic:
        # QuantumState -> metric_data
        return depth_array, current_depth, previous_size, invalid, meas_count

    def handle_measure(self, invalues, eqn, context_dic):
        """Sample the measurement outcome and count the measured qubits."""
        target, metric_data = invalues
        *depth_data, meas_count = metric_data

        num_measured = target[1] if isinstance(eqn.invars[0].aval, AbstractQubitArray) else 1
        meas_res = self._sample_measurement(eqn, num_measured, meas_count)

        # Associate the following in context_dic:
        # meas_result -> meas_res
        # QuantumState -> metric_data
        return meas_res, (*depth_data, meas_count + num_measured)

    def handle_fuse(self, invalues, eqn, context_dic):
        """Concatenate the lookup tables of the operands, each a QubitArray or a single qubit."""
        handle_a, handle_b = (_as_qubit_array_handle(value) for value in invalues)

        # Associate the following in context_dic:
        # QubitArray -> (qubit_array_out, size_out)
        return _fuse_qubit_tables(handle_a, handle_b, self.max_qubits)

    def handle_slice(self, invalues, eqn, context_dic):
        """Select the entries of the lookup table in the slice, following Python semantics."""
        (qubit_array, size), start, stop = invalues

        start, stop = normalize_slice_bounds(size, start, stop)
        size_out = stop - start

        table_size = size_out if not is_abstract(size_out) else self.max_qubits

        array_idx_mapping = _create_lookup_table(start, table_size)
        qubit_array_out = qubit_array[array_idx_mapping]

        # Associate the following in context_dic:
        # QubitArray -> (qubit_array_out, size_out)
        return qubit_array_out, size_out

    def handle_reset(self, invalues, eqn, context_dic):
        """Leave the depth unchanged: resets are currently ignored."""
        _, metric_data = invalues

        # Associate the following in context_dic:
        # QuantumState -> metric_data
        return metric_data

    def handle_delete_qubits(self, invalues, eqn, context_dic):
        """Leave the depth unchanged: deleted qubits keep their ids and still count toward ``max_qubits``."""
        _, metric_data = invalues

        # Associate the following in context_dic:
        # QuantumState -> metric_data
        return metric_data


def extract_depth(res: tuple, jaspr: Jaspr, _) -> int:
    """Extract depth from the profiling result.

    Parameters
    ----------
    res : tuple
        The output of the profiler: the Jaspr's return values followed by the metric data.
    jaspr : Jaspr
        The profiled Jaspr.
    _ : None
        The auxiliary data of the profiler, unused by this metric.

    Returns
    -------
    int
        The depth of the circuit.

    Raises
    ------
    ValueError
        If the computation needed more than ``max_qubits`` qubits.

    """
    metric = res[-1] if len(jaspr.outvars) > 1 else res
    _, depth, _, overflowed, _ = metric

    if overflowed:
        raise ValueError("The depth metric computation overflowed the maximum number of qubits supported.")

    return int(depth)


# LRU cache controlled by QRISP_COMPILATION_CACHE_SIZE env var
@qrisp_lru_compilation_cache
def get_depth_profiler(
    jaspr: Jaspr, meas_behavior: Callable, max_qubits: int = 1024, callback_threshold: int | None = None
) -> tuple[Callable, None]:
    """Build a depth profiling computer for a given Jaspr.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr expression to profile.

    meas_behavior : Callable
        The measurement behavior function.

    max_qubits : int, optional
        The maximum number of qubits supported for depth computation. Default is 1024.

    callback_threshold : int | None, optional
        Minimum value of ``call_count * inlined_eqn_count`` required to
        trigger ``jax.pure_callback`` wrapping.  ``None`` (default)
        disables callbacks entirely (fastest execution).  ``0`` wraps
        every reused sub-jaxpr (fastest compilation).

    Returns
    -------
    tuple[Callable, None]
        A depth profiler function and None as auxiliary data.

    """
    depth_metric = DepthMetric(meas_behavior, max_qubits)

    return build_metric_profiler(jaspr, depth_metric, callback_threshold), None


def simulate_depth(jaspr: Jaspr, *_, **__) -> int:
    """Simulate depth metric via actual simulation.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr to simulate.
    *_ : Any
        The arguments of the Jaspr (unused).
    **__ : Any
        Keyword arguments of the simulation (unused).

    Raises
    ------
    NotImplementedError
        Always, since simulation-based depth computation is not implemented yet.

    """
    raise NotImplementedError("Depth metric via simulation is not implemented yet.")
