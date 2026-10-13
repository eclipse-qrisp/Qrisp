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

"""Defines CountOpsMetric, a profiling metric that counts quantum gate operations in a Jaspr."""

from collections.abc import Callable

from qrisp._cache_config import qrisp_lru_compilation_cache
from qrisp.jasp.interpreter_tools.interpreters.profiling_interpreter import (
    SizeBasedMetric,
    build_metric_profiler,
)
from qrisp.jasp.interpreter_tools.interpreters.utilities import (
    get_op_counts,
    get_quantum_operations,
)
from qrisp.jasp.jasp_expression import Jaspr
from qrisp.jasp.primitives import (
    AbstractQubitArray,
)


class CountOpsMetric(SizeBasedMetric):
    """A metric implementation that counts quantum operations in a Jaspr.

    Parameters
    ----------
    meas_behavior : Callable
        The measurement behavior function.

    profiling_dic : dict[str, int]
        A dictionary mapping quantum operation names to their profiling indices.

    """

    def __init__(self, meas_behavior: Callable, profiling_dic: dict[str, int]) -> None:
        """Initialize the CountOpsMetric."""
        super().__init__(meas_behavior=meas_behavior)

        self._profiling_dic: dict[str, int] = profiling_dic

    @property
    def profiling_dic(self) -> dict[str, int]:
        """Return the mapping from quantum operation names to their counter indices."""
        return self._profiling_dic

    @classmethod
    def from_cache_key(cls, cache_key) -> "CountOpsMetric":
        """Rebuild the metric from its profiling dictionary and measurement behavior."""
        zipped_profiling_dic, meas_behavior = cache_key
        profiling_dic = dict(zipped_profiling_dic)
        return cls(meas_behavior, profiling_dic)

    def cache_key(self) -> tuple[tuple, Callable]:
        """Return the profiling dictionary and the measurement behavior as a hashable key."""
        return (tuple(self.profiling_dic.items()), self.meas_behavior)

    def initial_metric(self) -> tuple[list[int], list[int]]:
        """Return one zero counter per operation, and the incrementation constants 1 to 5."""
        # The XLA compiler showed some scalability problems in compile time.
        # Through a process involving a lot of blood and sweat
        # we reverse engineered what to do to improve these problems
        # 1. represent the integers that count the gates as a list
        # of integers (instead of an array)
        # 2. Avoid telling the compiler that it is constants that
        # are being added. To do this, we supply a list of the first
        # few integers as arguments, which will be used to do the
        # incrementation (i.e. CZ_count += 1). It therefore doesn't
        # look like a constant is being added but a variable
        initial_metric_value = ([0] * len(self.profiling_dic), list(range(1, 6)))

        return initial_metric_value

    ##############################################################
    ### Quantum primitive handlers
    ##############################################################

    def handle_measure(self, invalues, eqn, context_dic):
        """Sample the measurement outcome and count the measured qubits."""
        target, (counting_array, incrementation_constants) = invalues
        counting_array = list(counting_array)

        # The measurement count so far numbers the measured qubits (see _sample_measurement).
        counting_index = self.profiling_dic["measure"]
        meas_res = self._sample_measurement(eqn, target, counting_array[counting_index])

        if isinstance(eqn.invars[0].aval, AbstractQubitArray):
            counting_array[counting_index] += target
        else:
            counting_array[counting_index] += incrementation_constants[0]

        return (meas_res, (counting_array, incrementation_constants))

    # QubitArrays are represented by their size (see SizeBasedMetric),
    # so creating one just forwards the size.

    def handle_create_qubits(self, invalues, eqn, context_dic):
        """Represent the new QubitArray by its size, leaving the counters unchanged."""
        # Associate the following in context_dic:
        # QubitArray -> size
        # QuantumState -> metric_data
        return invalues

    def handle_quantum_gate(self, invalues, eqn, context_dic):
        """Add the operations of the gate, after decomposing it into counted operations."""
        # For each operation, we look up the index of its counter in the
        # profiling dictionary and increment that entry of the counter list.

        counting_array = list(invalues[-1][0])
        incrementation_constants = invalues[-1][1]

        op_counts = get_op_counts(eqn.params["gate"])

        for op_name, count in op_counts.items():
            counting_index = self.profiling_dic[op_name]

            # It seems like at this point we can just
            # naively increment the Jax tracer but
            # unfortunately the XLA compiler really
            # doesn't like constants. (Compile time blows up)
            # We therefore jump through a lot of hoops
            # to make it look like we are adding variables.
            # This is done by supplying a list of tracers
            # that will contain the numbers from 1 to n.
            # Here is the next problem: If n is very large
            # this also slows down the compilation because
            # each function call has to have n+ arguments.
            # We therefore perform the following loop:
            remaining = count
            while remaining:
                incrementor = min(remaining, len(incrementation_constants))
                remaining -= incrementor
                counting_array[counting_index] += incrementation_constants[incrementor - 1]

        return (counting_array, incrementation_constants)

    def handle_delete_qubits(self, invalues, eqn, context_dic):
        """Leave the counters unchanged: deleting qubits applies no operation."""
        # Trivial behavior: return the last argument (the counting array).
        return invalues[-1]

    def handle_reset(self, invalues, eqn, context_dic):
        """Leave the counters unchanged: resets are not counted as operations."""
        # Trivial behavior: return the last argument (the counting array).
        return invalues[-1]


def extract_count_ops(res: tuple, jaspr: Jaspr, profiling_dic: dict) -> dict:
    """Extract the operation counts from the profiling result.

    Parameters
    ----------
    res : tuple
        The output of the profiler: the Jaspr's return values followed by the metric data.
    jaspr : Jaspr
        The profiled Jaspr.
    profiling_dic : dict
        The mapping from operation names to counter indices.

    Returns
    -------
    dict
        The number of each operation that occurs at least once.

    """
    if len(jaspr.outvars) > 1:
        profiling_array = res[-1][0]
    else:
        profiling_array = res[0]

    # Transform to a dictionary containing gate counts
    res_dic = {}
    for k in profiling_dic.keys():
        if int(profiling_array[profiling_dic[k]]):
            res_dic[k] = int(profiling_array[profiling_dic[k]])

    return res_dic


# LRU cache controlled by QRISP_COMPILATION_CACHE_SIZE env var
@qrisp_lru_compilation_cache
def get_count_ops_profiler(
    jaspr: Jaspr, meas_behavior: Callable, callback_threshold: int | None = None
) -> tuple[Callable, dict]:
    """Build a count operations profiling computer for a given Jaspr.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr expression to profile.

    meas_behavior : Callable
        The measurement behavior function.

    callback_threshold : int | None, optional
        Minimum value of ``call_count * inlined_eqn_count`` required to
        trigger ``jax.pure_callback`` wrapping.  ``None`` (default)
        disables callbacks entirely (fastest execution).  ``0`` wraps
        every reused sub-jaxpr (fastest compilation).

    Returns
    -------
    tuple[Callable, dict]
        A count operations profiler function and the profiling dictionary.

    """
    quantum_operations = get_quantum_operations(jaspr)
    profiling_dic = {quantum_operations[i]: i for i in range(len(quantum_operations))}

    if "measure" not in profiling_dic:
        profiling_dic["measure"] = -1

    count_ops_metric = CountOpsMetric(meas_behavior, profiling_dic)

    return build_metric_profiler(jaspr, count_ops_metric, callback_threshold), profiling_dic
