"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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
from typing import Callable, Tuple, Dict

import jax
import jax.numpy as jnp
from jax.random import key

from qrisp.jasp.interpreter_tools import (
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


class CountOpsMetric(BaseMetric):
    """
    A metric implementation that counts quantum operations in a Jaspr.

    Parameters
    ----------
    profiling_dic : Dict[str, int]
        A dictionary mapping quantum operation names to their profiling indices.

    meas_behavior : Callable
        The measurement behavior function.

    """

    def __init__(self, meas_behavior: Callable, profiling_dic: Dict[str, int]) -> None:
        """Initialize the CountOpsMetric."""

        super().__init__(meas_behavior=meas_behavior)

        self._profiling_dic: Dict[str, int] = profiling_dic

    @property
    def profiling_dic(self) -> Dict[str, int]:
        """Return the profiling dictionary."""
        return self._profiling_dic

    @classmethod
    def from_cache_key(cls, cache_key):
        zipped_profiling_dic, meas_behavior = cache_key
        profiling_dic = dict(zipped_profiling_dic)
        return cls(meas_behavior, profiling_dic)

    def cache_key(self) -> Tuple:
        return (tuple(self.profiling_dic.items()), self.meas_behavior)

    def handle_measure(self, invalues, eqn, context_dic):

        counting_index = self.profiling_dic["measure"]
        counting_array = list(invalues[-1][0])
        incrementation_constants = invalues[-1][1]

        # If I understand correctly the logic here,
        # `meas_number` is the current count of measurements performed so far.
        # It’s also the next available measurement id (offset) for key generation.
        meas_number = counting_array[counting_index]

        if isinstance(eqn.invars[0].aval, AbstractQubitArray):

            def body_fun(i, acc):
                return self._measurement_body_fun(meas_number, i, acc)

            meas_res = jax.lax.fori_loop(0, invalues[0], body_fun, jnp.int64(0))
            counting_array[counting_index] += invalues[0]

        else:  # measuring a single qubit

            meas_res = self.meas_behavior(key(meas_number))
            self._validate_measurement_result(meas_res)
            counting_array[counting_index] += incrementation_constants[0]

        return (meas_res, (counting_array, incrementation_constants))

    # Since we don't need to track to which qubits a certain operation
    # is applied, we can implement a really simple behavior for most
    # Qubit/QubitArray handling methods.
    # We represent qubit arrays simply with integers (indicating their)
    # size.

    def handle_create_qubits(self, invalues, eqn, context_dic):

        # Since we represent QubitArrays via integers, it is sufficient
        # to simply return the input as the output for this primitive.
        return invalues

    def handle_get_qubit(self, invalues, eqn, context_dic):

        # Trivial behavior since we don't need qubit address information
        return invalues

    def handle_get_size(self, invalues, eqn, context_dic):

        # The QubitArray size is represented via an integer.
        return invalues[0]

    def handle_fuse(self, invalues, eqn, context_dic):

        # The size of the fused qubit array is the size of the two added.
        return invalues[0] + invalues[1]

    def handle_slice(self, invalues, eqn, context_dic):

        # For the slice operation, we need to make sure, we don't go out
        # of bounds.
        start = jnp.max(jnp.array([invalues[1], 0]))
        stop = jnp.min(jnp.array([invalues[2], invalues[0]]))
        return stop - start

    def handle_quantum_gate(self, invalues, eqn, context_dic):

        # In the case of a quantum gate, we determine the array index
        # to be increment via dictionary look-up and perform the increment
        # via the Jax-given .at method.

        counting_array = list(invalues[-1][0])
        incrementation_constants = invalues[-1][1]

        op = eqn.params["gate"]

        if op.definition:
            op_counts = op.definition.transpile().count_ops()
        else:
            op_counts = {op.name: 1}

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
            while count:
                incrementor = min(count, len(incrementation_constants))
                count -= incrementor
                counting_array[counting_index] += incrementation_constants[
                    incrementor - 1
                ]

        return (counting_array, incrementation_constants)

    def handle_delete_qubits(self, invalues, eqn, context_dic):

        # Trivial behavior: return the last argument (the counting array).
        return invalues[-1]

    def handle_reset(self, invalues, eqn, context_dic):

        # Trivial behavior: return the last argument (the counting array).
        return invalues[-1]

    def handle_parity(self, invalues, eqn, context_dic):
        """Handle the `jasp.parity` primitive"""

        # Parity is a classical operation on measurement results
        # Compute XOR and handle expectation
        expectation = eqn.params.get("expectation", 0)
        result = jnp.bitwise_xor(sum(invalues) % 2, expectation)

        return jnp.array(result, dtype=bool)


def extract_count_ops(res: Tuple, jaspr: Jaspr, profiling_dic: dict) -> dict:
    """Extract depth from the profiling result."""

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


@lru_cache(int(1e5))
def get_count_ops_profiler(
    jaspr: Jaspr, meas_behavior: Callable
) -> Tuple[Callable, dict]:
    """
    Build a count operations profiling computer for a given Jaspr.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr expression to profile.

    meas_behavior : Callable
        The measurement behavior function.

    Returns
    -------
    Tuple[Callable, dict]
        A count operations profiler function and the profiling dictionary.

    """

    quantum_operations = get_quantum_operations(jaspr)
    profiling_dic = {quantum_operations[i]: i for i in range(len(quantum_operations))}

    if "measure" not in profiling_dic:
        profiling_dic["measure"] = -1

    count_ops_metric = CountOpsMetric(meas_behavior, profiling_dic)
    profiling_eqn_evaluator = make_profiling_eqn_evaluator(count_ops_metric)
    jitted_evaluator = jax.jit(eval_jaxpr(jaspr, eqn_evaluator=profiling_eqn_evaluator))

    def count_ops_profiler(*args):

        # Filter out types that are known to be static (https://github.com/eclipse-qrisp/Qrisp/issues/258)
        # Import here to avoid circular import issues
        from qrisp.operators import FermionicOperator, QubitOperator

        STATIC_TYPES = (str, QubitOperator, FermionicOperator, types.FunctionType)

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
        initial_metric_value = ([0] * len(profiling_dic), list(range(1, 6)))

        filtered_args = [
            x for x in args + (initial_metric_value,) if type(x) not in STATIC_TYPES
        ]

        return jitted_evaluator(*filtered_args)

    return count_ops_profiler, profiling_dic
