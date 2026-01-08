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

"""
This file implements the tools to perform quantum resource estimation using Jasp
infrastructure. The idea here is to transform the quantum instructions within a
given Jaspr into "counting instructions". That means instead of performing some
quantum gate, we increment an index in an array, which keeps track of how many
instructions of each type have been performed.

To do this, we implement the 

qrisp.jasp.interpreter_tools.interpreters.profiling_interpreter.py

Which handles the transformation logic of the Jaspr.
This file implements the interfaces to evaluating the transformed Jaspr.

"""

from functools import lru_cache
import types

import jax
from jax.extend.core import ClosedJaxpr
from jax.tree_util import tree_flatten

from qrisp.jasp.interpreter_tools import make_profiling_eqn_evaluator, eval_jaxpr
from qrisp.jasp.evaluation_tools.jaspification import simulate_jaspr


def count_ops(meas_behavior):
    """
    Decorator to determine resources of large scale quantum computations.
    This decorator compiles the given Jasp-compatible function into a classical
    function computing the amount of each gates required. The decorated function
    will return a dictionary containing the operation counts.

    For many algorithms including classical feedback, the result of the
    measurements can heavily influence the required resources. To reflect this,
    users can specify the behavior of measurements during the computation of
    resources. The following strategies are available:

    * ``"0"`` - computes the resource as if measurements always return 0
    * ``"1"`` - computes the resource as if measurements always return 1
    * *callable* - allows the user to specify a random number generator (see examples)

    For more details on how the *callable* option can be used, consult the
    examples section.

    Finally it is also possible to call the Qrisp simulator to determine
    measurement behavior by providing ``sim``. This is of course much less
    scalable but in particular for algorithms involving repeat-until-success
    components, a necessary evil.

    Note that the ``sim`` option might return non-deterministic results, while
    the other methods do.

    .. warning::

        It is currently not possible to estimate programs, which include a
        :ref:`kernelized <quantum_kernel>` function.

    Parameters
    ----------
    meas_behavior : str or callable
        A string or callable indicating the behavior of the ressource computation
        when measurements are performed. Available are


    Returns
    -------
    resource_estimation decorator
        A decorator, producing a function to computed the required resources.

    Examples
    --------

    We compute the resources required to perform a large scale integer multiplication.

    ::

        from qrisp import count_ops, QuantumFloat, measure

        @count_ops(meas_behavior = "0")
        def main(i):

            a = QuantumFloat(i)
            b = QuantumFloat(i)

            c = a*b

            return measure(c)

        print(main(5))
        # {'cx': 506, 'x': 22, 'h': 135, 'measure': 55, '2cx': 2, 's': 45, 't': 90, 't_dg': 90}
        print(main(5000))
        # {'cx': 462552491, 'x': 20002, 'h': 112522500, 'measure': 37517500, '2cx': 2, 's': 37507500, 't': 75015000, 't_dg': 75015000}

    Note that even though the second computation contains more than 800 million gates,
    determining the resources takes less than 200ms, highlighting the scalability
    features of the Jasp infrastructure.

    **Modifying the measurement behavior via a random number generator**

    To specify the behavior, we specify an RNG function (for more details on
    what that means please check the `Jax documentation <https://docs.jax.dev/en/latest/jax.random.html>`_.
    This RNG takes as input a "key" and returns a boolean value.
    In this case, the return value will be uniformly distributed among True and False.

    ::


        from jax import random
        import jax.numpy as jnp
        from qrisp import QuantumFloat, measure, control, count_ops, x

        # Returns a uniformly distributed boolean
        def meas_behavior(key):
            return jnp.bool(random.randint(key, (1,), 0,1)[0])

        @count_ops(meas_behavior = meas_behavior)
        def main(i):

            qv = QuantumFloat(2)

            meas_res = measure(qv)

            with control(meas_res == i):
                x(qv)

            return measure(qv)

    This script executes two measurements and based on the measurement outcome
    executes two X gates. We can now execute this resource computation with
    different values of ``i`` to see, which measurements return ``True`` with
    our given random-number generator (recall that this way of specifying the
    measurement behavior is fully deterministic).

    ::

        print(main(0))
        # Yields: {'measure': 4, 'x': 2}
        print(main(1))
        # Yields: {'measure': 4}
        print(main(2))
        # Yields: {'measure': 4}
        print(main(3))
        # Yields: {'measure': 4}

    From this we conclude that our RNG returned 0 for both of the initial
    measurements.

    For some algorithms (such as :ref:`RUS`) sampling the measurement result
    from a simple distribution won't cut it because the required ressource can
    be heavily influenced by measurement outcomes. For this matter it is also
    possible to perform a full simulation. Note that this simulation is no
    longer deterministic.

    ::

        @count_ops(meas_behavior = "sim")
        def main(i):

            qv = QuantumFloat(2)

            meas_res = measure(qv)

            with control(meas_res == i):
                x(qv)

            return measure(qv)

        print(main(0))
        {'measure': 4, 'x': 2}
        print(main(1))
        {'measure': 4}

    """

    def count_ops_decorator(function):

        def ops_counter(*args):

            from qrisp.jasp import make_jaspr

            if not hasattr(function, "jaspr_dict"):
                function.jaspr_dict = {}

            args = list(args)

            signature = tuple([type(arg) for arg in args])
            if not signature in function.jaspr_dict:
                function.jaspr_dict[signature] = make_jaspr(function)(*args)

            return function.jaspr_dict[signature].count_ops(
                *args, meas_behavior=meas_behavior
            )

        return ops_counter

    return count_ops_decorator


def always_zero(key):
    return False


def always_one(key):
    return True


# This function is the central interface for performing resource estimation.
# It takes a Jaspr and returns a function, returning a dictionary (with the counted
# operations).
def profile_jaspr(jaspr, meas_behavior="0"):

    if isinstance(meas_behavior, str):
        if meas_behavior == "0":
            meas_behavior = always_zero
        elif meas_behavior == "1":
            meas_behavior = always_one
        elif not meas_behavior == "sim":
            raise Exception(
                f"Don't know how to compute required resources via method {meas_behavior}"
            )

    if callable(meas_behavior):

        def profiler(*args):

            # The profiling array computer is a function that computes the array
            # countaining the gate counts.
            # The profiling dic is a dictionary of type {str : int}, which indicates
            # which operation has been computed at which index of the array.
            profiling_array_computer, profiling_dic = get_profiling_array_computer(
                jaspr, meas_behavior
            )

            args = tree_flatten(args)[0]

            # Compute the profiling array
            if len(jaspr.outvars) > 1:
                profiling_array = profiling_array_computer(*args)[-1][0]
            else:
                profiling_array = profiling_array_computer(*args)[0]

            # Transform to a dictionary containing gate counts
            res_dic = {}
            for k in profiling_dic.keys():
                if int(profiling_array[profiling_dic[k]]):
                    res_dic[k] = int(profiling_array[profiling_dic[k]])

            return res_dic

    else:

        def profiler(*args):
            return simulate_jaspr(jaspr, *args, return_gate_counts=True)

    return profiler


# This function takes a Jaspr and returns a function computing the "counting array"
@lru_cache(int(1e5))
def get_profiling_array_computer(jaspr, meas_behavior):

    # Find the occuring quantum operations and store their names in a dictionary,
    # indicating, which tracer counts what operation
    quantum_operations = get_quantum_operations(jaspr)
    profiling_dic = {quantum_operations[i]: i for i in range(len(quantum_operations))}

    if "measure" not in profiling_dic:
        profiling_dic["measure"] = -1

    profiling_eqn_evaluator = make_profiling_eqn_evaluator(profiling_dic, meas_behavior)

    evaluator = jax.jit(eval_jaxpr(jaspr, eqn_evaluator=profiling_eqn_evaluator))

    # This function calls the profiling interpeter to evaluate the gate counts
    def profiling_array_computer(*args):

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
        final_arg = ([0] * len(profiling_dic), list(range(1, 6)))

        # Filter out types that are known to be static (https://github.com/eclipse-qrisp/Qrisp/issues/258)
        filtered_args = []
        from qrisp.operators import QubitOperator, FermionicOperator

        for x in list(args) + [final_arg]:
            if type(x) not in [
                str,
                QubitOperator,
                FermionicOperator,
                types.FunctionType,
            ]:
                filtered_args.append(x)

        res = evaluator(*filtered_args)

        return res

    return profiling_array_computer, profiling_dic


# This functions determines the set of primitives that appear in a given Jaxpr
def get_quantum_operations(jaxpr):

    quantum_operations = set()

    for eqn in jaxpr.eqns:
        # Add current primitive
        if eqn.primitive.name == "jasp.quantum_gate":

            if eqn.params["gate"].definition:
                for op_name in (
                    eqn.params["gate"].definition.transpile().count_ops().keys()
                ):
                    quantum_operations.add(op_name)
            else:
                quantum_operations.add(eqn.params["gate"].name)

        if eqn.primitive.name == "cond":
            quantum_operations.update(
                get_quantum_operations(eqn.params["branches"][0].jaxpr)
            )
            quantum_operations.update(
                get_quantum_operations(eqn.params["branches"][1].jaxpr)
            )
            continue

        # Handle call primitives (like cond/pjit)
        for param in eqn.params.values():
            if isinstance(param, ClosedJaxpr):
                quantum_operations.update(get_quantum_operations(param.jaxpr))

    return list(quantum_operations)
