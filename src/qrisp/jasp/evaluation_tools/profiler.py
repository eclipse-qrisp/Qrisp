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

from functools import wraps
from typing import Any, Callable, NamedTuple, Tuple

from jax.tree_util import tree_flatten

from qrisp.jasp.evaluation_tools.jaspification import simulate_jaspr
from qrisp.jasp.interpreter_tools.interpreters.count_ops_metric import (
    extract_count_ops,
    get_count_ops_profiler,
)
from qrisp.jasp.interpreter_tools.interpreters.depth_metric import (
    extract_depth,
    get_depth_profiler,
    simulate_depth,
)
from qrisp.jasp.interpreter_tools.interpreters.num_qubits_metric import (
    extract_num_qubits,
    get_num_qubits_profiler,
    simulate_num_qubits,
)
from qrisp.jasp.interpreter_tools.interpreters.utilities import (
    always_one,
    always_zero,
    simulation,
)
from qrisp.jasp.jasp_expression import Jaspr


class MetricSpec(NamedTuple):
    """Specification of a metric to be computed via profiling."""

    build_profiler: Callable[[Jaspr, Callable], Tuple[Callable, Any]]
    extract_metric: Callable[[Tuple, Jaspr, Any], Any]
    simulate_fallback: Callable[[Jaspr, Any], Any]


METRIC_DISPATCH = {
    "count_ops": MetricSpec(
        build_profiler=get_count_ops_profiler,
        extract_metric=extract_count_ops,
        simulate_fallback=simulate_jaspr,
    ),
    "depth": MetricSpec(
        build_profiler=get_depth_profiler,
        extract_metric=extract_depth,
        simulate_fallback=simulate_depth,
    ),
    "num_qubits": MetricSpec(
        build_profiler=get_num_qubits_profiler,
        extract_metric=extract_num_qubits,
        simulate_fallback=simulate_num_qubits,
    ),
}


def _normalize_meas_behavior(meas_behavior) -> Callable:
    """Normalize the measurement behavior into a callable."""

    if isinstance(meas_behavior, str):
        if meas_behavior == "0":
            return always_zero
        if meas_behavior == "1":
            return always_one
        if meas_behavior == "sim":
            return simulation
        raise ValueError(
            f"Don't know how to compute required resources via method {meas_behavior}"
        )

    if callable(meas_behavior):
        return meas_behavior

    raise TypeError("meas_behavior must be a str or callable")


# TODO: Move each metric implementation into its dedicated module (already present).
# Keeping them here for now to avoid circular imports.


def count_ops(meas_behavior: str | Callable) -> Callable:
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
    measurement behavior by providing ``"sim"``. This is of course much less
    scalable but in particular for algorithms involving repeat-until-success
    components, a necessary evil.

    Note that the ``"sim"`` option might return non-deterministic results, while
    the other methods do.

    .. warning::

        It is currently not possible to estimate programs, which include a
        :ref:`kernelized <quantum_kernel>` function.

    Parameters
    ----------
    meas_behavior : str or callable
        A string or callable indicating the behavior of the resource computation
        when measurements are performed. Available strings are ``"0"``, ``"1"``, and ``"sim"``.


    Returns
    -------
    resource_estimation decorator : Callable
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
        # {'s': 45, 'x': 22, 't_dg': 98, 'cx': 510, 't': 96, 'h': 139, 'measure': 55}
        print(main(5000))
        # {'t': 751506, 'h': 1127254, 'x': 2002, 's': 375750, 't_dg': 751508, 'cx': 4629255, 'measure': 752500}

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

            signature = tuple(type(arg) for arg in args)
            shape_signature = tuple(
                arg.shape for arg in tree_flatten(args)[0] if hasattr(arg, "shape")
            )
            hash_key = (signature, shape_signature, hash(meas_behavior))

            if hash_key not in function.jaspr_dict:
                function.jaspr_dict[hash_key] = make_jaspr(function)(*args)

            return function.jaspr_dict[hash_key].count_ops(
                *args, meas_behavior=meas_behavior
            )

        return ops_counter

    return count_ops_decorator


def depth(meas_behavior: str | Callable, max_qubits: int = 1024) -> Callable:
    """
    Decorator to determine the depth of large scale quantum computations.

    This decorator compiles the given Jasp-compatible function into a classical
    function computing the circuit depth required. The decorated function returns
    an integer indicating the depth of the quantum computation.

    The depth is computed by tracking, for each qubit, the time at which it
    becomes available again after an operation. Multi-qubit gates increase the
    depth of all qubits they act on to the same value.

    Parameters
    ----------
    meas_behavior : str or callable
        A string or callable indicating the behavior of the resource computation
        when measurements are performed. Available strings are ``"0"`` and ``"1"``.
        A callable must take a JAX PRNG key as input and return a boolean.

    max_qubits : int, optional
        The maximum number of qubits supported for depth computation.
        Default is 1024.

    Returns
    -------
    depth decorator : Callable
        A decorator producing a function that computes the depth required.

    Examples
    --------

    Let's consider a simple circuit:

    ::

        from qrisp import *

        @depth(meas_behavior="0")
        def circuit(n):
            qv = QuantumFloat(n)
            h(qv[0])
            h(qv[1])
            cx(qv[0], qv[1])
            h(qv[0])

        print(circuit(2))  # Output: 3

    The first two Hadamards run in parallel (depth 1), the CNOT
    increases depth to 2, and the final Hadamard gives depth 3.

    Now, consider a circuit with measurement and classical control:

    ::

        @depth(meas_behavior="0")
        def circuit(n):
            qv = QuantumFloat(n)
            m = measure(qv[0])

            with control(m == 0):
                h(qv[0])
                x(qv[1])
                h(qv[0])

            with control(m == 1):
                cx(qv[0], qv[1])
                h(qv[0])
                x(qv[0])

        print(circuit(2))  # Output: 2

    The same circuit with ``meas_behavior="1"`` yields a depth of 3,
    because a different branch of the computation is taken.

    **Macro-gates and gate definitions**

    If a gate has a ``definition`` (for example a Toffoli gate implemented
    as a sequence of simpler gates), the `transpile` method is applied to
    the definition to determine the depth of the macro-gate.

    .. note::

        Computing depth requires tracking qubit dependencies. As a result,
        compilation time for the depth metric can be noticeably slower for large circuits
        compared to ``count_ops``. This will be improved in future versions.
        However, the scalability offered by Jasp after the initial compilation
        is not affected.

    .. note::

        The ``max_qubits`` parameter sets an upper limit on the number of qubits
        that can be handled for depth computation. This is necessary as JAX
        requires static shapes for JIT compilation. The default value of 1024
        can be adjusted based on the expected number of qubits in the circuits
        to be analyzed.

    .. warning::

        It is currently not possible to estimate programs, which include a
        :ref:`kernelized <quantum_kernel>` function.

    """

    def depth_decorator(function):

        def depth_counter(*args):
            from qrisp.jasp import make_jaspr

            if not hasattr(function, "jaspr_dict"):
                function.jaspr_dict = {}

            signature = tuple(type(arg) for arg in args)
            shape_signature = tuple(
                arg.shape for arg in tree_flatten(args)[0] if hasattr(arg, "shape")
            )
            hash_key = (signature, shape_signature, hash(meas_behavior))

            if hash_key not in function.jaspr_dict:
                function.jaspr_dict[hash_key] = make_jaspr(function)(*args)

            return function.jaspr_dict[hash_key].depth(
                *args, meas_behavior=meas_behavior, max_qubits=max_qubits
            )

        return depth_counter

    return depth_decorator


def num_qubits(meas_behavior: str | Callable, max_allocations: int = 1000) -> Callable:
    """
    Decorator to track qubit allocation and deallocation events during a quantum computation.

    This decorator compiles a Jasp-compatible quantum function into a classical
    function that tracks the number of qubits allocated and deallocated throughout the computation.

    The counter is:

    - increased whenever qubits are allocated (e.g., via ``QuantumVariable`` creation),
    - decreased whenever qubits are explicitly deleted (e.g., via ``qv.delete()``),

    and the decorated function returns a dictionary containing all the allocation and deallocation events.
    The keys of the dictionary are of the form ``"allocX"`` where ``X`` is a unique
    identifier for each allocation or deallocation event, and the values are the number
    of qubits allocated (positive) or deallocated (negative) at that event.
    The dictionary returned preserves the execution order of events.

    From this information, users can reconstruct several resource metrics,
    such as the maximum number of qubits allocated at any point in time,
    or the final number of qubits still allocated at the end of the computation (see examples).

    Parameters
    ----------
    meas_behavior : str or callable
        A string or callable indicating the behavior of the resource computation
        when measurements are performed. Available strings are ``"0"`` and ``"1"``.
        A callable must take a JAX PRNG key as input and return a boolean.

    max_allocations : int, optional
        The maximum number of allocation/deallocation events supported for tracking.
        Default is 1000. This is necessary as JAX requires static shapes for JIT compilation.

    Returns
    -------
    Callable
        A decorator producing a function that returns a dictionary
        with the number of qubits allocated and deallocated at each event.

    Examples
    --------

    Let's consider a simple circuit in which the number of allocated
    qubits depends on the measurement outcome:

    ::

        from qrisp import *

        @num_qubits(meas_behavior="0")
        def circuit(n1, n2, n3):
            qv = QuantumFloat(n1)
            m = measure(qv[0])

            with control(m == 0):
                qv2 = QuantumFloat(n2)
                h(qv2[0])

            with control(m == 1):
                qv3 = QuantumFloat(n3)
                h(qv3[0])

        print(circuit(2, 3, 4))  # Output: {'alloc1': 2, 'alloc2': 3}

    Here, the measurement of the first qubit determines whether we allocate 3 or 4 additional qubits.
    The output dictionary contains two allocation events, one for the initial allocation of 2 qubits,
    and one for the conditional allocation of either 3 or 4 qubits.
    If we change the measurement behavior to ``"1"``, we get a different output.

    Note that deallocation affects the final count:

    ::

        @num_qubits(meas_behavior="0")
        def circuit(n):
            qv = QuantumFloat(2 * n)
            h(qv[0])
            qv.delete()

            qv = QuantumFloat(n)
            h(qv[0])

        print(circuit(4))  # Output: {'alloc1': 8, 'alloc2': -8, 'alloc3': 4}

    Here, we first allocate 8 qubits, then deallocate them, and finally allocate 4 more qubits.

    Let's see a final example with branching and deallocation:

    ::

        from qrisp import *

        @num_qubits(meas_behavior="1")
        def workflow(num_qubits_input):

            list_of_qvs = []

            for i in range(2):
                qv = QuantumFloat(num_qubits_input)
                h(qv[i])
                list_of_qvs.append(qv)

            qv_2 = QuantumFloat(1)
            h(qv_2[0])
            m = measure(qv_2[0])

            with control(m == 1):
                qv4 = QuantumFloat(10)
                h(qv4[0])
                qv4.delete()

            qv_2.delete()

            for i in range(2):
                list_of_qvs[i].delete()

        print(workflow(8))
        # Output:
        # {'alloc1': 8, 'alloc2': 8, 'alloc3': 1, 'alloc4': 10,
        # 'alloc5': -10, 'alloc6': -1, 'alloc7': -8, 'alloc8': -8}

    We can retrieve all the information about qubit allocation and deallocation events from the output dictionary,
    which allows us to reconstruct the number of qubits allocated at any point in time.

    For example, we can compute the peak number of qubits allocated during the computation as follows:

    ::

        def peak_allocated_qubits(alloc_dict: dict) -> int:

            current = 0
            peak = 0
            for delta in alloc_dict.values():
                current += delta
                peak = max(peak, current)
            return peak

        peak_allocated_qubits(workflow(8))  # Output: 27

    Or, we can compute the number of qubits still allocated at the end of the computation as follows:

    ::

        def qubits_still_allocated(alloc_dict: dict) -> int:
            return sum(alloc_dict.values())

        qubits_still_allocated(workflow(8))  # Output: 0

    And so on.

    .. warning::

        Programs that include a :ref:`kernelized <quantum_kernel>` function
        cannot currently be analyzed.

    """

    def num_qubits_decorator(function):

        def qubits_counter(*args):

            from qrisp.jasp import make_jaspr

            if not hasattr(function, "jaspr_dict"):
                function.jaspr_dict = {}

            signature = tuple(type(arg) for arg in args)
            shape_signature = tuple(
                arg.shape for arg in tree_flatten(args)[0] if hasattr(arg, "shape")
            )
            hash_key = (signature, shape_signature, hash(meas_behavior))

            if hash_key not in function.jaspr_dict:
                function.jaspr_dict[hash_key] = make_jaspr(function)(*args)

            return function.jaspr_dict[hash_key].num_qubits(
                *args, meas_behavior=meas_behavior, max_allocations=max_allocations
            )

        return qubits_counter

    return num_qubits_decorator


def profile_jaspr(
    jaspr: Jaspr, mode: str, meas_behavior: str | Callable = "0", **kwargs: Any
) -> Callable:
    """
    Profile a Jaspr according to a given metric mode.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr to be profiled.

    mode : str
        The profiling mode to be used.
        Currently supported modes are "depth", "count_ops", and "num_qubits".

    meas_behavior : str or callable, optional
        The measurement behavior to be used during profiling. Default is "0".

    **kwargs : Any
        Additional keyword arguments to be passed to the profiler builder.
        For example, `max_qubits` for depth profiling,
        or `max_allocations` for num_qubits profiling.

    Returns
    -------
    Callable
        A function that computes the specified metric when called with the same
        arguments as the original Jaspr.

    """

    meas_behavior_callable = _normalize_meas_behavior(meas_behavior)
    metric_spec = METRIC_DISPATCH[mode]

    if (
        meas_behavior_callable.__name__ == "simulation"
        and metric_spec.simulate_fallback is not None
    ):

        @wraps(metric_spec.simulate_fallback)
        def simulation_wrapper(*args):
            return metric_spec.simulate_fallback(jaspr, *args, return_gate_counts=True)

        return simulation_wrapper

    # `profiler` is a function that computes the metric we are interested in.
    # `aux` is any auxiliary data that might be needed to reconstruct the metric
    # (for example the profiling dictionary for count_ops).
    profiler, aux = metric_spec.build_profiler(jaspr, meas_behavior_callable, **kwargs)

    @wraps(profiler)
    def profiler_wrapper(*args):
        args = tree_flatten(args)[0]
        res = profiler(*args)
        return metric_spec.extract_metric(res, jaspr, aux)

    return profiler_wrapper
