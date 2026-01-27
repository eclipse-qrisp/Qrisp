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
}


# TODO: `count_ops` and `depth` should be moved to their own (already existing) files.
# So far we keep them here to avoid circular imports.


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


def depth(meas_behavior) -> Callable:
    """
    Decorator to determine the depth of large scale quantum computations.
    This decorator compiles the given Jasp-compatible function into a classical
    function computing the depth required. The decorated function will return
    an integer indicating the depth of the quantum computation.

    Parameters
    ----------
    meas_behavior : str or callable
        A string or callable indicating the behavior of the resource computation
        when measurements are performed. Available strings are ``"0"`` and ``"1"``.

    Returns
    -------
    depth decorator
        A decorator, producing a function to computed the depth required.

    """

    def depth_decorator(function):

        def depth_counter(*args):

            from qrisp.jasp import make_jaspr

            if not hasattr(function, "jaspr_dict"):
                function.jaspr_dict = {}

            args = list(args)

            signature = tuple([type(arg) for arg in args])
            if not signature in function.jaspr_dict:
                function.jaspr_dict[signature] = make_jaspr(function)(*args)

            return function.jaspr_dict[signature].depth(
                *args, meas_behavior=meas_behavior
            )

        return depth_counter

    return depth_decorator


def profile_jaspr(
    jaspr: Jaspr, mode: str, meas_behavior: str | Callable = "0"
) -> Callable:
    """
    Profile a Jaspr according to a given metric mode.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr to be profiled.
    mode : str
        The profiling mode to be used. Currently supported modes are "depth".
    meas_behavior : str or callable, optional
        The measurement behavior to be used during profiling. Default is "0".


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
    profiler, aux = metric_spec.build_profiler(jaspr, meas_behavior_callable)

    @wraps(profiler)
    def profiler_wrapper(*args):
        args = tree_flatten(args)[0]
        res = profiler(*args)
        return metric_spec.extract_metric(res, jaspr, aux)

    return profiler_wrapper
