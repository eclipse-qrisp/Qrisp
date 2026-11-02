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

"""User-facing decorators for quantum resource estimation of Jasp functions."""

# Resource estimation transforms the quantum instructions of a Jaspr into
# classical "counting instructions": instead of performing a quantum gate, a
# metric updates classical data, for example by incrementing a gate counter.
# The transformation itself is implemented in
# qrisp.jasp.interpreter_tools.interpreters.profiling_interpreter, while this
# module implements the decorators (count_ops, depth, num_qubits) that
# evaluate the transformed Jaspr.

import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any, NamedTuple

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
from qrisp.jasp.interpreter_tools.interpreters.profiling_interpreter import (
    get_cached_jaspr,
)
from qrisp.jasp.interpreter_tools.interpreters.utilities import (
    always_one,
    always_zero,
    simulation,
)
from qrisp.jasp.jasp_expression import Jaspr
from qrisp.misc.exceptions import QrispDeprecationWarning


class MetricSpec(NamedTuple):
    """Specification of a metric to be computed via profiling.

    Attributes
    ----------
    build_profiler : Callable[..., tuple[Callable, Any]]
        Builds the profiler of a Jaspr from a measurement behavior and the metric's
        keyword arguments, and returns it with auxiliary data for the extraction.
    extract_metric : Callable[[tuple, Jaspr, Any], Any]
        Turns the profiler output into the user-facing result.
    simulate_fallback : Callable[..., Any]
        Computes the metric by simulation, used for ``meas_behavior="sim"``.

    """

    build_profiler: Callable[..., tuple[Callable, Any]]
    extract_metric: Callable[[tuple, Jaspr, Any], Any]
    simulate_fallback: Callable[..., Any]


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


def _normalize_meas_behavior(meas_behavior: str | Callable) -> Callable:
    """Normalize the measurement behavior into a callable.

    Parameters
    ----------
    meas_behavior : str or Callable
        ``"0"``, ``"1"``, ``"sim"``, or a callable taking a JAX PRNG key.

    Returns
    -------
    Callable
        The measurement behavior as a callable.

    Raises
    ------
    ValueError
        If ``meas_behavior`` is an unknown string.
    TypeError
        If ``meas_behavior`` is neither a string nor a callable.

    """
    if isinstance(meas_behavior, str):
        if meas_behavior == "0":
            return always_zero
        if meas_behavior == "1":
            return always_one
        if meas_behavior == "sim":
            return simulation
        raise ValueError(f"Don't know how to compute required resources via method {meas_behavior}")

    if callable(meas_behavior):
        return meas_behavior

    raise TypeError("meas_behavior must be a str or callable")


# TODO: Move each metric implementation into its dedicated module (already present).
# Keeping them here for now to avoid circular imports.


def count_ops(meas_behavior: str | Callable, callback_threshold: int | None = None) -> Callable:
    """Decorator to determine resources of large scale quantum computations.

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

    callback_threshold : int or None, optional
        For very large algorithms, compile time can blow up due to aggressively
        inlining of the Jax pipeline. ``callback_threshold`` allows to mitigate
        this by trading compilation speed for execution speed.
        ``None`` (default) disables callbacks (fastest execution).
        ``0`` wraps every reused subroutine (fastest compilation).
        ``500`` is a good middle ground for many large algorithms.

    Returns
    -------
    Callable
        A decorator, producing a function that computes the required resources.

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
        # {'t': 100, 's': 50, 't_dg': 100, 'h': 139, 'x': 22, 'cx': 577, 'measure': 49}
        print(main(5000))
        # {'t': 75025000, 's': 37512500, 't_dg': 75025000, 'h': 112527499,
        #  'x': 20002, 'cx': 500089987, 'measure': 37512499}

    Note that even though the second computation contains more than 800 million gates,
    determining the resources takes well under a second, highlighting the scalability
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
    from a simple distribution won't cut it because the required resources can
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
        # {'measure': 4, 'x': 2}
        print(main(1))
        # {'measure': 4}

    """

    def count_ops_decorator(function):
        """Turn ``function`` into a function returning its operation counts."""

        def ops_counter(*args):
            """Return the operation counts of ``function`` called with ``args``."""
            jaspr = get_cached_jaspr(function, args, meas_behavior)
            return jaspr.count_ops(*args, meas_behavior=meas_behavior, callback_threshold=callback_threshold)

        return ops_counter

    return count_ops_decorator


def depth(
    meas_behavior: str | Callable,
    max_qubits: int = 1024,
    callback_threshold: int | None = None,
) -> Callable:
    """Decorator to determine the depth of large scale quantum computations.

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

    callback_threshold : int or None, optional
        For very large algorithms, compile time can blow up due to aggressively
        inlining of the Jax pipeline. ``callback_threshold`` allows to mitigate
        this by trading compilation speed for execution speed.
        ``None`` (default) disables callbacks (fastest execution).
        ``0`` wraps every reused subroutine (fastest compilation).
        ``500`` is a good middle ground for many large algorithms.

    Returns
    -------
    Callable
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

    .. warning::

        The depth metric is an experimental feature and may not behave as expected in certain edge cases.

        -   The memory management operations ``reset`` and ``delete`` are currently ignored.
            Qubits freed by these calls still count toward the ``max_qubits`` limit.

        -   Slice bounds follow Python semantics for common cases (negative indices,
            out-of-bounds clamping, empty slices, etc.). However, non-unit steps are not supported.


    """

    def depth_decorator(function):
        """Turn ``function`` into a function returning its circuit depth."""

        def depth_counter(*args):
            """Return the circuit depth of ``function`` called with ``args``."""
            jaspr = get_cached_jaspr(function, args, meas_behavior)
            return jaspr.depth(
                *args, meas_behavior=meas_behavior, max_qubits=max_qubits, callback_threshold=callback_threshold
            )

        return depth_counter

    return depth_decorator


def _warn_max_allocations_deprecated() -> None:
    """Warn that the ``max_allocations`` argument of ``num_qubits`` has no effect anymore."""
    warnings.warn(
        "The ``max_allocations`` argument of ``num_qubits`` is deprecated and has no effect: "
        "the number of allocations is no longer bounded. It will be removed in a future release.",
        QrispDeprecationWarning,
        stacklevel=3,
    )


def num_qubits(
    meas_behavior: str | Callable,
    max_allocations: int | None = None,
    callback_threshold: int | None = None,
) -> Callable:
    """Decorator to track qubit allocation and deallocation events during a quantum computation.

    This decorator compiles a Jasp-compatible quantum function into a resource-analysis
    function that tracks qubit allocation and deallocation events throughout the computation.

    An internal allocation counter is updated as follows:

    - increased whenever qubits are allocated (e.g., via ``QuantumVariable`` creation),
    - decreased whenever qubits are explicitly deleted (e.g., via ``qv.delete()``),

    Only a fixed number of running counters is tracked, so there is no limit on the
    number of allocation and deallocation events.

    The decorated function returns a dictionary containing information about
    all allocation and deallocation events.

    These are:

    - ``total_allocated``: the total number of qubits allocated during the computation.
    - ``total_deallocated``: the total number of qubits deallocated during the computation.
    - ``peak_allocations``: the maximum number of qubits allocated at any point during the computation.
    - ``finally_allocated``: the number of qubits still allocated at the end of the computation.

    See the examples below for more details on how to interpret these values.

    Parameters
    ----------
    meas_behavior : str or callable
        A string or callable indicating the behavior of the resource computation
        when measurements are performed. Available strings are ``"0"`` and ``"1"``.
        A callable must take a JAX PRNG key as input and return a boolean.

    max_allocations : int, optional
        Deprecated and ignored. The number of allocation/deallocation events is no longer
        bounded. Passing a value emits a ``QrispDeprecationWarning``.

    callback_threshold : int or None, optional
        For very large algorithms, compile time can blow up due to aggressively
        inlining of the Jax pipeline. ``callback_threshold`` allows to mitigate
        this by trading compilation speed for execution speed.
        ``None`` (default) disables callbacks (fastest execution).
        ``0`` wraps every reused subroutine (fastest compilation).
        ``500`` is a good middle ground for many large algorithms.

    Returns
    -------
    Callable
        A decorator producing a function that returns a dictionary containing aggregated
        statistics about allocation and deallocation events during the computation.

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

        print(circuit(2, 3, 4))
        # Output:
        # {'total_allocated': 5, 'total_deallocated': 0,
        # 'peak_allocations': 5, 'finally_allocated': 5}

    Here, the measurement of the first qubit determines whether we allocate 3 or 4 additional qubits.
    The output dictionary contains information about the total number of allocated qubits (5),
    the total number of deallocated qubits (0), the peak number of allocated qubits
    at any point during the computation (5), and the number of qubits still allocated at the end of the computation (5).
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

        print(circuit(4))
        # Output:
        # {'total_allocated': 12, 'total_deallocated': 8,
        # 'peak_allocations': 8, 'finally_allocated': 4}

    Here, we first allocate 8 qubits, then deallocate them, and finally allocate 4 more qubits.

    Let's see a final example with branching and deallocation:

    ::

        from qrisp import *

        @num_qubits(meas_behavior="1")
        def circuit(num_qubits_input):

            list_of_qvs = []

            for i in range(2):
                qv = QuantumFloat(num_qubits_input)
                h(qv[i])
                list_of_qvs.append(qv)

            qv_2 = QuantumFloat(1)
            h(qv_2[0])
            m = measure(qv_2[0])

            qv_2.delete()

            with control(m == 1):
                qv4 = QuantumFloat(10)
                h(qv4[0])
                qv4.delete()

            for i in range(2):
                list_of_qvs[i].delete()

        print(circuit(8))
        # Output:
        # {'total_allocated': 27, 'total_deallocated': 27,
        # 'peak_allocations': 26, 'finally_allocated': 0}

    In this example, the peak number of allocated qubits is different from the
    total allocated because ``qv_2`` is deleted before the subsequent allocation of ``qv4``.
    The final number of allocated qubits is 0 because all allocated qubits are eventually deallocated.

    .. warning::

        Programs that include a :ref:`kernelized <quantum_kernel>` function
        cannot currently be analyzed.

    """
    if max_allocations is not None:
        _warn_max_allocations_deprecated()

    def num_qubits_decorator(function):
        """Turn ``function`` into a function returning its qubit allocation statistics."""

        def qubits_counter(*args):
            """Return the qubit allocation statistics of ``function`` called with ``args``."""
            jaspr = get_cached_jaspr(function, args, meas_behavior)
            return jaspr.num_qubits(
                *args,
                meas_behavior=meas_behavior,
                callback_threshold=callback_threshold,
            )

        return qubits_counter

    return num_qubits_decorator


def profile_jaspr(jaspr: Jaspr, mode: str, meas_behavior: str | Callable = "0", **kwargs: Any) -> Callable:
    """Profile a Jaspr according to a given metric mode.

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
        For example, `max_qubits` for depth profiling.

    Returns
    -------
    Callable
        A function that computes the specified metric when called with the same
        arguments as the original Jaspr.

    """
    meas_behavior_callable = _normalize_meas_behavior(meas_behavior)
    metric_spec = METRIC_DISPATCH[mode]

    if meas_behavior_callable.__name__ == "simulation" and metric_spec.simulate_fallback is not None:

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
        """Profile the Jaspr on ``args`` and return the extracted metric."""
        args = tree_flatten(args)[0]
        res = profiler(*args)
        return metric_spec.extract_metric(res, jaspr, aux)

    return profiler_wrapper
