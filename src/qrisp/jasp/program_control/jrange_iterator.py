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

import warnings

import jax.numpy as jnp
from jax._src.array import ArrayImpl
from jax import jit

from qrisp.jasp.tracing_logic import check_for_tracing_mode
from qrisp.misc.exceptions import QrispDeprecationWarning


class JRangeIterator:
    """Iterator used by :func:`jrange` in JASP tracing mode.

    Traces two loop iterations to capture the quantum operations performed
    inside the loop body. The collected environments are compiled into a
    JAX loop primitive by :class:`~qrisp.environments.JIterationEnvironment`.

    Parameters
    ----------
    *args : int
        One argument: stop value (start defaults to 0).
        Two arguments: start and stop values.

    """

    def __init__(self, *args):

        # Differentiate between the 2 possible cases of input signature

        if len(args) == 1:
            # In the case of one input argument, this argument is the stop value
            self.start = None
            self.stop = jnp.asarray(args[0], dtype="int64")
        elif len(args) == 2:
            self.start = jnp.asarray(args[0], dtype="int64")
            self.stop = jnp.asarray(args[1], dtype="int64")

        # The loop index should be inclusive because this makes loop inversion
        # much easier. For more details check inv_transform.py.
        self.stop -= 1

    def __iter__(self):
        """Initialise the loop index and reset the iteration counter."""

        self.iteration = 0

        # We create the loop iteration index tracer
        if self.start is None:
            self.loop_index = self.stop - self.stop
        else:
            self.loop_index = self.start + 0
        return self

    def __next__(self):
        """Run the next tracing step.

        Traces two iterations inside a
        :class:`~qrisp.environments.JIterationEnvironment` to capture the
        quantum operations performed by the loop body. The first iteration
        opens the environment; the second closes it and opens a new one to
        detect created qubits and compute the loop semantics.
        """

        # The idea is now to trace two iterations to capture what values get
        # updated after each iteration.
        # We capture the loop semantics using the JIterationEnvironment.
        # The actual jax loop primitive is then compiled in
        # JIterationEnvironment.jcompile
        from qrisp.jasp import TracingQuantumSession
        from qrisp import reset

        self.iteration += 1
        if self.iteration == 1:
            from qrisp.environments import JIterationEnvironment

            self.iter_env = JIterationEnvironment()
            # Enter the environment
            self.iter_env.__enter__()

            # We perform a trivial addition on the loop cancelation index.
            # This way the loop cancelation index will appear in the collected
            # quantum environment jaxpr and can therefore be identified as such.
            self.stop + 0

            self.iter_1_qvs = list(TracingQuantumSession.get_instance().qv_list)

            return self.loop_index

        elif self.iteration == 2:

            qs = TracingQuantumSession.get_instance()
            created_qvs = set(list(qs.qv_list)) - set(self.iter_1_qvs)
            created_qvs = list(created_qvs)
            created_qvs = sorted(created_qvs, key=lambda x: hash(x))

            # Perform the incrementation
            self.loop_index += 1

            # Exit the old environment and enter the new one.
            self.iter_env.__exit__(None, None, None)
            self.iter_env.__enter__()
            # Similar to the incrementation above

            self.stop + 0
            self.iter_2_qvs = list(TracingQuantumSession.get_instance().qv_list)

            return self.loop_index

        elif self.iteration == 3:

            qs = TracingQuantumSession.get_instance()
            created_qvs = set(list(qs.get_instance().qv_list)) - set(self.iter_2_qvs)
            created_qvs = list(created_qvs)
            created_qvs = sorted(created_qvs, key=lambda x: hash(x))

            self.loop_index += 1

            self.iter_env.__exit__(None, None, None)
            raise StopIteration


def jrange(*args):
    """
    Performs a loop with a dynamic bound. Similar to the Python native ``range``
    with one or two arguments. If it receives just one argument, this value is
    interpreted as the stop value and the start value is assumed to be 0.
    Two arguments represent start and stop value.

    .. note::

        The ``step`` parameter was removed in version 0.9. Use a step-1
        loop and multiply the loop variable by your desired step.

    .. warning::

        Similar to the :ref:`ClControlEnvironment <ClControlEnvironment>`, this feature must not have
        external carry values, implying values computed within the loop can't
        be used outside of the loop. It is however possible to carry on values
        from the previous iteration.

    .. warning::

        Each loop iteration must perform exactly the same instructions - the only
        thing that changes is the loop index

    Parameters
    ----------
    start : int
        The loop index to start at.
    stop : int
        The loop index to stop at.

    Examples
    --------

    We construct a function that encodes an integer into an arbitrarily sized
    :ref:`QuantumVariable`:

    ::

        from qrisp import QuantumFloat, control, x
        from qrisp.jasp import jrange, make_jaspr

        @qache
        def int_encoder(qv, encoding_int):

            for i in jrange(qv.size):
                with control(encoding_int & (1<<i)):
                    x(qv[i])

        def test_f(a, b):

            qv = QuantumFloat(a)

            int_encoder(qv, b+1)

            return measure(qv)

        jaspr = make_jaspr(test_f)(1,1)

    Test the result:

    >>> jaspr(5, 8)
    9
    >>> jaspr(5, 9)
    10

    We now give examples that violate the above rules (ie. no carries and changing
    iteration behavior).

    To create a loop with carry behavior we simply return the final loop index

    ::

        @qache
        def int_encoder(qv, encoding_int):

            for i in jrange(qv.size):
                with control(encoding_int & (1<<i)):
                    x(qv[i])
            return i


        def test_f(a, b):

            qv = QuantumFloat(a)

            int_encoder(qv, b+1)

            return measure(qv)

        jaspr = make_jaspr(test_f)(1,1)

    >>> jaspr(5, 8)
    Exception: Found jrange with external carry value

    To demonstrate the second kind of illegal behavior, we construct a loop
    that behaves differently on the first iteration:

    ::

        @qache
        def int_encoder(qv, encoding_int):

            flag = True
            for i in jrange(qv.size):
                if flag:
                    with control(encoding_int & (1<<i)):
                        x(qv[i])
                else:
                    x(qv[0])
                flag = False

        def test_f(a, b):

            qv = QuantumFloat(a)

            int_encoder(qv, b+1)

            return measure(qv)

        jaspr = make_jaspr(test_f)(1,1)

    In this script, ``int_encoder`` defines a boolean flag that changes the
    semantics of the iteration behavior. After the first iteration the flag
    is set to ``False`` such that the alternate behavior is activated.

    >>> jaspr(5, 8)
    Exception: Jax semantics changed during jrange iteration

    Since the ``step`` argument has been removed as of v0.9, multiply the loop
    variable by your desired step inside the body.

    The following example steps through every second qubit (equivalent to step 2):

    ::

        from qrisp.jasp import jrange, make_jaspr, qache
        from qrisp import QuantumFloat, x, measure

        @qache
        def stepped_loop(qv):
            # Number of iterations for step 2
            n = (qv.size + 1) // 2
            # Step-1 loop
            for k in jrange(n):
                # Multiply by the desired step
                i = 2 * k
                x(qv[i])

        def test_f(a):
            qv = QuantumFloat(a)
            stepped_loop(qv)
            return measure(qv)

        jaspr = make_jaspr(test_f)(1)

    >>> jaspr(3)
    5
    >>> jaspr(4)
    5

    Reversing a ``jrange`` loop (equivalent to step size -1) can be done in
    two ways.

    The first is to compute the index manually:

    ::

        from qrisp.jasp import jrange, make_jaspr, qache
        from qrisp import QuantumFloat, x, measure

        @qache
        def reversed_loop(qv):
            # Step-1 loop
            for j in jrange(qv.size):
                # Compute index in reverse
                i = qv.size - j - 1
                x(qv[i])

        def test_f(a):
            qv = QuantumFloat(a)
            reversed_loop(qv)
            return measure(qv)

        jaspr = make_jaspr(test_f)(1)

    >>> jaspr(3)
    7
    >>> jaspr(4)
    15

    The second way is to wrap the forward loop in an
    :meth:`~qrisp.environments.InversionEnvironment`:

    First, the forward loop without inversion:

    ::

        from qrisp import QuantumVariable, x, invert
        from qrisp.jasp import jrange, make_jaspr, qache

        @qache
        def loop_with_offset(qv, start):
            # Forward jrange loop
            for i in jrange(qv.size - start):
                # Offset the loop variable by start
                x(qv[i + start])

        def test_f(a):
            qv = QuantumVariable(a)
            loop_with_offset(qv, 2)
            return measure(qv)

        jaspr = make_jaspr(test_f)(1)

    >>> jaspr(4)
    12

    This applies ``x`` to qubits 2 and 3, giving state ``|0011⟩``.
    Wrapping the same loop in ``invert()`` reverses the iteration order and
    daggers the operations:

    ::

        @qache
        def reversed_loop_with_offset(qv, start):
            # Reverses the enclosed loop
            with invert():
                # Same forward loop, now runs backwards
                for i in jrange(qv.size - start):
                    x(qv[i + start])

        def test_f_rev(a):
            qv = QuantumVariable(a)
            reversed_loop_with_offset(qv, 2)
            return measure(qv)

        jaspr_rev = make_jaspr(test_f_rev)(1)

    >>> jaspr_rev(4)
    12

    Because ``x`` is self-inverse, the result is the same — the loop still
    iterates from ``qv.size - start - 1`` down to ``start``. JASP handles
    the reversed iteration and proper daggers automatically, including at
    higher nesting levels.

    """

    if len(args) == 3:
        warnings.warn(
            "DeprecationWarning: The step argument of jrange has been removed "
            "in version 0.9. Use arithmetic on the loop variable to achieve "
            "stepping behavior.",
            QrispDeprecationWarning,
            stacklevel=2,
        )

    if len(args) not in (1, 2):
        raise TypeError(
            f"jrange takes 1 or 2 arguments ({len(args)} given)"
        )

    new_args = []
    if check_for_tracing_mode():
        for i in range(len(args)):
            if isinstance(args[i], (int, ArrayImpl)):
                new_args.append(make_tracer(args[i]))
            else:
                new_args.append(args[i])

        return JRangeIterator(*new_args)

    else:
        for i in range(len(args)):
            if not isinstance(args[i], int):
                new_args.append(int(args[i]))
            else:
                new_args.append(args[i])

        return range(*new_args)


def make_tracer(x):
    """Create a JIT-compiled tracer from a Python scalar.

    Parameters
    ----------
    x : bool, int, float, or complex
        The value to convert into a tracer.

    Returns
    -------
    ArrayImpl
        A traced JAX array representing the given value.

    Raises
    ------
    Exception
        If the type of *x* is not supported.
    """
    if isinstance(x, bool):
        dtype = jnp.bool
    elif isinstance(x, int):
        dtype = jnp.int64
    elif isinstance(x, float):
        dtype = jnp.float64
    elif isinstance(x, complex):
        dtype = jnp.complex32
    else:
        raise Exception(f"Don't know how to tracerize type {type(x)}")

    def tracerizer():
        return jnp.array(x, dtype)

    return jit(tracerizer)()


def jlen(x):
    """Return the length of *x*, supporting both lists and JAX arrays.

    Parameters
    ----------
    x : list or ArrayImpl
        The object whose length to return.

    Returns
    -------
    int
        ``len(x)`` if *x* is a list, otherwise ``x.size``.
    """
    if isinstance(x, list):
        return len(x)
    else:
        return x.size
