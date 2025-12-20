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

import jax.numpy as jnp
from jax._src.array import ArrayImpl
from jax import jit

from qrisp.jasp.tracing_logic import check_for_tracing_mode


class JRangeIterator:

    def __init__(self, *args):

        # Differentiate between the 3 possible cases of input signature

        if len(args) == 1:
            # In the case of one input argument, this argument is the stop value
            self.start = None
            self.stop = jnp.asarray(args[0], dtype="int64")
            self.step = jnp.asarray(1, dtype="int64")
        elif len(args) == 2:
            self.start = jnp.asarray(args[0], dtype="int64")
            self.stop = jnp.asarray(args[1], dtype="int64")
            self.step = jnp.asarray(1, dtype="int64")
        elif len(args) == 3:
            # Three arguments denote the case of a non-trivial step
            self.start = jnp.asarray(args[0], dtype="int64")
            self.stop = jnp.asarray(args[1], dtype="int64")
            self.step = jnp.asarray(args[2], dtype="int64")

        # The loop index should be inclusive because this makes loop inversion
        # much easier. For more details check inv_transform.py.
        self.stop -= 1

    def __iter__(self):
        self.iteration = 0

        # We create the loop iteration index tracer
        if self.start is None:
            self.loop_index = self.stop - self.stop
        else:
            self.loop_index = self.start + 0
        return self

    def __next__(self):
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

            if qs.gc_mode == "auto":
                for qv in created_qvs:
                    reset(qv)
                    qv.delete()
            elif qs.gc_mode == "debug" and len(created_qvs):
                raise Exception(
                    f"QuantumVariables {created_qvs} went out of scope without deletion during jrange"
                )

            # Perform the incrementation
            self.loop_index += self.step

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

            if qs.gc_mode == "auto":
                for qv in created_qvs:
                    reset(qv)
                    qv.delete()
            elif qs.gc_mode == "debug" and len(created_qvs):
                raise Exception(
                    f"QuantumVariables {created_qvs} went out of scope without deletion during jrange"
                )

            self.loop_index += self.step

            self.iter_env.__exit__(None, None, None)
            raise StopIteration


def jrange(*args):
    """
    Performs a loop with a dynamic bound. Similar to the Python native ``range``,
    this iterator can receive multiple arguments. If it receives just one, this
    value is interpreted as the stop value and the start value is assumed to be 0.
    Two arguments represent start and stop value, whereas three represent start,
    stop and step.

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
    step : int
        The value to increase the loop index by after each iteration.

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

    """

    new_args = []
    if check_for_tracing_mode():
        for i in range(len(args)):
            if i == 2:
                new_args.append(args[i])
                continue
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
    if isinstance(x, list):
        return len(x)
    else:
        return x.size
