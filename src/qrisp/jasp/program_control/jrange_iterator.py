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

import jax.numpy as jnp
from jax._src.array import ArrayImpl
from jax import jit

from qrisp.jasp.tracing_logic import check_for_tracing_mode


# ---------------------------------------------------------------------------
# Marker function for robust identification of jrange loop index and
# threshold inside a compiled Jaxpr.  Called once per environment, right
# before __exit__, with the *updated* loop index:
#   invars[0] = updated loop index
#   invars[1] = threshold (stop value)
# Returns the updated loop index to keep the variable live.
# ---------------------------------------------------------------------------
def _jrange_marker(updated_loop_index, threshold):
    """Identity marker returning the updated loop index (invars[0])."""
    return updated_loop_index


# JIT-compile once so every call site shares the same compiled object.
_jrange_marker = jit(_jrange_marker)

# Public constant exported for use in other modules.
JRANGE_MARKER_NAME = "_jrange_marker"


class JRangeIterator:

    def __init__(self, *args):

        # Differentiate between the 2 possible cases of input signature
        if len(args) == 1:
            self.start = None
            self.stop = jnp.asarray(args[0], dtype="int64")
        elif len(args) == 2:
            self.start = jnp.asarray(args[0], dtype="int64")
            self.stop = jnp.asarray(args[1], dtype="int64")
        else:
            raise Exception("jrange only supports 1 or 2 arguments (step size 1 only)")

        # The loop index should be inclusive because this makes loop inversion
        # much easier. For more details check inv_transform.py.
        self.stop -= 1

    def __iter__(self):
        self.iteration = 0

        if self.start is None:
            self.loop_index = self.stop - self.stop
        else:
            self.loop_index = self.start
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
            self.iter_env.__enter__()

            self.iter_1_qvs = list(TracingQuantumSession.get_instance().qv_list)

            return self.loop_index

        elif self.iteration == 2:

            qs = TracingQuantumSession.get_instance()
            created_qvs = set(list(qs.qv_list)) - set(self.iter_1_qvs)
            created_qvs = list(created_qvs)
            created_qvs = sorted(created_qvs, key=lambda x: hash(x))

            # Perform the incrementation (step size 1)
            self.loop_index += 1

            # Marker called right before __exit__ with the updated
            # loop index.  invars[0] = updated loop index,
            # invars[1] = threshold.  Assignment keeps it live.
            self.loop_index = _jrange_marker(self.loop_index, self.stop)

            # Exit the old environment and enter the new one.
            self.iter_env.__exit__(None, None, None)
            self.iter_env.__enter__()

            self.iter_2_qvs = list(TracingQuantumSession.get_instance().qv_list)

            return self.loop_index

        elif self.iteration == 3:

            qs = TracingQuantumSession.get_instance()
            created_qvs = set(list(qs.qv_list)) - set(self.iter_2_qvs)
            created_qvs = list(created_qvs)
            created_qvs = sorted(created_qvs, key=lambda x: hash(x))

            self.loop_index += 1

            # Marker for the second environment, right before __exit__.
            self.loop_index = _jrange_marker(self.loop_index, self.stop)

            self.iter_env.__exit__(None, None, None)
            raise StopIteration


def jrange(*args):
    """
    Performs a loop with a dynamic bound. Similar to the Python native ``range``,
    this iterator can receive one argument (stop) or two arguments (start, stop).
    Step size is always 1.

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
    start : int, optional
        The loop index to start at. Defaults to 0.
    stop : int
        The loop index to stop at (exclusive, as in standard Python range).

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
    """

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
