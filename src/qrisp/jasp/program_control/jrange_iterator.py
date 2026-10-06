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

"""Implements jrange, a dynamic-bound loop iterator for Jasp, plus tracer and length helpers."""

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax import jit
from jax._src.array import ArrayImpl

from qrisp.jasp.tracing_logic import check_for_tracing_mode

if TYPE_CHECKING:
    from qrisp.environments import JIterationEnvironment

# Name of the jit equation that JIterationEnvironment and inv_transform.py look for.
JRANGE_MARKER_NAME = "_jrange_marker"

# The loop body is traced twice, to find the values that change between iterations.
_TRACED_ITERATIONS = 2


@jit
def _jrange_marker(updated_loop_index: Any, threshold: Any) -> Any:
    """Mark the end of a traced ``jrange`` iteration.

    Each traced iteration ends with a call to this function, which appears in the
    body as a ``jit`` equation named ``_jrange_marker``. Its inputs identify the
    incremented loop index (``invars[0]``) and the stop index (``invars[1]``).

    Parameters
    ----------
    updated_loop_index : jax.Array
        The loop index after the increment.
    threshold : jax.Array
        The stop index, which is inclusive.

    Returns
    -------
    jax.Array
        ``updated_loop_index``, so that the call is kept in the body.

    """
    del threshold  # Only needed as an input of the equation.
    return updated_loop_index


class JRangeIterator:
    """Iterator returned by :func:`jrange` in Jasp mode.

    It traces the loop body twice, each time in a ``JIterationEnvironment``. When
    the environments are flattened, the two traced iterations show which values
    the body updates, and they are compiled into a ``while_loop`` primitive.

    Parameters
    ----------
    *args : jax.Array or int
        ``stop`` or ``start, stop``, converted to 64-bit integers.

    Attributes
    ----------
    start : jax.Array or None
        The first loop index, or ``None`` if only ``stop`` was given.
    stop : jax.Array
        The last loop index, ``stop - 1``. An inclusive stop index makes it easier
        to invert the loop (see ``inv_transform.py``).
    iteration : int
        The number of ``next`` calls since the iteration started.
    loop_index : jax.Array
        The loop index of the traced iteration.
    iter_env : JIterationEnvironment
        The environment of the traced iterations.

    Raises
    ------
    ValueError
        If not given one or two arguments.

    See Also
    --------
    jrange : Returns this iterator in Jasp mode.

    Examples
    --------
    >>> from qrisp.jasp.program_control.jrange_iterator import JRangeIterator
    >>> iterator = JRangeIterator(2, 5)
    >>> int(iterator.start), int(iterator.stop)
    (2, 4)

    """

    iteration: int
    loop_index: Any
    iter_env: "JIterationEnvironment"

    def __init__(self, *args: Any) -> None:
        """Initialize the iterator from ``stop`` or ``start, stop``."""
        match args:
            case (stop,):
                self.start = None
            case (start, stop):
                self.start = jnp.asarray(start, dtype="int64")
            case _:
                raise ValueError("jrange only supports 1 or 2 arguments (step size 1 only)")
        self.stop = jnp.asarray(stop, dtype="int64") - 1

    def __iter__(self) -> "JRangeIterator":
        """Start the iteration at ``start``, or at 0.

        Returns
        -------
        JRangeIterator
            The iterator itself.

        """
        self.iteration = 0
        # stop - stop is a traced 0, so the loop index is a variable of the program.
        self.loop_index = self.stop - self.stop if self.start is None else self.start
        return self

    def __next__(self) -> Any:
        """Trace the next iteration of the loop body.

        Each iteration is traced in ``iter_env`` and ends with a call to the marker
        on the incremented loop index.

        Returns
        -------
        jax.Array
            The loop index of the iteration.

        Raises
        ------
        StopIteration
            Once both iterations are traced.

        """
        self.iteration += 1
        if self.iteration == 1:
            # qrisp.environments imports qrisp.jasp, so it is imported at call time.
            from qrisp.environments import JIterationEnvironment

            self.iter_env = JIterationEnvironment()
        elif self.iteration <= _TRACED_ITERATIONS + 1:
            # End the iteration traced since the previous call.
            self.loop_index = _jrange_marker(self.loop_index + 1, self.stop)
            self.iter_env.__exit__(None, None, None)
        if self.iteration > _TRACED_ITERATIONS:
            raise StopIteration
        self.iter_env.__enter__()
        return self.loop_index


def jrange(*args: Any) -> "JRangeIterator | range":
    """Return a range whose bounds can be traced integers.

    Like :class:`range`, ``jrange(stop)`` counts from 0 and ``jrange(start, stop)``
    from ``start`` up to ``stop - 1``, with step 1. In :ref:`Jasp <jasp>` mode, the
    bounds can be traced values, such as the size of a :ref:`QuantumVariable`
    whose size is a function argument, and the loop is compiled into a
    ``while_loop`` primitive. Outside of Jasp mode, a :class:`range` is returned.

    Parameters
    ----------
    *args : int or jax.Array
        ``stop``, or ``start`` and ``stop``. Outside of Jasp mode, they are
        converted to :class:`int`.

    Returns
    -------
    JRangeIterator or range
        In Jasp mode, an iterator that traces the loop body, otherwise a
        :class:`range`.

    Raises
    ------
    TypeError
        If not given one or two arguments. There is no ``step`` argument since
        version 0.9.

    Warnings
    --------
    In Jasp mode, the loop body is traced twice and compiled into a loop, so

    - values computed in the loop can be used in the next iteration, but not
      after the loop,
    - every iteration must apply the same operations, only the loop index changes.

    Otherwise, an exception is raised when the Jaspr is created.

    See Also
    --------
    qrisp.jasp.q_fori_loop : Loop whose results can be used after the loop.
    qrisp.jasp.q_while_loop : Loop with a traced condition.

    Notes
    -----
    For another step, compute the index from the loop variable: ``2 * k`` for
    ``k`` in ``jrange((n + 1) // 2)`` steps through ``0, 2, ..., n - 1``. To
    iterate backwards, use ``n - 1 - k``, or put the loop in an
    :ref:`InversionEnvironment`, which also inverts its operations.

    Examples
    --------
    Outside of Jasp mode, ``jrange`` is ``range``:

    >>> from qrisp.jasp import jrange
    >>> list(jrange(2, 5))
    [2, 3, 4]

    In Jasp mode, the number of iterations can depend on the arguments:

    >>> from qrisp import QuantumFloat, cx, invert, measure, x
    >>> from qrisp.jasp import boolean_simulation
    >>> @boolean_simulation
    ... def flip_all(n):
    ...     qf = QuantumFloat(n)
    ...     for i in jrange(qf.size):
    ...         x(qf[i])
    ...     return measure(qf)
    >>> int(flip_all(3)), int(flip_all(5))
    (7, 31)

    Inverting a loop reverses its order. A chain of CX gates copies the first qubit
    to all qubits, but only to the second qubit when it runs backwards:

    >>> def cx_chain(qf):
    ...     for i in jrange(qf.size - 1):
    ...         cx(qf[i], qf[i + 1])
    >>> @boolean_simulation
    ... def forward_and_backward(n):
    ...     forward, backward = QuantumFloat(n), QuantumFloat(n)
    ...     x(forward[0])
    ...     x(backward[0])
    ...     cx_chain(forward)
    ...     with invert():
    ...         cx_chain(backward)
    ...     return measure(forward), measure(backward)
    >>> [int(result) for result in forward_and_backward(4)]
    [15, 3]

    """
    if len(args) not in (1, 2):
        raise TypeError(
            f"jrange takes 1 or 2 arguments ({len(args)} given). "
            "The step argument of jrange has been removed "
            "in version 0.9. Use arithmetic on the loop variable to achieve "
            "stepping behavior."
        )

    if check_for_tracing_mode():
        # Python integers become traced values, so the bounds are variables of the program.
        bounds = [make_tracer(arg) if isinstance(arg, (int, ArrayImpl)) else arg for arg in args]
        return JRangeIterator(*bounds)

    return range(*[arg if isinstance(arg, int) else int(arg) for arg in args])


def make_tracer(x: bool | int | float | complex) -> jax.Array:
    """Return a Python scalar as a JAX array computed by a jitted function.

    In Jasp mode, the result is a traced value (the output of a ``jit`` equation
    named ``tracerizer``) instead of a constant. Booleans, integers, floats and
    complex numbers become ``bool``, ``int64``, ``float64`` and ``complex128``
    arrays.

    Parameters
    ----------
    x : bool or int or float or complex
        The value.

    Returns
    -------
    jax.Array
        A 0-dimensional array holding ``x``.

    Raises
    ------
    TypeError
        If ``x`` has another type, for example a NumPy integer.

    See Also
    --------
    jlen : Length of a list or of a traced array.

    Examples
    --------
    >>> from qrisp.jasp import make_tracer
    >>> make_tracer(3).dtype, make_tracer(2.5).dtype
    (dtype('int64'), dtype('float64'))

    """
    if isinstance(x, bool):
        dtype = jnp.bool
    elif isinstance(x, int):
        dtype = jnp.int64
    elif isinstance(x, float):
        dtype = jnp.float64
    elif isinstance(x, complex):
        dtype = jnp.complex128
    else:
        raise TypeError(f"Don't know how to tracerize type {type(x)}")

    def tracerizer() -> jax.Array:
        """Return ``x`` as an array of type ``dtype``.

        Returns
        -------
        jax.Array
            The array.

        """
        return jnp.array(x, dtype)

    return jit(tracerizer)()


def jlen(x: Any) -> Any:
    """Return the length of a list, or the size of any other object.

    In Jasp mode, the size of a :ref:`QuantumVariable` or of a qubit array can be
    a traced integer, which :func:`len` cannot return.

    Parameters
    ----------
    x : list or QuantumVariable or DynamicQubitArray or jax.Array
        A list, or an object with a ``size`` attribute.

    Returns
    -------
    int or jax.Array
        ``len(x)`` for a list, otherwise ``x.size``.

    See Also
    --------
    jrange : Range whose bounds can be traced integers.

    Examples
    --------
    >>> from qrisp import QuantumFloat
    >>> from qrisp.jasp import jlen
    >>> jlen([1, 2, 3]), jlen(QuantumFloat(4))
    (3, 4)

    """
    if isinstance(x, list):
        return len(x)
    return x.size
