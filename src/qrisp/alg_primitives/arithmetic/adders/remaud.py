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

"""Implements Remaud's ancilla-free, polylogarithmic-depth in-place adder and its ladder-synthesis helpers."""

from qrisp.circuit import Qubit
from qrisp.core import QuantumVariable
from qrisp.core.gate_application_functions import cx, mcx, x
from qrisp.environments import control, invert
from qrisp.jasp import DynamicQubitArray, jlen, jnp, jrange, while_loop

__all__ = ["remaud_adder"]

# A qubit register as seen by the helpers below: slicing a QuantumVariable yields a
# list of Qubits in static mode and a DynamicQubitArray in dynamic (Jasp) mode.
QubitRegister = QuantumVariable | list[Qubit] | DynamicQubitArray


def _compute_ladder_iterations(N: int) -> int:
    """Computes the number of iterations required to reduce a given size `N` to at most 2 by repeatedly halving it.

    This function uses a loop to simulate the halving process and counts
    the number of iterations needed.

    Args:
        N (int): The initial size to be reduced.

    Returns:
        int: The number of iterations required to reduce `N` to a value
        less than or equal to 2.

    """
    iterations = 1
    current_size = N

    def body_fun(val):
        i, current_size = val
        current_size = current_size // 2
        i += 1
        return i, current_size

    def cond_fun(val):
        return val[1] > 2

    i, current_size = while_loop(cond_fun, body_fun, (iterations, current_size))
    return i


def _X_prime_func(i: int, k: int, n: int) -> int:
    """Simulate the lists produced by the recursive calls of the algorithm by returning the corresponding index.

    This function is derived from observing that the original algorithm generates
    qubit lists by the following pattern: Starting with a range of integers from 0 to N-1,
    skip the first index, select every other index, and append the second-to-last index
    if the list size is even.

    Example:
    N=25
    Iteration 1: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    Iteration 2: [3, 7, 11, 15, 19, 21]
    Iteration 3: [7, 15, 19]
    Iteration 4: [15]

    """
    return jnp.int64((2 ** (k + 1) - 1) + 2 ** (k + 1) * i - 2**k * ((i + 1) == n // 2) * ((n + 1) & 1))


def _ladder1_synth_jax(qf: QubitRegister) -> None:
    """Iteratively synthesizes a logarithmic-depth CNOT circuit equivalent to a CNOT ladder of length n-1.

    This construction is based on the recursive algorithm described in: https://arxiv.org/abs/2501.16802, Algorithm 1.

    Args:
        qf: QuantumVariable or list[Qubit] or DynamicQubitArray
            The qubits to be used in the circuit. Any size is supported; a register of 0 or 1 qubits yields no gates.

    This implementation differs from the one in the paper in two key aspects:
        - It uses an iterative structure instead of a recursive one.
        - It leverages the '_X_prime_func' function to track qubit indices instead of relying on Python lists.
    These modifications are necessary because dynamic programming does not support recursive structures or
    dynamically sized lists. To address these limitations, we first calculate the number of recursive calls that
    would have been made in the original algorithm. Then, we use '_X_prime_func', which encapsulates the structure
    of the lists created during recursive calls, to reference qubits in the iterative process.
    Additionally, we avoid variable updates between iterations by using bit shifts, noting that N//2 is equivalent
    to N>>1.

    """
    N = jlen(qf)

    # The general construction below assumes N >= 3 (for N == 2 it would evaluate
    # 2**k and N >> k with k = -1). A ladder on 0 or 1 qubits is empty.
    with control(N == 2):
        cx(qf[0], qf[1])

    with control(N >= 3):
        # Add the first layer of gates
        for i in jrange(1, (N + 1) // 2 - 1):
            cx(qf[2 * i - 1], qf[2 * i])
        cx(qf[-2], qf[-1])

        # Compute the number of recursive call in the original algorithm
        max_iterations = _compute_ladder_iterations(N)

        # Fill in the left part of the circuit
        for k in jrange(max_iterations - 2):
            # CL
            cx(
                qf[_X_prime_func((N >> (k + 1)) - 2, k, N >> k)],
                qf[_X_prime_func((N >> (k + 1)) - 1, k, N >> k)],
            )

            for i in jrange(1, ((N >> (k + 1)) + 1) // 2 - 1):
                cx(
                    qf[_X_prime_func(2 * i - 1, k, N >> k)],
                    qf[_X_prime_func(2 * i, k, N >> k)],
                )

        # Conditionally append the deepest layer
        with control(N >> (max_iterations - 1) == 2):
            cx(
                qf[_X_prime_func(0, max_iterations - 2, N >> (max_iterations - 2))],
                qf[_X_prime_func(1, max_iterations - 2, N >> (max_iterations - 2))],
            )

        # Fill in the right part of the circuit
        with invert():
            for k in jrange(max_iterations - 2):
                # CR
                cx(qf[_X_prime_func(0, k, N >> k)], qf[_X_prime_func(1, k, N >> k)])

                for i in jrange(1, ((N >> (k + 1)) + 1) // 2 - 1):
                    cx(
                        qf[_X_prime_func(2 * i, k, N >> k)],
                        qf[_X_prime_func(2 * i + 1, k, N >> k)],
                    )

        # Add the last layer of gates
        cx(qf[0], qf[1])
        for i in jrange(1, (N + 1) // 2 - 1):
            cx(qf[2 * i], qf[2 * i + 1])


def _ladder2_synth_jax(x: QubitRegister, y: QubitRegister, method: str = "khattar") -> None:
    """Iteratively synthesizes a polylogarithmic-depth MCX circuit equivalent to a Toffoli ladder of length n-1.

    This construction is based on the recursive algorithm described in `Algorithm 2 <https://arxiv.org/abs/2501.16802>`__.
    It generalizes _ladder1_synth_jax by alternating x and y qubits. If the circuit generated by _ladder1_synth_jax(n)
    contains cx(x[i], x[j]) then the circuit generated by _ladder2_synth_jax(n) contains mcx(x[i] + y[i:j], x[j]).
    Every MCX gate uses the Khattar method for synthesis.

    Args:
        x: QuantumVariable or list[Qubit] or DynamicQubitArray
            The first set of qubits. Any size is supported; a register of 0 or 1 qubits yields no gates.
        y: QuantumVariable or list[Qubit] or DynamicQubitArray
            The second set of qubits, one qubit shorter than x.
        method: str
            The method to use for synthesizing the MCX gates. Default is 'khattar'.

    """
    N = jlen(x)

    with control(N == 2):
        mcx([x[0], y[0]], x[1], method=method)

    with control(N >= 3):
        for i in jrange(1, (N + 1) // 2 - 1):
            mcx(
                x[2 * i - 1] + y[2 * i - 1],
                x[2 * i],
                method=method,
            )
        mcx(x[-2] + y[-1], x[-1], method=method)  # CHANGED

        max_iterations = _compute_ladder_iterations(N)
        for k in jrange(max_iterations - 2):
            # CL
            mcx(
                x[_X_prime_func((N >> (k + 1)) - 2, k, N >> k)]
                + y[_X_prime_func((N >> (k + 1)) - 2, k, N >> k) : _X_prime_func((N >> (k + 1)) - 1, k, N >> k)],
                x[_X_prime_func((N >> (k + 1)) - 1, k, N >> k)],
                method=method,
            )

            for i in jrange(1, ((N >> (k + 1)) + 1) // 2 - 1):
                mcx(
                    x[_X_prime_func(2 * i - 1, k, N >> k)]
                    + y[_X_prime_func(2 * i - 1, k, N >> k) : _X_prime_func(2 * i, k, N >> k)],
                    x[_X_prime_func(2 * i, k, N >> k)],
                    method=method,
                )

        with control(N >> (max_iterations - 1) == 2):
            mcx(
                x[_X_prime_func(0, max_iterations - 2, N >> (max_iterations - 2))]
                + y[
                    _X_prime_func(0, max_iterations - 2, N >> (max_iterations - 2)) : _X_prime_func(
                        1, max_iterations - 2, N >> (max_iterations - 2)
                    )
                ],
                x[_X_prime_func(1, max_iterations - 2, N >> (max_iterations - 2))],
                method=method,
            )

        with invert():
            for k in jrange(max_iterations - 2):
                # CR
                mcx(
                    x[_X_prime_func(0, k, N >> k)] + y[_X_prime_func(0, k, N >> k) : _X_prime_func(1, k, N >> k)],
                    x[_X_prime_func(1, k, N >> k)],
                    method=method,
                )

                for i in jrange(1, ((N >> (k + 1)) + 1) // 2 - 1):
                    mcx(
                        x[_X_prime_func(2 * i, k, N >> k)]
                        + y[_X_prime_func(2 * i, k, N >> k) : _X_prime_func(2 * i + 1, k, N >> k)],
                        x[_X_prime_func(2 * i + 1, k, N >> k)],
                        method=method,
                    )

        mcx([x[0], y[0]], x[1], method=method)
        for i in jrange(1, (N + 1) // 2 - 1):
            mcx([x[2 * i], y[2 * i]], x[2 * i + 1], method=method)


def _remaud_core(a: QubitRegister, b: QubitRegister, z: Qubit) -> None:
    """Ancilla-free in-place addition ``b += a`` with carry-out, as in Algorithm 3 of arXiv:2501.16802.

    a and b must have the same size n >= 1. The carry out of the addition is XORed into z. a is left unchanged.

    Args:
        a: QuantumVariable or list[Qubit] or DynamicQubitArray
            The register that is added.
        b: QuantumVariable or list[Qubit] or DynamicQubitArray
            The register that is modified in place.
        z: Qubit
            The qubit receiving the carry out (z ^= carry).

    """
    n = jlen(a)

    for i in jrange(1, n):
        cx(a[i], b[i])

    _ladder1_synth_jax(a[1:] + z)

    with invert():
        _ladder2_synth_jax(a[:] + z, b[:])

    for i in jrange(1, n):
        cx(a[i], b[i])

    x(b[1:-1])

    _ladder2_synth_jax(a[:], b[:-1])

    x(b[1:-1])

    with invert():
        _ladder1_synth_jax(a[1:])

    for i in jrange(n):
        cx(a[i], b[i])


def remaud_adder(a: QuantumVariable, b: QuantumVariable, z: QuantumVariable) -> None:
    """In-place adder function based on `this paper, Algorithm 3 <https://arxiv.org/abs/2501.16802>`__.

    Performs the addition:

    ::

        b += a

    with a polylogarithmic depth and no use of auxiliary qubits.
    Both `a` and `b` should have the same size `N` and the result of the overflow is stored in the carry Qubit.

    Parameters
    ----------
    a : int or QuantumVariable or list[Qubit]
        The value that should be added.
    b : QuantumVariable or list[Qubit]
        The value that should be modified in the in-place addition.
    z : QuantumVariable
        A single-qubit variable receiving the carry value resulting from the overflow of the addition.

    Examples
    --------
    We add two integers:

    >>> from qrisp import QuantumFloat, remaud_adder
    >>> a = QuantumFloat(4)
    >>> b = QuantumFloat(4)
    >>> z = QuantumFloat(1)
    >>> a[:] = 4
    >>> b[:] = 5
    >>> remaud_adder(a,b,z)
    >>> print(b)
    {9: 1.0}

    """
    _remaud_core(a, b, z[0])
