"""
\********************************************************************************
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
********************************************************************************/
"""

from qrisp.core.gate_application_functions import cx
from qrisp.qtypes import QuantumFloat
from qrisp.environments import invert, control
from qrisp.jasp import jlen, jrange, while_loop, jnp


def compute_ladder_iterations(N):
    """
    Computes the number of iterations required to reduce a given size `N`
    to a value less than or equal to 2 by repeatedly halving it.
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


def X_prime_func(i, k, n):
    """
    Simulate the lists produced by the recursive calls
    of the algorithm by returning the corresponding index.
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
    return jnp.int64(
        (2 ** (k + 1) - 1)
        + 2 ** (k + 1) * i
        - 2**k * ((i + 1) == n // 2) * ((n + 1) & 1)
    )


def ladder1_synth_jax(n):
    """
    Iteratively synthesizes a logarithmic-depth CNOT circuit equivalent to a CNOT ladder of length n-1.
    This construction is based on the recursive algorithm described in: https://arxiv.org/abs/2501.16802
    Args:
        n (int): The number of qubits.

    This implementation differs from the one in the paper in two key aspects:
        - It uses an iterative structure instead of a recursive one.
        - It leverages the 'X_prime_func' function to track qubit indices instead of relying on Python lists.
    These modifications are necessary because dynamic programming does not support recursive structures or dynamically sized lists.
    To address these limitations, we first calculate the number of recursive calls that would have been made in the original algorithm.
    Then, we use 'X_prime_func', which encapsulates the structure of the lists created during recursive calls, to reference qubits in the iterative process.
    Additionally, we avoid variable updates between iterations by using bit shifts, noting that N//2 is equivalent to N>>1.
    """

    qf = QuantumFloat(n)
    N = jlen(qf)

    # Add the first layer of gates
    for i in jrange(1, (N + 1) // 2 - 1):
        cx(qf[2 * i - 1], qf[2 * i])
    cx(qf[-2], qf[-1])

    # Compute the number of recursive call in the original algorithm
    max_iterations = compute_ladder_iterations(N)

    # Fill in the left part of the circuit
    for k in jrange(max_iterations - 2):
        # CL
        cx(
            qf[X_prime_func((N >> (k + 1)) - 2, k, N >> k)],
            qf[X_prime_func((N >> (k + 1)) - 1, k, N >> k)],
        )

        for i in jrange(1, ((N >> (k + 1)) + 1) // 2 - 1):
            cx(
                qf[X_prime_func(2 * i - 1, k, N >> k)],
                qf[X_prime_func(2 * i, k, N >> k)],
            )

    # Conditionally append the deepest layer
    with control(N >> (max_iterations - 1) == 2):
        cx(
            qf[X_prime_func(0, max_iterations - 2, N >> (max_iterations - 2))],
            qf[X_prime_func(1, max_iterations - 2, N >> (max_iterations - 2))],
        )

    # Fill in the right part of the circuit
    with invert():
        for k in jrange(max_iterations - 2):
            # CR
            cx(qf[X_prime_func(0, k, N >> k)], qf[X_prime_func(1, k, N >> k)])

            for i in jrange(1, ((N >> (k + 1)) + 1) // 2 - 1):
                cx(
                    qf[X_prime_func(2 * i, k, N >> k)],
                    qf[X_prime_func(2 * i + 1, k, N >> k)],
                )

    # Add the last layer of gates
    cx(qf[0], qf[1])
    for i in jrange(1, (N + 1) // 2 - 1):
        cx(qf[2 * i], qf[2 * i + 1])
