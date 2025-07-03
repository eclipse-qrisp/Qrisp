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

from qrisp.core.gate_application_functions import x, cx, mcx
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


def ladder1_synth_jax(qf):
    """
    Iteratively synthesizes a logarithmic-depth CNOT circuit equivalent to a CNOT ladder of length n-1.
    This construction is based on the recursive algorithm described in: https://arxiv.org/abs/2501.16802, Algorithm 1.
    Args:
        qf: QuantumFloat
            The quantum float representing the qubits to be used in the circuit.

    This implementation differs from the one in the paper in two key aspects:
        - It uses an iterative structure instead of a recursive one.
        - It leverages the 'X_prime_func' function to track qubit indices instead of relying on Python lists.
    These modifications are necessary because dynamic programming does not support recursive structures or dynamically sized lists.
    To address these limitations, we first calculate the number of recursive calls that would have been made in the original algorithm.
    Then, we use 'X_prime_func', which encapsulates the structure of the lists created during recursive calls, to reference qubits in the iterative process.
    Additionally, we avoid variable updates between iterations by using bit shifts, noting that N//2 is equivalent to N>>1.
    """
    N = jlen(qf)

    # with control(N == 1):
    #     raise ValueError("The number of qubits should be greater than or equal to 2.")

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


def ladder2_synth_jax(x, y, method="khattar"):
    """
    Iteratively synthesizes a polylogarithmic-depth MCX circuit equivalent to a Toffoli ladder of length n-1.
    This construction is based on the recursive algorithm described in: https://arxiv.org/abs/2501.16802, Algorithm 2.
    It generalizes the ladder ladder1_synth_jax by alternating x and y qubits. If the circutit generated
    by ladder1_synth_jax(n) contains cx(x[i], x[j]) then the circuit generated by ladder2_synth_jax(n) contains mcx(x[i] + y[i:j], x[j]).
    Every MCX gate uses the Khattar method for synthesis.

    Args:
        x: QuantumFloat
            The first set of qubits.
        y: QuantumFloat
            The second set of qubits.
        method: str
            The method to use for synthesizing the MCX gates. Default is 'khattar'.
    """

    N = jlen(x)

    for i in jrange(1, (N + 1) // 2 - 1):
        mcx(
            x[2 * i - 1] + y[2 * i - 1],
            x[2 * i],
            method=method,
        )
    mcx(x[-2] + y[-1], x[-1], method=method) #CHANGED

    max_iterations = compute_ladder_iterations(N)
    for k in jrange(max_iterations - 2):
        # CL
        mcx(
            x[X_prime_func((N >> (k + 1)) - 2, k, N >> k)]
            + y[
                X_prime_func((N >> (k + 1)) - 2, k, N >> k) : X_prime_func(
                    (N >> (k + 1)) - 1, k, N >> k
                )
            ],
            x[X_prime_func((N >> (k + 1)) - 1, k, N >> k)],
            method=method,
        )

        for i in jrange(1, ((N >> (k + 1)) + 1) // 2 - 1):

            mcx(
                x[X_prime_func(2 * i - 1, k, N >> k)]
                + y.reg[
                    X_prime_func(2 * i - 1, k, N >> k) : X_prime_func(2 * i, k, N >> k)
                ],
                x[X_prime_func(2 * i, k, N >> k)],
                method=method,
            )

    with control(N >> (max_iterations - 1) == 2):
        mcx(
            x[X_prime_func(0, max_iterations - 2, N >> (max_iterations - 2))]
            + y[
                X_prime_func(
                    0, max_iterations - 2, N >> (max_iterations - 2)
                ) : X_prime_func(1, max_iterations - 2, N >> (max_iterations - 2))
            ],
            x[X_prime_func(1, max_iterations - 2, N >> (max_iterations - 2))],
            method=method,
        )

    with invert():
        for k in jrange(max_iterations - 2):
            # CR
            mcx(
                x[X_prime_func(0, k, N >> k)]
                + y[X_prime_func(0, k, N >> k) : X_prime_func(1, k, N >> k)],
                x[X_prime_func(1, k, N >> k)],
                method=method,
            )

            for i in jrange(1, ((N >> (k + 1)) + 1) // 2 - 1):
                mcx(
                    x[X_prime_func(2 * i, k, N >> k)]
                    + y.reg[
                        X_prime_func(2 * i, k, N >> k) : X_prime_func(
                            2 * i + 1, k, N >> k
                        )
                    ],
                    x[X_prime_func(2 * i + 1, k, N >> k)],
                    method=method,
                )

    mcx([x[0], y[0]], x[1], method=method)
    for i in jrange(1, (N + 1) // 2 - 1):
        mcx([x[2 * i], y[2 * i]], x[2 * i + 1], method=method)


def remaud_adder(A, B, Z):
    """Implements the Remauad ripple-carry logarithmic adder with a polylogarithmic depth.
    This adder is based on the algorithm described in: https://arxiv.org/abs/2501.16802, Algorithm 3.
    It adds two n-bit numbers a and b, it stores the result of the addition into B and the carry bit in Z. A is left invariant
    The algorithm makes extensive use of the ladder1_synth_jax and ladder2_synth_jax functions to synthesize the necessary circuits.

    Args:
        A: QuantumFloat
            The first n-bit to be added.
        B: QuantumFloat
            The second n-bit to be added.
        Z: QuantumFloat
            The single-bit register to store the carry bit.
    """
    n = jlen(A)

    for i in jrange(1, n):
        cx(A[i], B[i])
    
    ladder1_synth_jax(A[1:] + Z[:])

    with invert():
        ladder2_synth_jax(A[:] + Z[:], B[:]) 
    
    for i in jrange(1, n):
        cx(A[i], B[i])

    x(B[1:-1])
    
    ladder2_synth_jax(A[:], B[:-1])
    
    x(B[1:-1])
    
    with invert():
        ladder1_synth_jax(A[1:])
    
    for i in jrange(n):
        cx(A[i], B[i])
