"""********************************************************************************
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

import math
from collections.abc import Sequence
from itertools import accumulate
from typing import Literal

import jax.numpy as jnp

from qrisp.circuit import Qubit
from qrisp.core import QuantumVariable, cx, ry, x
from qrisp.environments import control
from qrisp.jasp import jlen, jrange


def dicke_state(
    qv: QuantumVariable | Sequence[Qubit],
    k: int,
    method: Literal["deterministic", "divide-and-conquer"] = "divide-and-conquer",
) -> None:
    """Dicke State initialization of a QuantumVariable.
    
    A Dicke state is an equal positive superposition of all basis states with a given Hamming weight. We label the Dicke
    state of Hamming weight :math:`k` on :math:`n` qubits as :math:`D(n, k)`.
    

    Parameters
    ----------
    qv : QuantumVariable
        Initial quantum variable to be prepared. Has to be in the state |00...011...1> where the number of 1's is equal
        to ``k``.
    k : int
        The Hamming weight (i.e. number of "ones") for the desired Dicke state.
    method : Literal["deterministic", "divide-and-conquer"]
        The method to be used for preparing the Dicke state. "deterministic" implements the preparation from
        https://arxiv.org/pdf/1904.07358. "divide-and-conquer" implements the preparation from
        https://arxiv.org/pdf/2112.12435. The code largely uses the notation from these two papers.


    Examples
    --------
    We initiate a QuantumVariable in the "0011" state and from this create the Dicke state with Hamming weight 2.

    ::

        from qrisp import QuantumVariable, x, dicke_state

        qv = QuantumVariable(4)
        x(qv[2])
        x(qv[3])

        dicke_state(qv, 2)

    """
    n = jlen(qv)

    # If k > n/2, it is easier to create D(n, n-k) instead of D(n, k), and then apply the X gate to all qubits.
    large_k = k > n // 2
    if large_k:
        x(qv[n - k : k])  # Partially undo the initial state, reducing its Hamming weight from k to n-k.
        k = n - k

    if method == "deterministic":
        _apply_dicke_unitary(qv, n, k)
    elif method == "divide-and-conquer":
        n1 = (n + 1) // 2  # ceil(n/2)
        n2 = n // 2  # floor(n/2)

        _divide(qv, n1, n2, k)

        _apply_dicke_unitary(qv[:n1], n1, k)
        _apply_dicke_unitary(qv[n1:], n2, k)

    else:
        raise ValueError(f"Unknown `method`: {method}. Possible methods are: 'deterministic' and 'divide-and-conquer'.")

    # If we were prepating D(n, n-k), now we apply the X gates to transform it into D(n, k).
    if large_k:
        x(qv)


def _divide(qv: QuantumVariable | Sequence[Qubit], n1: int, n2: int, k: int) -> None:
    r"""
    The "divide" step of the "divide-and-conquer" method of creating Dicke states.

    This takes a computational basis state consisting of ``n1 + n2 - k`` zeros followed by ``k`` ones and changes it
    into a superposition :math:`\frac{1}{\sqrt{\binom{n}{k}}} \sum_{k_1 = 0}^k \sqrt{\binom{n_1}{k_1} \binom{n_2}{k-k_1}} |0\rangle^{\otimes n_1-k_1} |1\rangle^{\otimes k_1} |0\rangle^{\otimes n_2-k+k_1} |1\rangle^{\otimes k-k_1}`,
    ready for the "conquer" step.
    
    Parameters
    ----------
    qv : QuantumVariable
        The quantum variable to be divided. Has to be in the state |00...011...1> where the number of 1's is equal
        to ``k``.
    n1 : int
        The size of the first half of the quantum variable.
    n2 : int
        The size of the second half of the quantum variable.
    k : int
        The Hamming weight (i.e. number of "ones") of the Dicke state to be constructed.
    
    """

    # Variables that follow :math:`x_i` and :math:`s_i` from the paper https://arxiv.org/pdf/2112.12435 (page 8).
    xi = [math.comb(n1, i) * math.comb(n2, k - i) for i in range(k + 1)]
    si = list(accumulate(reversed(xi)))
    si.reverse()

    for i in range(k):
        param = 2 * jnp.arccos(jnp.sqrt(xi[i] / si[i]))
        if i == 0:
            ry(param, qv[n1 - 1 - i])
        else:
            with control(qv[n1 - i]):
                ry(param, qv[n1 - 1 - i])

    for i in range(k):
        cx(qv[n1 - 1 - i], qv[n1 + n2 - k + i])


def _apply_dicke_unitary(qv: QuantumVariable | Sequence[Qubit], n: int, k: int) -> None:
    """
    The Dicke unitary constructed according to Lemma 2 in https://arxiv.org/pdf/1904.07358.

    Parameters
    ----------
    qv : QuantumVariable
        Initial quantum variable to be prepared. Has to be in target subspace.
    n : int
        The size of the quantum variable.
    k : int
        The Hamming weight (i.e. number of "ones") of the Dicke state to be constructed.
    """

    for offset in jrange(n - k):
        index2 = n - offset
        split_cycle_shift(qv, index2, k)

    for offset in jrange(k - 1):
        index = k - offset
        split_cycle_shift(qv, index, index - 1)


def split_cycle_shift(qv: QuantumVariable | Sequence[Qubit], highIndex: int, lowIndex: int) -> None:
    """Helper function for Dicke State initialization of a QuantumVariable, based on the deterministic alogrithm in https://arxiv.org/abs/1904.07358.

    Parameters
    ----------
    qv : QuantumVariable
        Initial quantum variable to be prepared. Has to be in target subspace.
    highIndex : int
        Index for indication of preparation steps, as seen in original algorithm.
    lowIndex : int
        Index for indication of preparation steps, as seen in original algorithm.

    """

    if len(qv) == 1:
        return  # If there is just one qubit, do nothing.

    # index == highIndex
    param = 2 * jnp.arccos(jnp.sqrt(1 / highIndex))
    cx(qv[highIndex - 2], qv[highIndex - 1])
    with control(qv[highIndex - 1]):
        ry(param, qv[highIndex - 2])
    cx(qv[highIndex - 2], qv[highIndex - 1])

    # index != highIndex
    for i in jrange(1, lowIndex):
        index = highIndex - i
        param = 2 * jnp.arccos(jnp.sqrt((highIndex - index + 1) / (highIndex)))

        cx(qv[index - 2], qv[highIndex - 1])
        with control([qv[highIndex - 1], qv[index - 1]]):
            ry(param, qv[index - 2])
        cx(qv[index - 2], qv[highIndex - 1])
