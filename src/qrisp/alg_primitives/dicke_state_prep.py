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

from collections.abc import Sequence
from typing import Literal
import math
from itertools import accumulate

import jax.numpy as jnp

from qrisp.circuit import Qubit
from qrisp.core import QuantumVariable, cx, ry, x
from qrisp.jasp import jlen, jrange
from qrisp.environments import control


def dicke_state(
    qv: QuantumVariable | Sequence[Qubit],
    k: int,
    method: Literal["deterministic", "divide-and-conquer"] = "divide-and-conquer",
) -> None:
    """Dicke State initialization of a QuantumVariable, based on the deterministic alogrithm in https://arxiv.org/abs/1904.07358.
    This algorithm creates an equal superposition of Dicke states for a given Hamming weight. The initial input variable has to be within this subspace.

    Parameters
    ----------
    qv : QuantumVariable
        Initial quantum variable to be prepared. Has to be in target subspace.
    k : int
        The Hamming weight (i.e. number of "ones") for the desired dicke state.


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

    large_k = False
    if k > n // 2:
        large_k = True
        x(qv[n - k : k])  # Partially undo the initial state.
        k = n - k

    if method == "deterministic":
        apply_dicke_unitary(qv, n, k)
    elif method == "divide-and-conquer":
        n1 = (n + 1) // 2  # ceil(n/2)
        n2 = n // 2  # floor(n/2)

        divide(qv, n1, n2, k)

        apply_dicke_unitary(qv[:n1], n1, k)
        apply_dicke_unitary(qv[n1:], n2, k)

    else:
        raise ValueError(f"Unknown `method`: {method}. Possible methods are: 'deterministic' and 'divide-and-conquer'.")

    if large_k:
        x(qv)


def divide(qv: QuantumVariable | Sequence[Qubit], n1: int, n2: int, k: int) -> None:

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


def apply_dicke_unitary(qv: QuantumVariable | Sequence[Qubit], n: int, k: int) -> None:

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
