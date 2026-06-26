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

import jax.numpy as jnp

from qrisp.circuit import Qubit
from qrisp.core import QuantumVariable, cx, ry
from qrisp.jasp import jlen, jrange


def dicke_state(qv: QuantumVariable | Sequence[Qubit], k: int) -> None:
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
    from qrisp.environments import control

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
