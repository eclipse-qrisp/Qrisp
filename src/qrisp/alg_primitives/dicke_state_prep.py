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

import numpy as np

from qrisp.core import cx, ry


def dicke_state(qv, k):
    """
    Dicke State initialization of a QuantumVariable, based on the deterministic alogrithm in https://arxiv.org/abs/1904.07358.
    This algorithm creates an equal superposition of Dicke states for a given Hamming weight. The initial input variable has to be within this subspace.

    Parameters
    ----------
    qv : QuantumVariable
        Initial quantum variable to be prepared. Has to be in target subspace.
    k : Int
        The Hamming weight (i.e. number of "ones") for the desired dicke state


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

    n = len(qv)
    for index2 in reversed(range(k + 1, n + 1)):
        split_cycle_shift(qv, index2, k)

    for index in reversed(range(2, k + 1)):
        split_cycle_shift(qv, index, index - 1)


def split_cycle_shift(qv, highIndex, lowIndex):
    """
    Helper function for Dicke State initialization of a QuantumVariable, based on the deterministic alogrithm in https://arxiv.org/abs/1904.07358.

    Parameters
    ----------
    qv : QuantumVariable
        Initial quantum variable to be prepared. Has to be in target subspace.
    highIndex : Int
        Index for indication of preparation steps, as seen in original algorithm.
    lowIndex : Int
        Index for indication of preparation steps, as seen in original algorithm.
    """

    from qrisp import control

    index_range = [highIndex - i for i in range(lowIndex)]
    for index in index_range:
        param = 2 * np.arccos(np.sqrt((highIndex - index + 1) / (highIndex)))

        if index == highIndex:
            cx(qv[highIndex - 2], qv[highIndex - 1])
            with control(qv[highIndex - 1]):
                ry(param, qv[highIndex - 2])
            cx(qv[highIndex - 2], qv[highIndex - 1])

        else:
            cx(qv[index - 2], qv[highIndex - 1])
            with control([qv[highIndex - 1], qv[index - 1]]):
                ry(param, qv[index - 2])
            cx(qv[index - 2], qv[highIndex - 1])
