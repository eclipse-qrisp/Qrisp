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

from qrisp import QuantumCircuit

# Adder based on https://arxiv.org/abs/1712.02630


def TR_gate():
    qc = QuantumCircuit(3)
    qc.crx(-np.pi / 2, 1, 2)
    qc.p(-np.pi / 4, 1)
    qc.cx(0, 1)

    qc.p(np.pi / 4, 0)
    qc.crx(np.pi / 2, 0, 2)

    qc.p(np.pi / 4, 1)
    qc.crx(np.pi / 2, 1, 2)

    # Error in Thapliyal paper? Doesnt work if there is no inverse here
    result = qc.to_gate().inverse()
    result.name = "TR"
    return result


def thapliyal_procedure(qc, qubit_list_1, qubit_list_2, output_qubit):
    if len(qubit_list_1) != len(qubit_list_2):
        raise Exception(
            "Tried to call Thapliyal-procedure with qubit lists of unequal length"
        )

    n = len(qubit_list_1)

    # Step 1
    for i in range(1, n):
        qc.cx(qubit_list_1[i], qubit_list_2[i])

    # Step 2
    qc.cx(qubit_list_1[-1], output_qubit)

    for i in range(n - 2, 0, -1):
        qc.cx(qubit_list_1[i], qubit_list_1[i + 1])

    # Step 3
    for i in range(n - 1):
        qc.mcx([qubit_list_1[i], qubit_list_2[i]], qubit_list_1[i + 1])

    # Step 4
    qc.append(TR_gate(), [qubit_list_1[-1], qubit_list_2[-1], output_qubit])

    for i in range(n - 2, -1, -1):
        qc.append(TR_gate(), [qubit_list_1[i], qubit_list_2[i], qubit_list_1[i + 1]])

    # Step 5
    for i in range(1, n - 1):
        qc.cx(qubit_list_1[i], qubit_list_1[i + 1])

    # Step 6
    for i in range(1, n):
        qc.cx(qubit_list_1[i], qubit_list_2[i])
