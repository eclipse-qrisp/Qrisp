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

import numpy as np

from qrisp.circuit import QuantumCircuit
from qrisp.misc.GMS_tools import GXX_wrapper


# Function to synthesize a multi controlled CX gate from GMS gates
def gms_multi_cx(n):
    qc = QuantumCircuit(n + 1)

    if n == 2:
        for i in range(3):
            qc.ry(np.pi / 2, i)

        qc.rz(np.pi / 4, 2)

        qc.append(GXX_wrapper(3, 3 * [3 * [np.pi / 2]]), qc.qubits)

        for i in range(3):
            qc.rx(-np.pi / 2, i)

        qc.rz(-np.pi / 2, 2)

        for i in range(3):
            qc.rx(-np.pi / 4, i)

        qc.append(GXX_wrapper(3, 3 * [3 * [np.pi / 4]]), qc.qubits)

        qc.rz(np.pi / 2, 2)

        qc.append(GXX_wrapper(3, 3 * [3 * [np.pi / 2]]), qc.qubits)

        for i in range(3):
            qc.rx(np.pi / 2, i)
            qc.ry(-np.pi / 2, i)

        result = qc.to_gate()

        result.name = "GMS toffoli"

        return result
    else:
        raise Exception(str(n) + "-controlled x gate not implemented for gms method")
