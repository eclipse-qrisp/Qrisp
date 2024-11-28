"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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

from qrisp import QuantumCircuit
from qrisp.alg_primitives.mcx_algs.circuit_library import toffoli_qc

# Function to synthesize a multi controlled X gate using gray synthesis
def gray_multi_cx(n):
    qs = QuantumCircuit(n+1)
    if n == 1:
        qs.cx(qs.qubits[0], qs.qubits[1])
    elif n == 2:
        qs = toffoli_qc.copy()

    else:
        qs = QuantumCircuit(n+1)
        qs.h(qs.qubits[-1])
        from qrisp.alg_primitives.logic_synthesis import GraySynthGate
        gsg = GraySynthGate((2**(n+1)-1)*[0] + [np.pi])
        qs.append(gsg, qs.qubits)
        qs.h(qs.qubits[-1])

    result = qs.to_gate()
    result.name = "gray multi cx"

    return result