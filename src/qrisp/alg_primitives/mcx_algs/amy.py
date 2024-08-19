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

from qrisp import QuantumCircuit, QuantumVariable
from qrisp.alg_primitives.mcx_algs.circuit_library import amy_toffoli_qc

#Implementation of the depth 2 Toffoli found in https://arxiv.org/pdf/1206.0758.pdf
def amy_toffoli(ctrl, target, ctrl_state = "11"):
    
    qc = QuantumCircuit(4)
    
    for i in [0,1]:
        if ctrl_state[i] == "0":
            qc.x(i)
    
    qc.append(amy_toffoli_qc.to_gate("amy_qc"), qc.qubits)
    
    for i in [0,1]:
        if ctrl_state[i] == "0":
            qc.x(i)
    
    amy_anc = QuantumVariable(1)
    
    amy_gate = qc.to_gate(name = "amy_toffoli")
    amy_gate.permeability = {0 : True, 1 : True, 2 : False, 3 : False}
    amy_gate.is_qfree = True
    
    amy_anc.qs.append(amy_gate, list(ctrl) + list(target) + list(amy_anc))
    amy_anc.delete()