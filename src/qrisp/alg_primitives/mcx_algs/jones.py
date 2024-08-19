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

from qrisp import QuantumCircuit, Operation, QuantumVariable
from qrisp.alg_primitives.mcx_algs.circuit_library import jones_toffoli_qc, amy_toffoli_qc, ctrl_state_wrap

# Implementation of the T-Depth 1 Toffoli found in https://arxiv.org/pdf/1212.5069.pdf
class JonesToffoli(Operation):
    
    def __init__(self, ctrl_state = "11"):
        
        
        definition = QuantumCircuit(5)
        
        if ctrl_state[0] == "0":
            definition.x(definition.qubits[0])
        if ctrl_state[1] == "0":
            definition.x(definition.qubits[1])
        
        definition.append(amy_toffoli_qc.to_gate("jones_toffoli"), definition.qubits[:-1])
        
        if ctrl_state[0] == "0":
            definition.x(definition.qubits[0])
        if ctrl_state[1] == "0":
            definition.x(definition.qubits[1])
        
        name = "uncompiled_jones_toffoli"
        
        Operation.__init__(self, name = name, num_qubits = 5, definition = definition)
        
        self.permeability = {0: True, 1: True, 2: False, 3 : False, 4 : False}
        self.is_qfree = True
        self.ctrl_state = ctrl_state
    
    def recompile(self):
        
        res = ctrl_state_wrap(jones_toffoli_qc, self.ctrl_state).to_op("compiled_jones_toffoli")
        res.permeability = {0: True, 1: True, 2: False, 3 : False, 4 : False}
        res.is_qfree = True
            
        return res
    
    def inverse(self):
        return JonesToffoli(self.ctrl_state)
    
def jones_toffoli(ctrl, target, ctrl_state = "11"):
    if isinstance(target, (list, QuantumVariable)):
        target = target[0]
    
    jones_ancilla = QuantumVariable(2, qs = ctrl[0].qs(), name = "jones_ancilla_" + str(int(QuantumVariable.creation_counter[0])))
    
    target.qs().append(JonesToffoli(ctrl_state), list(ctrl) + [target] + list(jones_ancilla))
    jones_ancilla.delete()