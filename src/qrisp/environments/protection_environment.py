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

from qrisp.environments.quantum_environments import QuantumEnvironment

def custom_control(func):
    
    def adaptive_control_function(*args, **kwargs):
        
        from qrisp.core import recursive_qs_search
        from qrisp import merge, ControlEnvironment, ConditionEnvironment, QuantumEnvironment, InversionEnvironment, ConjugationEnvironment
        
        qs_list = recursive_qs_search([args, kwargs])
        
        merge(qs_list)
        
        if len(qs_list) == 0:
            return func(*args, **kwargs)
        
        qs = qs_list[0]
        
        control_qb = None
        for env in qs.env_stack[::-1]:
            if isinstance(env, (ControlEnvironment, ConditionEnvironment)):
                control_qb = env.condition_truth_value
                break
            if not isinstance(env, (QuantumEnvironment, InversionEnvironment, ConjugationEnvironment)):
                break

        if control_qb is None:
            return func(*args, **kwargs)
        
        
        with ProtectionEnvironment():
            res = func(*args, ctrl = control_qb, **kwargs)
        
        return res
        
    return adaptive_control_function


class ProtectionEnvironment(QuantumEnvironment):
    
    def compile(self):
        
        temp = list(self.env_qs.data)
        
        QuantumEnvironment.compile(self)
        
        for instr in self.env_qs.data:
            instr.op = ProtectedOperation(instr.op)
        
        self.env_qs.data = temp + self.env_qs.data


from qrisp.circuit import Operation
class ProtectedOperation(Operation):
    
    def __init__(self, base_op):
        
        self.base_op = base_op
        
        Operation.__init__(self, base_op.name, 
                           base_op.num_qubits, 
                           base_op.num_clbits, 
                           base_op.definition, 
                           base_op.params)
        
        self.permeability = base_op.permeability
        self.is_qfree = base_op.is_qfree
        
        self.manual_allocation_management = True
        
    def control(self, num_ctrl_qubits=1, ctrl_state=-1, method=None):
        temp = self.base_op.control(num_ctrl_qubits, ctrl_state, method)
        return ProtectedOperation(temp)
    
    def inverse(self):
        temp = self.base_op.inverse()
        return ProtectedOperation(temp)
    
    def get_unitary(self):
        return self.base_op.get_unitary()
    
    def unprotect(self):
        return self.base_op

    def copy(self):
        return ProtectedOperation(self.base_op.copy())

