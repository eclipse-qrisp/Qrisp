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

from qrisp.jasp.primitives import QuantumPrimitive, AbstractQuantumCircuit

from sympy import symbols

greek_letters = symbols('alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega')

class OperationPrimitive(QuantumPrimitive):
    
    def __init__(self, op):
        
        self.op = op
        
        name = op.name

        if len(op.params):
            name += "("
            
            for param in op.params[:-1]:
                name += str(param) + ", "
            
            name += str(op.params[-1]) 	+ ")"
            
        QuantumPrimitive.__init__(self, name)
        
        @self.def_abstract_eval
        def abstract_eval(qc, *args):
            """Abstract evaluation of the primitive.
            
            This function does not need to be JAX traceable. It will be invoked with
            abstractions of the actual arguments. 
            """
            if not isinstance(qc, AbstractQuantumCircuit):
                raise Exception(f"Tried to execute OperationPrimitive.bind with the first argument of tpye {type(qc)} instead of AbstractQuantumCircuit")
            
            return AbstractQuantumCircuit()
        
        @self.def_impl
        def append_impl(qc, *args):
            """Concrete evaluation of the primitive.
            
            This function does not need to be JAX traceable. It will be invoked with
            actual instances. 
            """
            parameter_args = args[:-self.op.num_qubits]
            qubit_args = args[-self.op.num_qubits:]
            
            temp_op = self.op.bind_parameters({greek_letters[i] : float(parameter_args[i]) for i in range(len(parameter_args))})
            qc.append(temp_op, list(qubit_args))
            return qc
    
    def inverse(self):
        return OperationPrimitive(self.op.inverse())
    
    def control(self, num_ctrl = 1, ctrl_state = -1):
        return OperationPrimitive(self.op.control(num_ctrl, ctrl_state = -1))
        


        
