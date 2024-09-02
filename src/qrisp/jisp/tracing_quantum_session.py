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

import jax

from qrisp.jisp import create_qubits, delete_qubits_p


class TracingQuantumSession:
    tr_qs_container = [None]
    
    def __init__(self, abs_qc):
        
        self.abs_qc = abs_qc
        self.qv_list = []
        self.deleted_qv_list = []
        self.qubit_cache = {}
        TracingQuantumSession.tr_qs_container = [self]
        
    def append(self, operation, qubits = [], clbits = []):
    
        if len(clbits):
            raise Exception("Tried to append Operation with non-zero classical bits in JAX mode.")
            
        if self.abs_qc is None:
            raise
            # self.abs_qc = qdef_p.bind()
            
        self.abs_qc = operation.bind(self.abs_qc, *(operation.params + [b for b in qubits]))
        
    def register_qv(self, qv, size):
        if qv.name in [temp_qv.name for temp_qv in self.qv_list + self.deleted_qv_list]:
            raise RuntimeError(
                "Variable name " + str(qv.name) + " already exists in quantum session"
            )
            
        # Determine amount of required qubits
        self.abs_qc, qv.reg = create_qubits(self.abs_qc, size)
        # Register in the list of active quantum variable
        self.qv_list.append(qv)
        
    def delete_qv(self, qv, verify=False):
        
        if verify == True:
            raise Exception("Tried to verify deletion in tracing mode.")
        
        # Check if quantum variable appears in this session
        if qv.name not in [qv.name for qv in self.qv_list]:
            raise Exception(
                "Tried to remove a non existent quantum variable from quantum session"
            )

        self.clear_qubits(qv.reg, verify)

        # Remove quantum variable from list
        for i in range(len(self.qv_list)):
            temp_qv = self.qv_list[i]

            if temp_qv.name == qv.name:
                self.qv_list.pop(i)
                break

        self.deleted_qv_list.append(qv)
        
    def clear_qubits(self, qubits, verify=False):
        
        self.abs_qc = delete_qubits_p.bind(self.abs_qc, qubits)
    
    @classmethod
    def release(cls):
        cls.tr_qs_container = [None]
    
    @classmethod
    def get_instance(cls):
        return cls.tr_qs_container[0]
        
        

def check_for_tracing_mode():
    return hasattr(jax._src.core.thread_local_state.trace_state.trace_stack.dynamic, "jaxpr_stack")

def check_live(tracer):
    if tracer is None:
        return True
    return not not tracer._trace.main.jaxpr_stack