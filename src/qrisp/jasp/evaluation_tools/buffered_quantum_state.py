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

from qrisp.circuit import QuantumCircuit, XGate
from qrisp.simulator import QuantumState, advance_quantum_state, gen_res_dict

class BufferedQuantumState:
    
    def __init__(self, simulator = "qrisp"):
        
        if simulator == "qrisp":
            self.quantum_state = QuantumState(n = 0)
        elif simulator == "stim":
            import stim
            self.quantum_state = stim.TableauSimulator()
        else:
            raise Exception("Don't know simulator {simulator}")
        self.buffer_qc = QuantumCircuit(0)
        self.deallocated_qubits = []
        self.simulator = simulator
    
    def add_qubit(self):
        if self.simulator == "qrisp":
            self.quantum_state.add_qubit()
            
        return self.buffer_qc.add_qubit()
    
    def append(self, op, qubits):
        self.buffer_qc.append(op, qubits)
            
    def apply_buffer(self):
        if self.simulator == "qrisp":
            self.quantum_state = advance_quantum_state(self.buffer_qc, self.quantum_state, self.deallocated_qubits)
        else:
            for instr in self.buffer_qc.data:
                qubit_indices = [self.buffer_qc.qubits.index(qb) for qb in instr.qubits]
                
                if instr.op.name == "x":
                    self.quantum_state.x(*qubit_indices)
                elif instr.op.name == "y":
                    self.quantum_state.y(*qubit_indices)
                elif instr.op.name == "z":
                    self.quantum_state.z(*qubit_indices)
                elif instr.op.name == "h":
                    self.quantum_state.h(*qubit_indices)
                elif instr.op.name == "cx":
                    self.quantum_state.cx(*qubit_indices)
                elif instr.op.name == "cy":
                    self.quantum_state.cy(*qubit_indices)
                elif instr.op.name == "cz":
                    self.quantum_state.cz(*qubit_indices)
                elif instr.op.name == "s":
                    self.quantum_state.s(*qubit_indices)
                elif instr.op.name == "s_dg":
                    self.quantum_state.s_dag(*qubit_indices)
                elif not instr.op.name in ["qb_alloc", "qb_dealloc"]:
                    raise Exception(f"Don't know how to simulate quantum gate {instr.op.name} with stim")
                
        self.buffer_qc = self.buffer_qc.clearcopy()
    
    def measure(self, qubit):
        self.apply_buffer()
        if self.simulator == "qrisp":
            mes_res = self.quantum_state.measure(self.buffer_qc.qubits.index(qubit), keep_res = True)
            rnd = np.random.random(1)
            if rnd < mes_res[0]:
                self.quantum_state = mes_res[1]
                return False
            else:
                self.quantum_state = mes_res[3]
                return True
        elif self.simulator == "stim":
            return self.quantum_state.measure(self.buffer_qc.qubits.index(qubit))
    
    def reset(self, qubit):
        meas_res = self.measure(qubit)
        if meas_res:
            self.buffer_qc.append(XGate(), qubit)
            
    def copy(self):
        res = BufferedQuantumState()
        res.buffer_qc = self.buffer_qc.copy()
        res.deallocated_qubits = list(self.deallocated_qubits)
        res.quantum_state = self.quantum_state.copy()
        return res
    
    def multi_measure(self, qubits, shots):
        self.apply_buffer()
        qubit_indices = [self.buffer_qc.qubits.index(qb) for qb in qubits]
        mes_ints, probs = self.quantum_state.multi_measure(qubit_indices)
        
        if shots is not None and shots != 0:
            samples = np.random.choice(len(mes_ints), int(shots), p=probs)
            
            samples = gen_res_dict(samples)
            res = {}
            for k, v in samples.items():
                res[mes_ints[k]] = v
            return res
        else:
            res = {}
            for i in range(len(mes_ints)):
                res[mes_ints[i]] = probs[i]
            return res
