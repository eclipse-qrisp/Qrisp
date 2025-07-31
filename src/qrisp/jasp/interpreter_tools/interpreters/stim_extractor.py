"""
********************************************************************************
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
********************************************************************************
"""

import numpy as np

from qrisp.jasp.interpreter_tools import extract_invalues, eval_jaxpr, insert_outvalues
from qrisp.jasp.primitives import OperationPrimitive, AbstractQubit, AbstractQubitArray, AbstractQuantumCircuit

stim_gate_translation = {"cx": "CX", "cz": "CZ", "cy" : "CY", "reset" : "RZ", "h": "H", "s" : "S", "s_dg": "S_DAG", "x" : "X", "y" : "Y", "z" : "Z"}

class StimMeasurement:
    def __init__(self, index, state):
        self.index = index
        self.state = state

class StimDetector:
    def __init__(self, index, state):
        self.index = index
        self.state = state


class StimState:
    
    def __init__(self):
        import stim
        self.circuit = stim.Circuit()
        self.measurement_amount = 0
        self.detector_amount = 0
        self.qubit_amount = 0

def stim_evaluator(eqn, context_dic):
    
    invalues = extract_invalues(eqn, context_dic)
    
    if isinstance(eqn.primitive, OperationPrimitive):
        op = eqn.primitive.op
        stim_circuit = invalues[-1].circuit
        qubit_indices = invalues[:-1]
        try:
            stim_gate = stim_gate_translation[op.name]
        except KeyError:
            raise Exception(f"Don't know how to translate instruction {op.name} to stim")
            
        stim_circuit.append(stim_gate, qubit_indices)
        outvalues = invalues[-1]
        
    elif eqn.primitive.name == "pjit":
        
        outvalues = eval_jaxpr(eqn.params["jaxpr"], 
                               eqn_evaluator = stim_evaluator)(*invalues)
        
    elif eqn.primitive.name == "jasp.measure":
        
        stim_state = invalues[-1]
        if isinstance(eqn.invars[0].aval, AbstractQubit):
            meas_res = StimMeasurement(stim_state.measurement_amount, stim_state)
            stim_state.measurement_amount += 1
            stim_state.circuit.append("MZ", invalues[0])
        elif isinstance(eqn.invars[0].aval, AbstractQubitArray):
            meas_res = []
            for i in range(len(invalues[0])):
                meas_res.append(StimMeasurement(stim_state.measurement_amount, stim_state))
                stim_state.measurement_amount += 1
                stim_state.circuit.append("MZ", invalues[0][i])
                
        outvalues = [meas_res, stim_state]
    elif eqn.primitive.name == "jasp.reset":
        
        stim_state = invalues[-1]
        if isinstance(eqn.invars[0].aval, AbstractQubit):
            stim_state.circuit.append("RZ", invalues[0])
        elif isinstance(eqn.invars[0].aval, AbstractQubitArray):
            for i in range(len(invalues[0])):
                stim_state.circuit.append("RZ", invalues[0][i])
                
        outvalues = stim_state
    
    elif eqn.primitive.name == "jasp.create_qubits":
        
        stim_state = invalues[-1]
        qubit_res = []
        for i in range(invalues[0]):
            qubit_res.append(stim_state.qubit_amount)
            stim_state.qubit_amount += 1
            
        outvalues = [qubit_res, stim_state]
        
    elif eqn.primitive.name == "jasp.delete_qubits":
        outvalues = invalues[-1]
    
    elif eqn.primitive.name == "convert_element_type":
        if isinstance(invalues[0], StimMeasurement):
            outvalues = invalues[0]
        elif isinstance(invalues[0], list) and len(invalues[0]) and isinstance(invalues[0][0], StimMeasurement):
            outvalues = invalues[0]
        else:
            return True
        
    elif eqn.primitive.name == "jasp.get_qubit":
        
        outvalues = invalues[0][invalues[1]]
        
    elif eqn.primitive.name == "eq":
        import stim
        if isinstance(invalues[0], StimMeasurement) and isinstance(invalues[1], StimMeasurement):
            state = invalues[0].state
            print(invalues[0].index-state.measurement_amount)
            print(invalues[1].index-state.measurement_amount)
            rec_0 = stim.target_rec(invalues[1].index-state.measurement_amount)
            rec_1 = stim.target_rec(invalues[0].index-state.measurement_amount)
            invalues[0].state.circuit.append("DETECTOR", [rec_0, rec_1])
            outvalues = StimDetector(invalues[0].state.detector_amount, invalues[0].state)
            invalues[0].state.detector_amount += 1
        else:
            return True
        
    elif len(eqn.invars) == 0 or not isinstance(eqn.invars[-1].aval, AbstractQuantumCircuit):
        return True
    
    elif eqn.primitive.name in ["while", "cond", "scan"]:
        return True
    else:
        raise Exception(f"Don't know how to process primitive {eqn.primitive}")
            
    insert_outvalues(eqn, context_dic, outvalues)


def extract_stim(fn):
    
    from qrisp.jasp import make_jaspr
    
    def return_function(*args):
        
        jaspr = make_jaspr(fn, garbage_collection = "none")(*args)
        
        args = list(args) + [StimState()]
        
        eval_res = eval_jaxpr(jaspr,
                              eqn_evaluator = stim_evaluator)(*args)
        
        if len(jaspr.outvars) == 1:
            return eval_res.circuit
        else:
            eval_res = list(eval_res)
            for i in range(len(eval_res)):
                if isinstance(eval_res[i], StimMeasurement):
                    eval_res[i] = eval_res[i].index
                elif isinstance(eval_res[i], list) and len(eval_res[i]):
                    for j in range(len(eval_res[i])):
                        if isinstance(eval_res[i][j], StimMeasurement):
                            eval_res[i][j] = eval_res[i][j].index
                        else:
                            break
                    else:
                        eval_res[i] = np.array(eval_res[i], dtype = np.int32)
            eval_res[-1] = eval_res[-1].circuit
            return tuple(eval_res)
        
    return return_function