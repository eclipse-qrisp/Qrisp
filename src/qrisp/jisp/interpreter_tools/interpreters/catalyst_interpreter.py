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

from jax import make_jaxpr

from catalyst.jax_primitives import AbstractQreg, qinst_p, qmeasure_p, qextract_p, qinsert_p

from qrisp.circuit import Operation
from qrisp.jisp import QuantumPrimitive, AbstractQuantumCircuit, AbstractQubitArray, AbstractQubit, eval_jaxpr


# Name translator from Qrisp gate naming to Catalyst gate naming
op_name_translation_dic = {"cx" : "CNOT",
                       "cy" : "CY", 
                       "cz" : "CZ", 
                       "crx" : "CRX",
                       "crz" : "CRZ",
                       "swap" : "SWAP",
                       "x" : "PauliX",
                       "y" : "PauliY",
                       "z" : "PauliZ",
                       "h" : "Hadamard",
                       "rx" : "RX",
                       "ry" : "RY",
                       "rz" : "RZ",
                       "s" : "S",
                       "t" : "T",
                       "p" : "Phasegate"}


def jispr_to_catalyst_jaxpr(jispr):
    
    args = []
    for invar in jispr.invars:
        if isinstance(invar.aval, AbstractQuantumCircuit):
            args.append((AbstractQreg(), 0))
        elif isinstance(invar.aval, AbstractQubitArray):
            args.append((0,0))
        elif isinstance(invar.aval, AbstractQubit):
            args.append(0)
        else:
            args.append(invar.aval)
        
    return make_jaxpr(eval_jaxpr(jispr, eqn_evaluator = catalyst_eqn_evaluator))(*args)
        
    

def catalyst_eqn_evaluator(eqn, context_dic):
    
    if isinstance(eqn.primitive, QuantumPrimitive):
        
        invars = eqn.invars
        outvars = eqn.outvars
        
        if eqn.primitive.name == "create_qubits":
            process_create_qubits(invars, outvars, context_dic)
        elif eqn.primitive.name == "get_qubit":
            process_get_qubit(invars, outvars, context_dic)
        elif eqn.primitive.name == "measure":
            process_measurement(invars, outvars, context_dic)
        elif isinstance(eqn.primitive, Operation):
            process_op(eqn.primitive, invars, outvars, context_dic)
        else:
            print(type(eqn.primitive))
            raise Exception(f"Don't know how to process QuantumPrimitive {eqn.primitive}")
    else:
        return True
            


def process_create_qubits(invars, outvars, context_dic):
    
    qreg, stack_size = context_dic[invars[0]]
    context_dic[outvars[1]] = (stack_size, context_dic[invars[1]])
    context_dic[outvars[0]] = (qreg, stack_size + context_dic[invars[1]])
    

def process_get_qubit(invars, outvars, context_dic):
    context_dic[outvars[0]] = context_dic[invars[0]][0] + context_dic[invars[1]]
    
def process_op(op, invars, outvars, context_dic):
    
    # This case is applies a quantum operation
    catalyst_register_tracer = context_dic[invars[0]][0]
    
    # For this the first step is to collect all the Catalyst qubit tracers
    # that are required for the Operation
    qb_vars = []
    
    qb_pos = []
    for i in range(op.num_qubits):
        qb_vars.append(invars[i+1+len(op.params)])
        qb_pos.append(context_dic[invars[i+1+len(op.params)]])
    
    num_qubits = len(qb_pos)
    
    catalyst_qb_tracers = []
    for i in range(num_qubits):
        catalyst_qb_tracer = qextract_p.bind(catalyst_register_tracer, 
                                             qb_pos[i])
        catalyst_qb_tracers.append(catalyst_qb_tracer)
    
    # We can now apply the gate primitive
    res_qbs = exec_qrisp_op(op, catalyst_qb_tracers)
        
    
    # Finally, we reinsert the qubits and update the register tracer
    for i in range(num_qubits):
        catalyst_register_tracer = qinsert_p.bind(catalyst_register_tracer, 
                                             qb_pos[i],
                                             res_qbs[i])
        
    context_dic[outvars[0]] = (catalyst_register_tracer, context_dic[invars[0]][1])

def exec_qrisp_op(op, catalyst_qbs):
    
    if op.definition:
        defn = op.definition
        for instr in defn.data:
            qubits = instr.qubits
            qubit_indices = [defn.qubits.index(qb) for qb in qubits]
            temp_catalyst_qbs = [catalyst_qbs[i] for i in qubit_indices]
            res_qbs = exec_qrisp_op(instr.op, temp_catalyst_qbs)
            
            for i in range(len(qubit_indices)):
                catalyst_qbs[qubit_indices[i]] = res_qbs[i]
                
        return catalyst_qbs
    
    else:
        res_qbs = qinst_p.bind(*catalyst_qbs, 
                               op = op_name_translation_dic[op.name], 
                               qubits_len = op.num_qubits)
        return res_qbs


def process_measurement(invars, outvars, context_dic):
    
    catalyst_register_tracer = context_dic[invars[0]][0]
                            
    if isinstance(invars[1].aval, AbstractQubitArray):
        
        qubit_array_data = context_dic[invars[1]]
        start = qubit_array_data[0]
        stop = start + qubit_array_data[1]
        catalyst_register_tracer, meas_res = exec_multi_measurement(catalyst_register_tracer, start, stop)
        
    else:
        qb_pos = context_dic[invars[1]]
        catalyst_qb_tracer = qextract_p.bind(catalyst_register_tracer, 
                                             qb_pos)
        meas_res, res_qb = qmeasure_p.bind(catalyst_qb_tracer)
        catalyst_register_tracer = qinsert_p.bind(catalyst_register_tracer, 
                                             qb_pos,
                                             res_qb)
        
    context_dic[outvars[0]] = (catalyst_register_tracer, context_dic[invars[0]][1])
    context_dic[outvars[1]] = meas_res
        
        
        
def exec_multi_measurement(catalyst_register, start, stop):
    
    from catalyst.jax_primitives import qmeasure_p, qextract_p, qinsert_p
    from jax.lax import fori_loop
    
    def loop_body(i, val):
        acc = val[1]
        reg = val[0]
        qb = qextract_p.bind(reg, i)
        res_bl, res_qb = qmeasure_p.bind(qb)
        reg = qinsert_p.bind(reg, i, res_qb)
        acc = acc + (2<<i)*res_bl
        return (reg, acc)
    
    return fori_loop(start, stop, loop_body, (catalyst_register, 0))
