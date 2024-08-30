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
from jax.lax import fori_loop

from catalyst.jax_primitives import AbstractQreg, qinst_p, qmeasure_p, qextract_p, qinsert_p, while_p

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
                       "t_dg" : "T_dg",
                       "s_dg" : "S_dg",
                       "p" : "Phasegate"}


def catalyst_eqn_evaluator(eqn, context_dic):
    """
    This function serves as the central interface to interpret Jisp equations 
    into Catalyst primitives.

    Parameters
    ----------
    eqn : jax.core.JaxprEqn
        The equation to be interpreted.
    context_dic : qrisp.jisp.interpreters.ContextDict
        The ContextDict representing the current state of the program by 
        translating from variables to values.

    Raises
    ------
    Exception
        Don't know how to interpret QuantumPrimitive.

    Returns
    -------
    bool
        Like other equation processing managers, if this function returns True,
        it implies that the equation should be processed by simply binding the 
        primitive.

    """
    
    # If the equations primitive is a Qrisp primitive, we process it according
    # to one of the implementations below. Otherwise we return True to indicate
    # default interpretation.
    if isinstance(eqn.primitive, QuantumPrimitive):
        
        invars = eqn.invars
        outvars = eqn.outvars
        
        if eqn.primitive.name == "create_qubits":
            process_create_qubits(invars, outvars, context_dic)
        elif eqn.primitive.name == "get_qubit":
            process_get_qubit(invars, outvars, context_dic)
        elif eqn.primitive.name == "measure":
            process_measurement(invars, outvars, context_dic)
        elif eqn.primitive.name == "get_size":
            process_get_size(invars, outvars, context_dic)
        elif eqn.primitive.name == "delete_qubits":
            pass
        elif isinstance(eqn.primitive, Operation):
            process_op(eqn.primitive, invars, outvars, context_dic)
        else:
            print(type(eqn.primitive))
            raise Exception(f"Don't know how to process QuantumPrimitive {eqn.primitive}")
    else:
        return True
    


def process_create_qubits(invars, outvars, context_dic):
    
    # The first invar of the create_qubits primitive is an AbstractQuantumCircuit
    # which is represented by an AbstractQreg and an integer
    qreg, stack_size = context_dic[invars[0]]
    
    # We create the new QubitArray representation by putting the appropriate tuple
    # in the context_dic
    context_dic[outvars[1]] = (stack_size, context_dic[invars[1]])
    
    # Furthermore we create the updated AbstractQuantumCircuit representation.
    # The new stack size is the old stask size + the size of the QubitArray
    context_dic[outvars[0]] = (qreg, stack_size + context_dic[invars[1]])
    

def process_get_qubit(invars, outvars, context_dic):
    # The get_qubit primitive needs to retrieve the integer that indexes the 
    # AbstractQubit in the stack.
    # For that we add the Qubit index (in the QubitArray) to the QubitArray 
    # starting index.
    
    qubit_array_starting_index = context_dic[invars[0]][0]
    qubit_index = context_dic[invars[1]]
    context_dic[outvars[0]] = qubit_array_starting_index + qubit_index
    
def process_get_size(invars, outvars, context_dic):
    # The size is simply the second entry of the QubitArray representation
    context_dic[outvars[0]] = context_dic[invars[0]][1]
    
    
def process_op(op, invars, outvars, context_dic):
    
    # Alias the AbstractQreg
    catalyst_register_tracer = context_dic[invars[0]][0]
    
    # We first collect all the Catalyst qubit tracers
    # that are required for the Operation
    qb_vars = []
    
    # For that we find all the integers (ie. positions) of the participating
    # qubits in the AbstractQreg
    qb_pos = []
    for i in range(op.num_qubits):
        qb_vars.append(invars[i+1+len(op.params)])
        qb_pos.append(context_dic[invars[i+1+len(op.params)]])
    
    num_qubits = len(qb_pos)
    
    # Extract the catalyst qubit tracers by using the qextract primitive.
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
    # This function takes a Qrisp operation and a list of catalyst qubit tracers
    # and applies the operation to these qubits.
    
    # If the Operation has a definition, we call this function recursively.
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
    
    # Otherwise we simply call the bind method
    else:
        
        res_qbs = qinst_p.bind(*catalyst_qbs, 
                               op = op_name_translation_dic[op.name], 
                               qubits_len = op.num_qubits)
        return res_qbs


def process_measurement(invars, outvars, context_dic):
    # This function treats measurement primitives
    
    # Retrieve the catalyst register tracer
    catalyst_register_tracer = context_dic[invars[0]][0]
    
    # We differentiate between the cases that the measurement is performed on a 
    # singular Qubit (returns a bool) or a QubitArray (returns an integer)
    
    # This case treats the QubitArray situation
    if isinstance(invars[1].aval, AbstractQubitArray):
        
        # Retrieve the start and the endpoint indices of the QubitArray
        qubit_array_data = context_dic[invars[1]]
        start = qubit_array_data[0]
        stop = start + qubit_array_data[1]
        
        # The multi measurement logic is outsourced into a dedicated function
        catalyst_register_tracer, meas_res = exec_multi_measurement(catalyst_register_tracer, start, stop)
    
    # The singular Qubit case
    else:
        
        # Get the position of the Qubit
        qb_pos = context_dic[invars[1]]
        
        # Get the catalyst Qubit tracer
        catalyst_qb_tracer = qextract_p.bind(catalyst_register_tracer, 
                                             qb_pos)
        
        # Call the measurement primitive
        meas_res, res_qb = qmeasure_p.bind(catalyst_qb_tracer)
        
        # Reinsert the Qubit
        catalyst_register_tracer = qinsert_p.bind(catalyst_register_tracer, 
                                             qb_pos,
                                             res_qb)
    
    # Insert the result values into the context dict
    context_dic[outvars[0]] = (catalyst_register_tracer, context_dic[invars[0]][1])
    context_dic[outvars[1]] = meas_res
        
        
def exec_multi_measurement(catalyst_register, start, stop):
    # This function performs the measurement of multiple qubits at once, returning
    # an integer. The qubits to be measured sit in one consecutive interval,
    # starting at index "start" and ending at "stop".
    
    # We realize this feature using the Catalyst while loop primitive.
    
    # For that we need to define the loop body and the condition for breaking the
    # loop
    
    def loop_body(i, acc, reg, start, stop):
        qb = qextract_p.bind(reg, i)
        res_bl, res_qb = qmeasure_p.bind(qb)
        reg = qinsert_p.bind(reg, i, res_qb)
        acc = acc + (1<<(i-start))*res_bl
        i += 1
        return i, acc, reg, start, stop
    
    def cond_body(i, acc, reg, start, stop):
        return i < stop
    
    # Turn them into jaxpr
    loop_jaxpr = make_jaxpr(loop_body)(0, 0, catalyst_register.aval, 0, 0)
    cond_jaxpr = make_jaxpr(cond_body)(0, 0, catalyst_register.aval, 0, 0)

    # Bind the while loop primitive
    i, acc, reg, start, stop = while_p.bind(*(start, 0, catalyst_register, start, stop),
                                     cond_jaxpr = cond_jaxpr, 
                                     body_jaxpr = loop_jaxpr,
                                     cond_nconsts = 0,
                                     body_nconsts = 0,
                                     nimplicit = 0,
                                     preserve_dimensions = True,
                                     ) 
    
    return reg, acc
