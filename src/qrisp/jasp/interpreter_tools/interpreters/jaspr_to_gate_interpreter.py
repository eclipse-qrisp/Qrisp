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

from qrisp.jasp.interpreter_tools import eval_jaxpr, extract_invalues, insert_outvalues

def pjit_to_gate(pjit_eqn, context_dic):
    """
    Wraps the content of a pjit primitive into a gate and appends the gate.

    Parameters
    ----------
    pjit_eqn : An equation with a pjit primitive
        The equation to execute.

    Returns
    -------
    return values
        The values/tracers of the pjit execution.

    """
    
    # Set alias for the function definition
    definition_jaxpr = pjit_eqn.params["jaxpr"]
    
    # Extract the invalues from the context dic
    invalues = extract_invalues(pjit_eqn, context_dic)
    
    from qrisp.circuit import QuantumCircuit

    if len(invalues) == 0 or not isinstance(invalues[0], QuantumCircuit):
        return True
    
    res = eval_qc(definition_jaxpr, invalues)
    
    new_qc = res[0]
    old_qc = invalues[0]
    
    # Append the wrapped old circuit to the new circuit
    old_qc.append(new_qc.to_op(name = pjit_eqn.params["name"]), new_qc.qubits, new_qc.clbits)
    res = list(res)
    res[0] = old_qc
      
    # Insert the result into the context dic
    insert_outvalues(pjit_eqn, context_dic, res)
    
    
def cond_to_cl_control(eqn, context_dic):
    from qrisp.circuit import QuantumCircuit, Clbit
    
    # Extract the invalues from the context dic
    invalues = extract_invalues(eqn, context_dic)

    if len(invalues) == 0:
        return True
    
    if not isinstance(invalues[0], Clbit):
        return True
    
    if not isinstance(invalues[1], QuantumCircuit):
        raise Exception

    false_jaxpr = eqn.params["branches"][0]
    true_jaxpr = eqn.params["branches"][1]
    
    false_qc = eval_qc(false_jaxpr, invalues[1:])[0]
    true_qc = eval_qc(true_jaxpr, invalues[1:])[0]
    
    old_qc = invalues[1]
    
    if len(false_qc.data):
        old_qc.append(false_qc.to_op().c_if(ctrl_state = 0), false_qc.qubits, [invalues[0]] + false_qc.clbits)
    if len(true_qc.data):
        old_qc.append(true_qc.to_op().c_if(ctrl_state = 1), true_qc.qubits, [invalues[0]] + true_qc.clbits)
    
    insert_outvalues(eqn, context_dic, [old_qc])
    

def eval_qc(definition_jaxpr, invalues):
    
    old_qc = invalues[0]
    new_qc = old_qc.clearcopy()
    invalues = list(invalues)
    invalues[0] = new_qc

    # Evaluate the definition
    def eqn_evaluator(eqn, context_dic):
        if eqn.primitive.name == "pjit":
            pjit_to_gate(eqn, context_dic)
        else:
            return True

    res = eval_jaxpr(definition_jaxpr.jaxpr, eqn_evaluator = eqn_evaluator)(*(invalues + definition_jaxpr.consts))
    
    if len(definition_jaxpr.jaxpr.outvars) == 1:
        res = [res]
    
    # Add new qubits/clbits to the circuit        
    for qb in set(new_qc.qubits) - set(old_qc.qubits):
        old_qc.add_qubit(qb)

    for cb in set(new_qc.clbits) - set(old_qc.clbits):
        old_qc.add_clbit(cb)
    
    
    # Remove unused qu/clbits from the new circuit
    unused_qubits = set(new_qc.qubits)
    unused_clbits = set(new_qc.clbits)
    
    for instr in new_qc.data:
        unused_qubits -= set(instr.qubits)
        unused_clbits -= set(instr.clbits)
    
    for qb in unused_qubits:
        new_qc.qubits.remove(qb)
    for cb in unused_clbits:
        new_qc.clbits.remove(cb)
    
    res = list(res)
    res[0] = new_qc
    return res


