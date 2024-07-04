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
from jax.core import Literal, ClosedJaxpr
from qrisp.jax.primitives import QuantumPrimitive
from qrisp.jax.flattening_tools import eval_jaxpr, extract_invalues, insert_outvalues

def extract_qc(jaxpr_or_function):
    """
    Converts a Qrisp-generated Jaxpr into a QuantumCircuit.

    Parameters
    ----------
    jaxpr : jax.core.Jaxpr
        The Jaxpr to be converted.
    
    Returns
    -------
    qc : QuantumCircuit
        The converted circuit.
    func_res : tuple
        A tuple of objects representing the result of the function.
    

    """
        
    if callable(jaxpr_or_function):
        jaxpr_gen = lambda *args : make_jaxpr(jaxpr_or_function)(*args).jaxpr
    else:
        if isinstance(jaxpr_or_function, ClosedJaxpr):
            jaxpr_or_function = jaxpr_or_function.jaxpr
        
        jaxpr_gen = lambda *args : jaxpr_or_function
    
    def qc_eval_function(*args):
        
        jaxpr = jaxpr_gen(*args)
        
        if len(jaxpr.invars) != len(args):
            raise Exception(f"Supplied inaccurate amount of arguments ({len(args)}) for Jaxpr (requires {len(jaxpr.invars)}).")
        
        eqn_evaluator_function_dic = {"pjit" : pjit_to_gate, "measure" : measure_to_clbit}
        
        outvals = eval_jaxpr(jaxpr, 
                             return_context_dic = True, 
                             eqn_evaluator_function_dic = eqn_evaluator_function_dic)(*args)
        
        if len(jaxpr.outvars) == 0:
            outvals = [outvals]
            
        context_dic = outvals[0]
        
        
        from qrisp.circuit import QuantumCircuit
        for val in context_dic.values():
            if isinstance(val, QuantumCircuit):
                if len(jaxpr.outvars) == 1:
                    return val, outvals[1]
                else:
                    return val, tuple(outvals)[1:]
            
        raise Exception("Could not find QuantumCircuit in Jaxpr")
        
    return qc_eval_function

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
    definition_jaxpr = pjit_eqn.params["jaxpr"].jaxpr
    
    # Extract the invalues from the context dic
    invalues = extract_invalues(pjit_eqn, context_dic)
        
    from qrisp.circuit import QuantumCircuit
    # Create new context dic and fill with invalues
    
    # Exchange the QuantumCircuit to an empty one to "track" the function
    if isinstance(invalues[0], QuantumCircuit):
        old_qc = invalues[0]
        new_qc = old_qc.clearcopy()
        invalues = list(invalues)
        invalues[0] = new_qc

    # Evaluate the definition
    res = eval_jaxpr(definition_jaxpr)(*invalues)
    
    if len(definition_jaxpr.outvars) == 1:
        res = [res]
    
    if isinstance(invalues[0], QuantumCircuit):
        
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
        
        # Append the wrapped old circuit to the new circuit
        old_qc.append(new_qc.to_op(name = pjit_eqn.params["name"]), new_qc.qubits, new_qc.clbits)
        
        res = list(res)
        res[0] = old_qc
      
    # Insert the result into the context dic
    insert_outvalues(pjit_eqn, context_dic, res)


def measure_to_clbit(measure_eqn, context_dic):
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
    
    qc = context_dic[measure_eqn.invars[0]]
    meas_qubit = context_dic[measure_eqn.invars[1]]
    
    clbit = qc.add_clbit()
    
    qc.measure(meas_qubit, clbit)
    
    insert_outvalues(measure_eqn, context_dic, ["burned_qc", clbit])
    
