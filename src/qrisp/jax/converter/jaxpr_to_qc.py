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

from jax.core import Literal, ClosedJaxpr
from qrisp.jax.primitives import QuantumPrimitive
from qrisp.jax.flattening_tools import eval_jaxpr

def jaxpr_to_qc(jaxpr):
    """
    Converts a Qrisp-generated Jaxpr into a QuantumCircuit.

    Parameters
    ----------
    jaxpr : jax.core.Jaxpr
        The Jaxpr to be converted.
    in_place : TYPE, optional
        If set to False, the AbstractCircuit is copied and the copy is continued to be processed. The default is True.

    Returns
    -------
    qc : QuantumCircuit
        The converted circuit.

    """

    if isinstance(jaxpr, ClosedJaxpr):
        jaxpr = jaxpr.jaxpr
    
    def qc_eval_function(*args):
        
        if len(jaxpr.invars) != len(args):
            raise Exception(f"Supplied inaccurate amount of arguments ({len(args)}) for Jaxpr (requires {len(jaxpr.invars)}).")
        
        eqn_evaluator_function_dic = {"pjit" : pjit_to_gate}
        
        outvals = eval_jaxpr(jaxpr, 
                             return_context_dic = True, 
                             eqn_evaluator_function_dic = eqn_evaluator_function_dic)(*args)
        
        context_dic = outvals[0]
        
        
        from qrisp.circuit import QuantumCircuit
        for val in context_dic.values():
            if isinstance(val, QuantumCircuit):
                return tuple([val] + list(outvals)[1:])
            
        raise Exception("Could not find QuantumCircuit in Jaxpr")
        
    return qc_eval_function

def pjit_to_gate(pjit_eqn, **kwargs):
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
    definition_jaxpr = kwargs["jaxpr"].jaxpr
    
    def jaxpr_evaluator(*args):
        
        from qrisp.circuit import QuantumCircuit
        # Create new context dic and fill with invalues
        
        # Exchange the QuantumCircuit to an empty one to "track" the function
        if isinstance(args[0], QuantumCircuit):
            old_qc = args[0]
            new_qc = old_qc.clearcopy()
            args = list(args)
            args[0] = new_qc

        # Evaluate the definition
        res = eval_jaxpr(definition_jaxpr)(*args)
        
        if isinstance(args[0], QuantumCircuit):
            
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

        return tuple(res)
    
    return jaxpr_evaluator

