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

from qrisp.jisp.primitives import QuantumPrimitive
from qrisp.jisp.interpreter_tools import eval_jaxpr, extract_invalues, insert_outvalues

def pjit_to_gate(pjit_primitive, *args, **kwargs):
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
    
    if len(definition_jaxpr.outvars) == 1:
        res = [res]
    
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
        old_qc.append(new_qc.to_op(name = kwargs["name"]), new_qc.qubits, new_qc.clbits)
        
        res = list(res)
        res[0] = old_qc
    
    return res
      

