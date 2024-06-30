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

from jax.core import Literal
from qrisp.jax import QuantumPrimitive, flatten_pjit

# 

def jaxpr_to_qc(jaxpr, in_place = True):
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
    
    # Inline Jaxpr to dissolve any pjit calls
    flattened_jaxpr = flatten_pjit(jaxpr)
    from qrisp.circuit import QuantumCircuit, Operation
    
    # The context dic to translate variables to values
    context_dic = {}
    
    # Iterate over the equations
    for eqn in flattened_jaxpr.eqns:
        
        # Set alias for in/outvars
        invars = list(eqn.invars)
        outvars = list(eqn.outvars)
        
        # If the primitive is not a Qrisp primitive, we simply execute the .bind method
        # to call the default implementation
        if not isinstance(eqn.primitive, QuantumPrimitive):
            eqn.primitive.bind(*[context_dic[var] for var in invars], **eqn.params)
        
        # Treat the Qrisp primitive
        else:
            
            # Circuit creation case
            if eqn.primitive.name == "qdef":
                context_dic[outvars[0]] = QuantumCircuit()
                
            # Qubit creation will return a QuantumCircuit and a list of Qubit
            elif eqn.primitive.name == "create_qubits":
                
                # This will contain the list of qubits
                qubit_list = []
                
                # Get the QuantumCircuit
                qc = context_dic[invars[0]]
                
                # If necessary, copy
                if not in_place:
                    qc = qc.copy()
                
                # This variable contains the information how many Qubits should be created
                amount_var = invars[1]
                
                # Extract value
                if isinstance(amount_var, Literal):
                    amount_val = amount_var.val
                else:
                    amount_val = context_dic[amount_var]
                
                for i in range(int(amount_val)):
                    qubit_list.append(qc.add_qubit())
                
                # Assign the new values to their variables
                context_dic[eqn.outvars[0]] = qc
                context_dic[eqn.outvars[1]] = qubit_list
            
            # Retrieves a qubit
            elif eqn.primitive.name == "get_qubit":
                
                index_var = invars[1]
                if isinstance(index_var, Literal):
                    index_val = index_var.val
                else:
                    index_val = context_dic[index_var]
                
                context_dic[outvars[0]] = context_dic[invars[0]][index_val]
                
            elif eqn.primitive.name == "measure":
                qc = context_dic[invars[0]]
                
                if not in_place:
                    qc = qc.copy()
                
                context_dic[outvars[1]] = qc.add_clbit()
                
                qc.measure(context_dic[invars[1]], context_dic[outvars[1]])
                
                context_dic[outvars[0]] = qc            
                
            
            elif isinstance(eqn.primitive, Operation):
                qc = context_dic[invars[0]]
                
                if not in_place:
                    qc = qc.copy()
                
                qc.append(eqn.primitive, [context_dic[qb_var] for qb_var in invars[1:]])
                context_dic[outvars[0]] = qc


    return qc