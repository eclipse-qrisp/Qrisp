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

from jax.core import JaxprEqn, Literal
from qrisp.jax.flattening_tools import evaluate_eqn

def evaluate_qcall(eqn, context_dic):
    
    # Evaluate the Variable representing the QuantumCircuit to be processed
    qc_to_be_extended = eqn.invars[0]
    if isinstance(context_dic[qc_to_be_extended], JaxprEqn):
        
        sub_eq = context_dic[qc_to_be_extended]
        res = evaluate_eqn(sub_eq, context_dic)
        
        # Update the context dict
        if sub_eq.primitive.multiple_results:
            for i in range(len(sub_eq.outvars)):
                context_dic[sub_eq.outvars[i]] = res[i]
        else:
            context_dic[sub_eq.outvars[0]] = res


    # Now the Variable representing the QuantumCircuit to extend with
    
    # First, we need to find the definition equation.
    definition_eq = context_dic[eqn.invars[1]]
    while definition_eq.primitive.name != "qdef":
        definition_eq = context_dic[definition_eq.invars[0]]
    
    # Create the new context dictionary
    # This context will have the arguments of qcall available
    # for the Variables of qdef
    new_context_dic = dict(context_dic)
    
    replacement_vars = [qc_to_be_extended] + eqn.invars[2:]
    for i in range(len(definition_eq.outvars)):
        new_context_dic[definition_eq.outvars[i]] = context_dic[replacement_vars[i]]
    
    
    # This is the last equation of the function definition
    concluding_equation = context_dic[eqn.invars[1]]
    return evaluate_eqn(concluding_equation, new_context_dic)

