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

from jax.core import JaxprEqn, Literal, ClosedJaxpr
from jax import jit, make_jaxpr
from qrisp.jax.flattening_tools import eval_jaxpr, extract_invalues, insert_outvalues

def evaluate_pjit_eqn(pjit_eqn, context_dic):
    
    definition_jaxpr = pjit_eqn.params["jaxpr"].jaxpr
    
    # Extract the invalues from the context dic
    invalues = extract_invalues(pjit_eqn, context_dic)
        
    res = jit(eval_jaxpr(definition_jaxpr), inline = True)(*invalues)

    # Insert the values into the context_dic
    insert_outvalues(pjit_eqn, context_dic, res)
                
# Flattens/Inlines a pjit calls in a jaxpr
def flatten_pjit(jaxpr):
    
    if isinstance(jaxpr, ClosedJaxpr):
        jaxpr = jaxpr.jaxpr
    
    eqn_evaluator_function_dic = {"pjit" : evaluate_pjit_eqn}
    return make_jaxpr(eval_jaxpr(jaxpr, 
                                 eqn_evaluator_function_dic = eqn_evaluator_function_dic))(*[var.aval for var in jaxpr.invars])
    