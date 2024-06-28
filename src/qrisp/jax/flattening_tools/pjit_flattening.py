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
from jax import jit, make_jaxpr

def eval_jaxpr(jaxpr, context_dic = {}):
    
    for eqn in jaxpr.eqns:
        
        invalues = []
        
        for i in range(len(eqn.invars)):
            
            invar = eqn.invars[i]
            
            if isinstance(invar, Literal):
                invalues.append(invar.val)
                continue
            invalues.append(context_dic[invar])
        
        if eqn.primitive.name == "pjit":
            res = evaluate_pjit_eqn(eqn, context_dic)
        else:
            res = eqn.primitive.bind(*invalues, **eqn.params)
            
        if eqn.primitive.multiple_results:
            for i in range(len(eqn.outvars)):
                context_dic[eqn.outvars[i]] = res[i]
        else:
            context_dic[eqn.outvars[0]] = res


def evaluate_pjit_eqn(pjit_eqn, context_dic):
    
    definition_jaxpr = pjit_eqn.params["jaxpr"].jaxpr
    new_context_dic = dict(context_dic)
    for i in range(len(definition_jaxpr.invars)):
        definition_invar = definition_jaxpr.invars[i]
        invar = pjit_eqn.invars[i]
        
        if isinstance(invar, Literal):
            new_context_dic[definition_invar] = invar.val
        else:
            new_context_dic[definition_invar] = context_dic[invar]
            
    invalues = []
    
    for i in range(len(pjit_eqn.invars)):
        
        invar = pjit_eqn.invars[i]
        
        if isinstance(invar, Literal):
            invalues.append(invar.val)
            continue
        invalues.append(context_dic[invar])
    
    return jit(exec_jaxpr, static_argnums = [0,1], inline = True)(definition_jaxpr, *invalues)

def exec_jaxpr(jaxpr, *args, **kwargs):
    context_dic = {jaxpr.invars[i] : args[i] for i in range(len(args))}
    eval_jaxpr(jaxpr, context_dic)
    return [context_dic[jaxpr.outvars[i]] for i in range(len(jaxpr.outvars))]

def flatten_pjit(jaxpr):
    return make_jaxpr(eval_jaxpr, static_argnums = [0])(jaxpr)
        
 
    