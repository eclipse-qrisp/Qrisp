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

from jax.core import JaxprEqn, Jaxpr, Literal
from jax import make_jaxpr

from qrisp.jax.flattening_tools import exec_eqn, eval_jaxpr, extract_invalues, eval_jaxpr_with_context_dic

def flatten_environment_eqn(env_eqn, context_dic):
    
    body_jaxpr = env_eqn.params["jaxpr"]
    
    invalues = extract_invalues(env_eqn, context_dic)
    
    new_context_dic = {}
        
    for i in range(len(body_jaxpr.invars)):
        new_context_dic[body_jaxpr.invars[i]] = invalues[i]
    
    for i in range(len(body_jaxpr.constvars)):
        new_context_dic[body_jaxpr.constvars[i]] = context_dic[body_jaxpr.constvars[i]]
    
    eval_jaxpr_with_context_dic(body_jaxpr, new_context_dic)
    
    for i in range(len(env_eqn.outvars)):
        context_dic[env_eqn.outvars[i]] = new_context_dic[body_jaxpr.outvars[i]]
    
 
def collect_environments(jaxpr):
    
    eqn_list = jaxpr.eqns
    new_eqn_list = []
    
    while len(eqn_list) != 0:
        
        eqn = eqn_list.pop(0)
        
        if eqn.primitive.name == "q_env" and eqn.params["stage"] == "exit":
            
            for i in range(len(new_eqn_list))[::-1]:
                enter_eq = new_eqn_list[i]
                if enter_eq.primitive.name == "q_env" and enter_eq.params["stage"] == "enter":
                    break
            else:
                raise
            
            environment_body_eqn_list = new_eqn_list[i+1:]
            
            invars = find_invars(environment_body_eqn_list)
            
            outvars = find_outvars(environment_body_eqn_list, [eqn] + eqn_list)
            
            invars.remove(enter_eq.outvars[0])
            
            environment_body_jaxpr = Jaxpr(constvars = invars,
                                           invars = enter_eq.outvars,
                                           outvars = outvars,
                                           eqns = environment_body_eqn_list)
            
            eqn = JaxprEqn(
                           params = {"stage" : "collected", "jaxpr" : environment_body_jaxpr},
                           primitive = eqn.primitive,
                           invars = list(enter_eq.invars),
                           outvars = list(eqn.outvars),
                           effects = eqn.effects,
                           source_info = eqn.source_info)
            
            
            new_eqn_list = new_eqn_list[:i]
        
        new_eqn_list.append(eqn)
        
    return Jaxpr(constvars = jaxpr.constvars, 
                 invars = jaxpr.invars,
                 outvars = jaxpr.outvars,
                 eqns = new_eqn_list)
    
                
            
def find_invars(eqn_list):
    
    defined_vars = {}
    invars = []
    
    for eqn in eqn_list:
        for var in eqn.invars:
            if isinstance(var, Literal):
                continue
            if var not in defined_vars:
                invars.append(var)
        
        for var in eqn.outvars:
            defined_vars[var] = None
    
    return list(set(invars))

def find_outvars(body_eqn_list, script_remainder_eqn_list):
    
    outvars = []
    
    for eqn in body_eqn_list:
        outvars.extend(eqn.outvars)
    
    outvars = list(set(outvars))
    
    required_remainder_vars = find_invars(script_remainder_eqn_list)
    
    return list(set(outvars).intersection(required_remainder_vars))
                
    
def flatten_environments(jaxpr):
    
    jaxpr = collect_environments(jaxpr)
    eqn_evaluator_function_dic = {"q_env" : flatten_environment_eqn}
    return make_jaxpr(eval_jaxpr(jaxpr, 
                                 eqn_evaluator_function_dic = eqn_evaluator_function_dic))(*[var.aval for var in jaxpr.invars])
    