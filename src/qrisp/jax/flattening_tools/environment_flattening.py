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

from jax.core import JaxprEqn, Jaxpr
from jax import make_jaxpr

from qrisp.jax.flattening_tools import exec_eqn, eval_jaxpr

def flatten_environment_eqn(env_eqn, context_dic):
    
    context_dic[env_eqn.params["circuit_var"]] = context_dic[env_eqn.invars[0]]
    
    for eqn in env_eqn.params["env_data"]:
        exec_eqn(eqn, context_dic)
    
    context_dic[env_eqn.outvars[0]] = context_dic[env_eqn.invars[0]]
    
 
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
            
            
            eqn = JaxprEqn(
                           params = {"stage" : "compile", "env_data" : new_eqn_list[i+1:], "circuit_var" : enter_eq.outvars[0]},
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


def flatten_environments(jaxpr):
    
    jaxpr = collect_environments(jaxpr)
    eqn_evaluator_function_dic = {"q_env" : flatten_environment_eqn}
    return make_jaxpr(eval_jaxpr(jaxpr, 
                                 eqn_evaluator_function_dic = eqn_evaluator_function_dic))(*[var.aval for var in jaxpr.invars])
    