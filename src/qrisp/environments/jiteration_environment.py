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

from qrisp.environments import QuantumEnvironment
from jax.lax import while_loop


def iteration_env_evaluator(eqn, context_dic):
    if isinstance(eqn.primitive, JIterationEnvironment):
        
        if not hasattr(eqn.primitive, "iteration_1"):
            eqn.primitive.iteration_1 = eqn
            return None

        iteration_1 = eqn.primitive.iteration_1
        iteration_2 = eqn
        iter_1_jispr = iteration_1.params["jispr"]
        iter_2_jispr = iteration_2.params["jispr"]
        
        # Move index to the last argument
        
        increment_eq = iter_1_jispr.eqns[-1]
        arg_pos = iter_1_jispr.invars.index(increment_eq.invars[0])
        
        iter_1_jispr.invars.append(iter_1_jispr.invars.pop(arg_pos))
        iteration_1.invars.append(iteration_1.invars.pop(arg_pos))
        
        permutation = find_signature_permutation(iter_1_jispr, iter_2_jispr)
        
        permuted_invars = [iter_2_jispr.invars[i] for i in permutation]
        iter_2_jispr.invars.clear()
        iter_2_jispr.invars.extend(permuted_invars)
        

        permuted_invars = [iteration_2.invars[i] for i in permutation]
        iteration_2.invars.clear()
        iteration_2.invars.extend(permuted_invars)
        
        # order = dict(zip(iteration_2.invars, permutation))
        # iteration_2.invars.sort(key=order.get)
        

        # Figure out which of the return values need to be updated.
        # The update needs to happen if the input values of both iterations are 
        # not the same. In this case we find the updated value in the 
        
        iter_1_invar_hashes = [hash(x) for x in iteration_1.invars]
        iter_2_invar_hashes = [hash(x) for x in iteration_2.invars]
        iter_1_outvar_hashes = [hash(x) for x in iteration_1.outvars]
        
        update_rules = []
        
        for i in range(len(iter_1_jispr.invars)):
            
            if iter_1_invar_hashes[i] not in iter_2_invar_hashes:
                res_index = iter_1_outvar_hashes.index(iter_2_invar_hashes[i])
                update_rules.append(res_index)
            else:
                update_rules.append(None)
        
        
        
        def body_fun(val):
            
            res = iter_1_jispr.eval(*val[:-1])
            
            if not isinstance(res, tuple):
                res = (res,)
            
            return_values = []
            
            for i in range(len(update_rules)):
                if update_rules[i] is None:
                    return_values.append(val[i])
                else:
                    return_values.append(res[update_rules[i]])
            
            return tuple(return_values + [val[-1]])
        
        def cond_fun(val):
            return val[-2] < val[-1]
        
        init_val = [context_dic[x] for x in iteration_1.invars]
        init_val.insert(-1, 0)
        res = while_loop(cond_fun, body_fun, init_val = tuple(init_val))
        for i in range(len(update_rules)-1):
            if update_rules[i] is not None:
                iteration_2.outvars[update_rules[i]]
                context_dic[iteration_2.outvars[update_rules[i]]] = res[i]
    
    else:
        return True

def find_signature_permutation(jaxpr_0, jaxpr_1):
    
    invars = list(jaxpr_0.invars)
    translation_dic = {}
    for i in range(len(jaxpr_0.eqns)):
        eqn_0 = jaxpr_0.eqns[i]
        eqn_1 = jaxpr_1.eqns[i]
        
        for j in range(len(eqn_0.invars)):
            var = eqn_0.invars[j]
            if var in invars:
                translation_dic[var] = eqn_1.invars[j]
                invars.remove(var)
        
        if len(invars) == 0:
            break

    return [jaxpr_1.invars.index(translation_dic[var]) for var in jaxpr_0.invars]


class JIterationEnvironment(QuantumEnvironment):
    
    def jcompile(self, eqn, context_dic):
        iteration_env_evaluator(eqn, context_dic)