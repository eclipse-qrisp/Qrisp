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

from jax.lax import while_loop, cond
import jax

from qrisp.jasp import TracingQuantumSession, check_for_tracing_mode, delete_qubits_p, AbstractQubitArray

def RUS(trial_function):
    
    def return_function(*trial_args):
        
        def body_fun(args):
            
            previous_trial_res = args[1:len(res_vals)+1]
            
            abs_qs = TracingQuantumSession.get_instance()
            abs_qc = abs_qs.abs_qc
            
            for res_val in previous_trial_res:
                if isinstance(res_val.aval, AbstractQubitArray):
                    abs_qc = delete_qubits_p.bind(abs_qc, res_val)
            
            abs_qs.abs_qc = abs_qc
            
            
            trial_args = args[len(res_vals)+1:]
            trial_res = trial_func_jaspr.inline(*trial_args)
            
            
            return tuple([abs_qs.abs_qc] + list(trial_res) + list(trial_args))
        
        def cond_fun(val):
            return ~val[1]
        
        # We now prepare the "init_val" keyword of the loop.
        
        # For that we extract the invalues for the first iteration
        # Note that the treshold is given as the last argument
        
        from qrisp.jasp import make_jaspr
        trial_func_jaspr = make_jaspr(trial_function)(*trial_args)
        trial_func_jaspr = trial_func_jaspr.flatten_environments()
        
        first_iter_res = trial_function(*trial_args)
        
        arg_vals, arg_tree_def = jax.tree.flatten(trial_args)
        res_vals, res_tree_def = jax.tree.flatten(first_iter_res)
        
        abs_qs = TracingQuantumSession.get_instance()
        
        combined_args = tuple([abs_qs.abs_qc] + list(res_vals) + list(arg_vals))
    
        def true_fun(combined_args):
            return combined_args
        
        def false_fun(combined_args):
            return while_loop(cond_fun, body_fun, init_val = combined_args)
    
        combined_res = cond(first_iter_res[0], true_fun, false_fun, combined_args)
        abs_qs.abs_qc = combined_res[0]
        
        flat_trial_function_res = combined_res[1:len(res_vals)+1]
        
        replacement_dic = {}
        
        for i in range(len(flat_trial_function_res)):
            replacement_dic[res_vals[i]] = flat_trial_function_res[i]
        
        from qrisp import QuantumVariable
        for res_val in first_iter_res:
            if isinstance(res_val, QuantumVariable):
                res_val.reg = replacement_dic[res_val.reg]
        
        trial_function_res = jax.tree.unflatten(res_tree_def, flat_trial_function_res)
        
        if len(first_iter_res) == 2:
            return trial_function_res[1]
        else:
            return trial_function_res[1:]
    
    return return_function