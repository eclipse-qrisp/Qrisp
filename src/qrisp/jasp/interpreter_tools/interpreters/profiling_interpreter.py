"""
\********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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

from qrisp.jasp.interpreter_tools.abstract_interpreter import insert_outvalues, extract_invalues, eval_jaxpr
from qrisp.jasp.primitives import QuantumPrimitive, OperationPrimitive

from jax.lax import while_loop, cond
import jax.numpy as jnp

def make_profiling_eqn_evaluator(profiling_dic):
    
    from qrisp import Jaspr

    def profiling_eqn_evaluator(eqn, context_dic):
        
        invalues = extract_invalues(eqn, context_dic)
        
        if isinstance(eqn.primitive, QuantumPrimitive):
            
            if isinstance(eqn.primitive, OperationPrimitive):
                
                counting_array = invalues[-1]
                
                op_name = eqn.primitive.op.name
                if op_name not in profiling_dic:
                    profiling_dic[op_name] = len(profiling_dic)
                    
                counting_index = profiling_dic[op_name]
                counting_array = counting_array.at[counting_index].add(1)
                
                insert_outvalues(eqn, context_dic, counting_array)
            
            elif eqn.primitive.name == "jasp.create_qubits":
                insert_outvalues(eqn, context_dic, invalues)
            
            elif eqn.primitive.name == "jasp.get_size":
                insert_outvalues(eqn, context_dic, invalues[0])
                
            elif eqn.primitive.name == "jasp.fuse":
                insert_outvalues(eqn, context_dic, invalues[0] + invalues[1])
                
            elif eqn.primitive.name == "jasp.slice":
                
                start = jnp.max(jnp.array([invalues[1], 0]))
                stop = jnp.min(jnp.array([invalues[2], invalues[0]]))
                
                insert_outvalues(eqn, context_dic, stop - start)
                
            elif eqn.primitive.name == "jasp.get_qubit":
                insert_outvalues(eqn, context_dic, None)
                
            else:
                if len(eqn.outvars) == 1:
                    insert_outvalues(eqn, context_dic, invalues[-1])
                else:
                    insert_outvalues(eqn, context_dic, (len(eqn.outvars)-1)*[0] + [invalues[-1]])
                
        elif eqn.primitive.name == "while":
            
            def body_fun(val):
                body_res = eval_jaxpr(eqn.params["body_jaxpr"], 
                                       eqn_evaluator = profiling_eqn_evaluator)(*val)
                return tuple(body_res)
    
            def cond_fun(val):
                res = eval_jaxpr(eqn.params["cond_jaxpr"], 
                                       eqn_evaluator = profiling_eqn_evaluator)(*val)
                return res
            
            outvalues = while_loop(cond_fun, body_fun, tuple(invalues))
            
            insert_outvalues(eqn, context_dic, outvalues)
            
        elif eqn.primitive.name == "cond":
            
            false_fun = eval_jaxpr(eqn.params["branches"][0], 
                                   eqn_evaluator = profiling_eqn_evaluator)
            
            true_fun = eval_jaxpr(eqn.params["branches"][1], 
                                   eqn_evaluator = profiling_eqn_evaluator)
            
            outvalues = cond(invalues[0], true_fun, false_fun, *invalues[1:])
            
            if not isinstance(outvalues, (list, tuple)):
                outvalues = (outvalues, )
            
            insert_outvalues(eqn, context_dic, outvalues)
        
        elif eqn.primitive.name == "pjit" and isinstance(eqn.params["jaxpr"].jaxpr, Jaspr):
            
            zipped_profiling_dic = tuple(profiling_dic.items())
            
            from qrisp.jasp.evaluation_tools import get_compiled_profiler
            
            profiler = get_compiled_profiler(eqn.params["jaxpr"].jaxpr, zipped_profiling_dic)
            
            outvalues = profiler(*invalues)
            
            if not isinstance(outvalues, (list, tuple)):
                outvalues = (outvalues, )
            
            insert_outvalues(eqn, context_dic, outvalues)
                
        else:
            return True
        
    return profiling_eqn_evaluator

    