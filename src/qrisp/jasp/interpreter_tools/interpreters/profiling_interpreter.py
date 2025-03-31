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

"""
This file implements the tools to perform quantum resource estimation using Jasp
infrastructure. The idea here is to transform the quantum instructions within a
given Jaspr into "counting instructions". That means instead of performing some
quantum gate, we increment an index in an array, which keeps track of how many
instructions of each type have been performed.

To do this, we implement the 

qrisp.jasp.interpreter_tools.interpreters.profiling_interpreter.py

Which handles the transformation logic of the Jaspr.
This file implements the interfaces to evaluating the transformed Jaspr.

"""


from functools import lru_cache

from qrisp.jasp.interpreter_tools.abstract_interpreter import insert_outvalues, extract_invalues, eval_jaxpr
from qrisp.jasp.primitives import QuantumPrimitive, OperationPrimitive

import jax
import jax.numpy as jnp


# This functions takes a "profiling dic", i.e. a dictionary of the form {str : int}
# indicating what kinds of quantum gates can appear in a Jaspr.
# It returns an equation evaluator, which increments a counter in an array for
# each quantum operation.
def make_profiling_eqn_evaluator(profiling_dic):
    
    from qrisp import Jaspr

    def profiling_eqn_evaluator(eqn, context_dic):
        
        invalues = extract_invalues(eqn, context_dic)
        
        if isinstance(eqn.primitive, QuantumPrimitive):
            
            # In the case of an OperationPrimitive, we determine the array index 
            # to be increment via dictionary look-up and perform the increment
            # via the Jax-given .at method.            
            if isinstance(eqn.primitive, OperationPrimitive):
                
                counting_array = invalues[-1]
                
                op_name = eqn.primitive.op.name
                if op_name not in profiling_dic:
                    profiling_dic[op_name] = len(profiling_dic)
                    
                counting_index = profiling_dic[op_name]
                counting_array = counting_array.at[counting_index].add(1)
                
                insert_outvalues(eqn, context_dic, counting_array)
            
            # Since we don't need to track to which qubits a certain operation
            # is applied, we can implement a really simple behavior for most 
            # Qubit/QubitArray handling methods.
            # We represent qubit arrays simply with integers (indicating their)
            # size.
            elif eqn.primitive.name == "jasp.create_qubits":
                # create_qubits has the signature (size, QuantumCircuit).
                # Since we represent QubitArrays via integers, it is sufficient
                # to simply return the input as the output for this primitive.
                insert_outvalues(eqn, context_dic, invalues)
            
            elif eqn.primitive.name == "jasp.get_size":
                # The QubitArray size is represented via an integer.
                insert_outvalues(eqn, context_dic, invalues[0])
                
            elif eqn.primitive.name == "jasp.fuse":
                # The size of the fused qubit array is the size of the two added.
                insert_outvalues(eqn, context_dic, invalues[0] + invalues[1])
                
            elif eqn.primitive.name == "jasp.slice":
                # For the slice operation, we need to make sure, we don't go out
                # of bounds.
                start = jnp.max(jnp.array([invalues[1], 0]))
                stop = jnp.min(jnp.array([invalues[2], invalues[0]]))
                
                insert_outvalues(eqn, context_dic, stop - start)
                
            elif eqn.primitive.name == "jasp.get_qubit":
                # Trivial behavior since we don't need qubit address information
                insert_outvalues(eqn, context_dic, None)
                
            elif eqn.primitive.name in ["jasp.delete_qubits", "jasp.reset"]:
                # Trivial behavior: return the last argument (the counting array).
                insert_outvalues(eqn, context_dic, invalues[-1])
                
            elif eqn.primitive.name == "jasp.measure":
                # The measurement returns always 0
                insert_outvalues(eqn, context_dic, [0, invalues[-1]])
                
        elif eqn.primitive.name == "while":
            
            # Reinterpreted body and cond function
            def body_fun(val):
                body_res = eval_jaxpr(eqn.params["body_jaxpr"], 
                                       eqn_evaluator = profiling_eqn_evaluator)(*val)
                return tuple(body_res)
    
            def cond_fun(val):
                res = eval_jaxpr(eqn.params["cond_jaxpr"], 
                                       eqn_evaluator = profiling_eqn_evaluator)(*val)
                return res
            
            outvalues = jax.lax.while_loop(cond_fun, body_fun, tuple(invalues))
            
            insert_outvalues(eqn, context_dic, outvalues)
            
        elif eqn.primitive.name == "cond":
            
            # Reinterpret both branches
            false_fun = eval_jaxpr(eqn.params["branches"][0], 
                                   eqn_evaluator = profiling_eqn_evaluator)
            
            true_fun = eval_jaxpr(eqn.params["branches"][1], 
                                   eqn_evaluator = profiling_eqn_evaluator)
            
            outvalues = jax.lax.cond(invalues[0], true_fun, false_fun, *invalues[1:])
            
            if not isinstance(outvalues, (list, tuple)):
                outvalues = (outvalues, )
            
            insert_outvalues(eqn, context_dic, outvalues)
        
        elif eqn.primitive.name == "pjit" and isinstance(eqn.params["jaxpr"].jaxpr, Jaspr):
            
            # For qached functions, we want to make sure, the compiled function
            # contains only a single implementation per qached function.
            
            # Within a Jaspr, it is made sure that qached function, which is called
            # multiple times only calls the Jaspr by reference in the pjit primitive
            # this way no "copies" of the implementation appear, thus keeping
            # the size of the intermediate representation limited.
            
            # We want to carry on this property. For this we use the lru_cache feature
            # on the get_compiler_profiler function. This function returns a 
            # jitted function, which will always return the same object if called
            # with the same object. Since identical qached function calls are
            # represented by the same jaxpr, we achieve our goal.
            
            zipped_profiling_dic = tuple(profiling_dic.items())
            
            profiler = get_compiled_profiler(eqn.params["jaxpr"].jaxpr, zipped_profiling_dic)
            
            outvalues = profiler(*invalues)
            
            if not isinstance(outvalues, (list, tuple)):
                outvalues = (outvalues, )
            
            insert_outvalues(eqn, context_dic, outvalues)
                
        else:
            return True
        
    return profiling_eqn_evaluator


@lru_cache(int(1E5))           
def get_compiled_profiler(jaxpr, zipped_profiling_dic):
    
    profiling_dic = dict(zipped_profiling_dic)
    
    profiling_eqn_evaluator = make_profiling_eqn_evaluator(profiling_dic)
    
    @jax.jit
    def profiler(*args):
        return eval_jaxpr(jaxpr, eqn_evaluator = profiling_eqn_evaluator)(*args)
    
    return profiler