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

from functools import lru_cache

import jax.numpy as jnp
import jax

from qrisp.jasp.primitives import OperationPrimitive
from jax.core import ClosedJaxpr

from qrisp.jasp.interpreter_tools import make_profiling_eqn_evaluator, eval_jaxpr

def profile_jaspr(jaspr):
    
    def profiler(*args):
    
        profiling_array_computer, profiling_dic = get_profiling_array_computer(jaspr)
    
        profiling_array = profiling_array_computer(*args)[-1]
        
        res_dic = {}
        for k in profiling_dic.keys():
            if int(profiling_array[profiling_dic[k]]):
                res_dic[k] = int(profiling_array[profiling_dic[k]])
            
        return res_dic

    return profiler

@lru_cache(int(1E5))           
def get_compiled_profiler(jaxpr, zipped_profiling_dic):
    
    profiling_dic = dict(zipped_profiling_dic)
    
    profiling_eqn_evaluator = make_profiling_eqn_evaluator(profiling_dic)
    
    @jax.jit
    def profiler(*args):
        return eval_jaxpr(jaxpr, eqn_evaluator = profiling_eqn_evaluator)(*args)
    
    return profiler

@lru_cache(int(1E5))
def get_profiling_array_computer(jaspr):
    
    primitives = get_primitives(jaspr)
    
    profiling_dic = {}
    
    for i in range(len(primitives)):
        if isinstance(primitives[i], OperationPrimitive) and not primitives[i].op.name in profiling_dic:
            profiling_dic[primitives[i].op.name] = len(profiling_dic) - 1
    
    @jax.jit
    def profiling_array_computer(*args):
        
        profiling_eqn_evaluator = make_profiling_eqn_evaluator(profiling_dic)
        
        args = list(args)
        args = args + [jnp.zeros(len(profiling_dic), dtype = "int64")]
        
        res = eval_jaxpr(jaspr, eqn_evaluator = profiling_eqn_evaluator)(*args)
        
        return res
    
    return profiling_array_computer, profiling_dic



def get_primitives(jaxpr):
    
    primitives = set()
    
    for eqn in jaxpr.eqns:
        # Add current primitive
        primitives.add(eqn.primitive)
        
        if eqn.primitive.name == "cond":
            primitives.update(get_primitives(eqn.params["branches"][0].jaxpr))
            primitives.update(get_primitives(eqn.params["branches"][1].jaxpr))
            continue
        
        # Handle call primitives (like cond/pjit)
        for param in eqn.params.values():
            if isinstance(param, ClosedJaxpr):
                primitives.update(get_primitives(param.jaxpr))
    
    return list(primitives)