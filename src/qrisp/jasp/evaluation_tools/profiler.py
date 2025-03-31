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

import jax
import jax.numpy as jnp

from qrisp.jasp.primitives import OperationPrimitive
from qrisp.jasp.interpreter_tools import make_profiling_eqn_evaluator, eval_jaxpr

def count_ops(function):
    """
    Decorator to determine resources of large scale quantum computations.
    This decorator compiles the given Jasp-compatible function into a classical
    function computing the amount of each gates required. The decorated function
    will return a dictionary containing the operation counts.
    
    .. warning::
        
        It is currently not possible to estimate programs, which include a 
        :ref:`kernelized <quantum_kernel>` function.

    Parameters
    ----------
    function : callable
        A Jasp-compatible function without :ref:`QuantumKernels <quantum_kernel>`.

    Returns
    -------
    resource_estimator
        A function computing the required resources.
        
    Examples
    --------
    
    We compute the resources required to perform a large scale integer multiplication.
    
    ::
        
        from qrisp import count_ops, QuantumFloat, measure
        
        @count_ops
        def main(i):
            
            a = QuantumFloat(i)
            b = QuantumFloat(i)
            
            c = a*b
            
            return measure(c)
        
        print(main(5))
        # {'cx': 506, 'x': 22, 'h': 135, 'measure': 55, '2cx': 2, 's': 45, 't': 90, 't_dg': 90}
        print(main(5000))
        # {'cx': 462552491, 'x': 20002, 'h': 112522500, 'measure': 37517500, '2cx': 2, 's': 37507500, 't': 75015000, 't_dg': 75015000}
    
    Note that even though the second computation contains more than 800 million gates, 
    determining the resources takes less than 200ms, highlighting the scalability
    features of the Jasp infrastructure.

    """
    
    def ops_counter(*args):
        
        from qrisp.jasp import make_jaspr
        
        if not hasattr(function, "jaspr_dict"):
            function.jaspr_dict = {}
        
        args = list(args)
        
        signature = tuple([type(arg) for arg in args])
        if not signature in function.jaspr_dict:
            function.jaspr_dict[signature] = make_jaspr(function)(*args)
        
        return function.jaspr_dict[signature].count_ops(*args)
    
    return ops_counter


# This function is the central interface for performing resource estimation.
# It takes a Jaspr and returns a function, returning a dictionary (with the counted
# operations).
def profile_jaspr(jaspr):
    
    def profiler(*args):
        
        # The profiling array computer is a function that computes the array 
        # countaining the gate counts.
        # The profiling dic is a dictionary of type {str : int}, which indicates
        # which operation has been computed at which index of the array.
        profiling_array_computer, profiling_dic = get_profiling_array_computer(jaspr)
    
        # Compute the profiling array
        profiling_array = profiling_array_computer(*args)[-1]
        
        # Transform to a dictionary containing gate counts
        res_dic = {}
        for k in profiling_dic.keys():
            if int(profiling_array[profiling_dic[k]]):
                res_dic[k] = int(profiling_array[profiling_dic[k]])
            
        return res_dic

    return profiler

# This function takes a Jaspr and returns a function computing the "counting array"
@lru_cache(int(1E5))
def get_profiling_array_computer(jaspr):
    
    # This functions determines the set of primitives that appear in a given Jaxpr
    primitives = get_primitives(jaspr)
    
    # Filter out the non OperationPrimitives and fill them in a dictionary
    profiling_dic = {}
    for i in range(len(primitives)):
        if isinstance(primitives[i], OperationPrimitive) and not primitives[i].op.name in profiling_dic:
            profiling_dic[primitives[i].op.name] = len(profiling_dic) - 1
        elif primitives[i].name == "jasp.measure" and not "measure" in profiling_dic:
            profiling_dic["measure"] = len(profiling_dic) - 1
    
    # This function calls the profiling interpeter to evaluate the gate counts
    @jax.jit
    def profiling_array_computer(*args):
        
        profiling_eqn_evaluator = make_profiling_eqn_evaluator(profiling_dic)
        
        args = list(args)
        args = args + [jnp.zeros(len(profiling_dic), dtype = "int64")]
        
        res = eval_jaxpr(jaspr, eqn_evaluator = profiling_eqn_evaluator)(*args)
        
        return res
    
    return profiling_array_computer, profiling_dic


# This functions determines the set of primitives that appear in a given Jaxpr
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
            if isinstance(param, jax.core.ClosedJaxpr):
                primitives.update(get_primitives(param.jaxpr))
    
    return list(primitives)