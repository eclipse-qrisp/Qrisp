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

import jax
import jax.numpy as jnp

from qrisp.jasp.tracing_logic import quantum_kernel, check_for_tracing_mode

def sample(func = None, shots = 0, post_processor = None):
    
    from qrisp.jasp import qache
    from qrisp.core import QuantumVariable, measure
    
    if isinstance(func, int):
        shots = func
        func = None
    
    if func is None:
        return lambda x : sample(x, shots, post_processor = post_processor)
    
    if post_processor is None:
        def identity(*args):
            if len(args) == 1:
                return args[0]
            return args
        
        post_processor = identity
    
    if isinstance(shots, jax.core.Tracer):
        raise Exception("Tried to sample with dynamic shots value (static integer required)")
    elif not isinstance(shots, int):
        raise Exception(f"Tried to sample with shots value of non-integer type {type(shots)}")
    
    @qache
    def user_func(*args):
        return func(*args)

    @jax.jit
    def sampling_eval_function(tracerized_shots, *args):
        
        return_amount = []
        
        @quantum_kernel
        def sampling_body_func(i, args):
            
            acc = args[0]
            
            qv_tuple = user_func(*args[1:])
            
            if not isinstance(qv_tuple, tuple):
                qv_tuple = (qv_tuple,)
            
            return_amount.append(len(qv_tuple))
            
            for qv in qv_tuple:
                if not isinstance(qv, QuantumVariable):
                    raise Exception("Tried to sample from function not returning a QuantumVariable")
            
            # Trace the DynamicQubitArray measurements
            @qache
            def sampling_helper_1(*args):
                res_list = []
                for reg in args:
                    res_list.append(measure(reg))
                return tuple(res_list)
            
            measurement_ints = sampling_helper_1(*[qv.reg for qv in qv_tuple])
            
            # Trace the decoding
            @jax.jit
            def sampling_helper_2(acc, i, *meas_ints):
                decoded_values = []
                for j in range(len(qv_tuple)):
                    decoded_values.append(qv_tuple[j].decoder(meas_ints[j]))
            
                if len(qv_tuple) > 1:
                    decoded_values = jnp.array(post_processor(*decoded_values))
                else:
                    decoded_values = post_processor(*decoded_values)
                acc = acc.at[i].set(decoded_values)
                
                return acc
            
            acc = sampling_helper_2(acc, i, *measurement_ints)
            
            return (acc, *args[1:])
        
        try:
            loop_res = jax.lax.fori_loop(0, tracerized_shots, sampling_body_func, (jnp.zeros(shots), *args))
            return loop_res[0]
        except ValueError:
            loop_res = jax.lax.fori_loop(0, tracerized_shots, sampling_body_func, (jnp.zeros((shots, return_amount[0])), *args))
            return loop_res[0]
    
    from qrisp.jasp import jaspify
    def return_function(*args):
        
        if check_for_tracing_mode():
            return sampling_eval_function(shots, *args)
        else: 
            def tracing_function(*args):
                from qrisp.jasp.program_control import expectation_value
                return expectation_value(func, shots, return_dict = True)(*args)
            
            return jaspify(tracing_function, return_dict = True)(*args)
    
    return return_function