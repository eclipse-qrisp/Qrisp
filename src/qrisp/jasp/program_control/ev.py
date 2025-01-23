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

from random import shuffle

import jax
import jax.numpy as jnp
import numpy as np

from qrisp.jasp.tracing_logic import quantum_kernel, check_for_tracing_mode


def expectation_value(func, shots, return_dict = False, post_processor = None):
    
    from qrisp.jasp import make_tracer, qache
    from qrisp.core import QuantumVariable, measure
    
    if isinstance(shots, int):
        shots = make_tracer(shots)
    
    if post_processor is None:
        def identity(*args):
            return args
        
        post_processor = identity
    
    @qache
    def user_func(*args):
        return func(*args)
    
    def expectation_value_eval_function(*args):
        
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
            def sampling_helper_2(*meas_ints):
                res_list = []
                for i in range(len(qv_tuple)):
                    res_list.append(qv_tuple[i].decoder(meas_ints[i]))
                
                return post_processor(*res_list)
            
            decoded_values = sampling_helper_2(*(list(measurement_ints)))
            
            meas_res = jnp.array(decoded_values)
            acc += meas_res
            
            return (acc, *args[1:])
        
        try:
            loop_res = jax.lax.fori_loop(0, shots, sampling_body_func, (jnp.array([0.]), *args))
            return loop_res[0][0]/shots
        except TypeError:
            loop_res = jax.lax.fori_loop(0, shots, sampling_body_func, (jnp.array([0.]*return_amount[0]), *args))
            return loop_res[0]/shots
        
    if return_dict:
        expectation_value_eval_function.__name__ = "dict_sampling_eval_function"
            
    return jax.jit(expectation_value_eval_function)