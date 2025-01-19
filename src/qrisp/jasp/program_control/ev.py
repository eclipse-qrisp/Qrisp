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


from qrisp.jasp.tracing_logic import quantum_kernel

def expectation_value(func, shots):
    
    from qrisp.jasp import make_tracer, qache
    from qrisp.core import QuantumVariable, measure
    
    if isinstance(shots, int):
        shots = make_tracer(shots)
    func = qache(func)
    
    @jax.jit
    def expectation_value_return_function(*args):
        
        return_amount = []
        
        @quantum_kernel
        def body_func(i, args):
            
            acc = args[0]
            res = func(*args[1:])
            
            if not isinstance(res, tuple):
                res = (res,)
            
            return_amount.append(len(res))
            
            meas_res = []
            for qv in res:
                if not isinstance(qv, QuantumVariable):
                    raise Exception("Tried to sample from function not returning a QuantumVariable")
                meas_res.append(measure(qv))
            
            meas_res = jnp.array(meas_res)
            acc += meas_res
            
            return (acc, *args[1:])
        
        try:
            loop_res = jax.lax.fori_loop(0, shots, body_func, (jnp.array([0.]), *args))
            return loop_res[0][0]/shots
        except TypeError:
            loop_res = jax.lax.fori_loop(0, shots, body_func, (jnp.array([0.]*return_amount[0]), *args))
            return loop_res[0]/shots
            
    return expectation_value_return_function