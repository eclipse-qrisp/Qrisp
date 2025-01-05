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

import jax.numpy as jnp
from jax import jit
from jax.core import eval_jaxpr

from qrisp.jasp import make_jaspr

from qrisp.jasp.interpreter_tools.interpreters.cl_func_interpreter import jaspr_to_cl_func_jaxpr

def boolean_simulation(func, bit_array_size = 2**14):
    
    @jit    
    def return_function(*args):
        
        jaspr = make_jaspr(func)(*args)
        cl_func_jaxpr = jaspr_to_cl_func_jaxpr(jaspr.flatten_environments(), bit_array_size)
        
        aval = cl_func_jaxpr.invars[0].aval
        res = eval_jaxpr(cl_func_jaxpr, 
                         [], 
                         jnp.zeros(aval.shape, dtype = aval.dtype), 
                         jnp.array(0, dtype = jnp.int64), *args)
        
        if len(res) == 3:
            return res[2]
        else:
            return res[2:]
    
    return return_function
