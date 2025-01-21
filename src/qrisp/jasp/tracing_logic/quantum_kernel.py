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

from qrisp.jasp.primitives import quantum_kernel_p
from qrisp.jasp.tracing_logic import TracingQuantumSession, qache

def quantum_kernel(func):
    
    func = qache(func)
    
    def return_function(*args, **kwargs):
        
        from qrisp.jasp.jasp_expression.centerclass import Jaspr, collect_environments
        
        qs = TracingQuantumSession.get_instance()
        
        qs.start_tracing(quantum_kernel_p.bind())
        
        try:
            res = func(*args, **kwargs)
        except Exception as e:
            qs.conclude_tracing()
            raise e
        
        eqn = jax._src.core.thread_local_state.trace_state.trace_stack.dynamic.jaxpr_stack[0].eqns[-1]
        flattened_jaspr = Jaspr.from_cache(collect_environments(eqn.params["jaxpr"].jaxpr)).flatten_environments()
        eqn.params["jaxpr"] = jax.core.ClosedJaxpr(flattened_jaspr, eqn.params["jaxpr"].consts)
        
        qs.conclude_tracing()
        
        return res
    
    return return_function
        
        