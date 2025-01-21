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

from qrisp.jasp.tracing_logic import quantum_kernel


def sample_(func, amount):
    
    def return_function(*args, **kwargs):
    
        @quantum_kernel
        def kernel(argtuple, dummy_arg):
            from qrisp.core import QuantumVariable, measure
            
            res = func(*argtuple, **kwargs)
            
            if not isinstance(res, tuple):
                res = (res,)
            
            meas_res = []
            for qv in res:
                if not isinstance(qv, QuantumVariable):
                    raise Exception("Tried to sample from function not returning a QuantumVariable")
                meas_res.append(measure(qv))
            
            return argtuple, jax.numpy.array(meas_res)
    
        scan_res = jax.lax.scan(kernel, args, length = amount)[1]
        
        if scan_res.shape[1] == 1:
            return scan_res[:,0]
        else:
            return scan_res
    
    return return_function

        