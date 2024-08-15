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

from jax import jit
from qrisp.jax import get_tracing_qs, check_for_tracing_mode

def qache(func):
    from qrisp.core.quantum_variable import QuantumVariable, flatten_qv, unflatten_qv
    
    def ammended_function(abs_qc, *args):
        
        qs = get_tracing_qs()
        qs.abs_qc = abs_qc
        
        res = func(*args)
        
        return qs.abs_qc, res
    
    ammended_function.__name__ = func.__name__
    
    ammended_function = jit(ammended_function)
    
    def return_function(*args, **kwargs):
        
        if not check_for_tracing_mode():
            return func(*args, **kwargs)
        
        abs_qs = get_tracing_qs()
        
        flattened_qvs = []
        
        for arg in args:
            if isinstance(arg, QuantumVariable):
                flattened_qvs.append(flatten_qv(arg))
                
        
        abs_qc_new, res = ammended_function(abs_qs.abs_qc, *args, **kwargs)
        
        for tup in flattened_qvs:
            unflatten_qv(*tup[::-1])
        
        abs_qs.abs_qc = abs_qc_new
        
        return res
    
    return return_function
        
        
