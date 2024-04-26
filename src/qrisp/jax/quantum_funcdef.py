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

import weakref

from qrisp.core import QuantumSession, QuantumVariable
from qrisp.jax import qdef_p, qcall_p

qc_cache = {}
def qfunc_def(function):
    
    def jitted_function(*args):
        
        if tuple(type(arg) for arg in args) in qc_cache and qc_cache[tuple(type(arg) for arg in args)]() is not None:
            function_qc = qc_cache[tuple(type(arg) for arg in args)]()
        
        else:
        
            old_abs_qc = QuantumSession.abs_qc
            
            qargs = []
            
            for arg in args:
                if isinstance(arg, QuantumVariable):
                    qargs.append(arg)
            
            
            quantum_function_tuple = qdef_p.bind(num_args = len(qargs))
            
            capturing_qc = quantum_function_tuple[0]
            capturing_qb_arrays = quantum_function_tuple[1:]
            
            old_qubit_arrays = []
            for i in range(len(qargs)):
                qarg = qargs[i]
                old_qubit_arrays.append(qarg.reg)
                qarg.reg = capturing_qb_arrays[i]
            
            QuantumSession.abs_qc = weakref.ref(capturing_qc)
            
            function(*args)
            
            function_qc = QuantumSession.abs_qc()
            
            for qv in qargs:
                qv.reg = old_qubit_arrays.pop(0)
            
            QuantumSession.abs_qc = old_abs_qc
            
            qc_cache[tuple(type(arg) for arg in args)] = weakref.ref(function_qc)
            
        def calling_function(*args):
            
            qb_array_args = [qv.reg for qv in args]
            
            conclusion_qc = qcall_p.bind(QuantumSession.abs_qc(), function_qc, *qb_array_args)
            QuantumSession.abs_qc = weakref.ref(conclusion_qc)
            
        
        return calling_function(*args)
    
    return jitted_function
            