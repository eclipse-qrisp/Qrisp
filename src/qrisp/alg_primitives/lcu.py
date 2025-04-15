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

from qrisp import *
from qrisp.jasp import *
import jax.numpy as jnp
import numpy as np

def inner_LCU(state_prep, unitaries, num_qubits, num_unitaries=None):
    
    qv = QuantumFloat(num_qubits)

    if isinstance(unitaries, (list, tuple)):
        num_unitaries = len(unitaries)
        unitary_func = lambda index: unitaries[index]

    elif callable(unitaries):
        if num_unitaries is None:
            raise ValueError("num_unitaries must be specified for dynamic unitaries")
        unitary_func = unitaries
    
    else:
        raise TypeError("unitaries must be a list/tuple or a callable function")
    
    # Specify the QunatumVariable that indicates which case to execute
    n = jnp.int64(jnp.ceil(jnp.log2(num_unitaries)))
    case_indicator = QuantumFloat(n)
    
    # LCU protocol with conjugate preparation
    with conjugate(state_prep)(case_indicator):
        for i in jrange(num_unitaries):
            qb = QuantumBool()
            with conjugate(mcx)(case_indicator, qb, ctrl_state=i):
                with control(qb):
                    unitary_func(i)(qv)
        
    return case_indicator, qv

@RUS(static_argnums=[1, 2, 3]) 
def LCU(state_prep, unitaries, num_qubits, num_unitaries=None):
    
    if isinstance(unitaries, (list, tuple)):
        num_unitaries = len(unitaries)
        unitary_func = lambda i: unitaries[i]

    elif callable(unitaries):
        if num_unitaries is None:
            raise ValueError("num_unitaries must be specified for dynamic unitaries")
        unitary_func = unitaries
    
    else:
        raise TypeError("unitaries must be a list/tuple or a callable function")
    
    case_indicator, qv = inner_LCU(state_prep, unitary_func, num_qubits, num_unitaries)
    
    # Success condition
    success_bool = (measure(case_indicator) == 0)
    return success_bool, qv

def view_LCU(state_prep, unitaries, num_qubits, num_unitaries=None):
    
    if isinstance(unitaries, (list, tuple)):
        num_unitaries = len(unitaries)
        unitary_func = lambda index: unitaries[index]

    elif callable(unitaries):
        if num_unitaries is None:
            raise ValueError("num_unitaries must be specified for dynamic unitaries")
        unitary_func = unitaries

    else:
        raise TypeError("unitaries must be list/tuple, or callable")
    
    jaspr = make_jaspr(inner_LCU)(state_prep, unitary_func, num_qubits, num_unitaries)
    
    # Convert Jaspr to quantum circuit and return the circuit
    return jaspr.to_qc(num_qubits, num_unitaries)[-1]
