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

from qrisp.jasp import qache, jrange
from qrisp.core import x, cx, QuantumVariable, mcx
from qrisp.environments import control, custom_control

@jit
def extract_boolean_digit(integer, digit):
    return jnp.bool((integer>>digit & 1))

# This is an adaption of the Gidney adder to also support classical input values
@custom_control
def jasp_cq_gidney_adder(a, b, ctrl = None):
    
    if isinstance(b, list):
        n = len(b)
    else:
        n = b.size

    # If the quantum target only has a single qubit, the addition can be performed
    # with a simply X-gate
    
    # The relevant observation to turn Gidney's adder into a semi-classical adder
    # is that the "a_i" qubit (in his paper these qubits are called "i") 
    # is only involved in two interactions:
    #   1. The CNOT gate from the ancilla qubit of the previous iteration
    #   2. The the quasi-toffoli gate into the ancilla of the current interaction
    # If we consider "a_i" as a classically known value, we can recover the quantum
    # value of the first step by applying an x gate onto the ancilla of the previous 
    # iteration if a_i is True.
    
    # To "simulate" the quasi-toffoli we then simply use this ancilla as a control
    # value instead.
    
    # Combining these steps results in a quasi-toffoli with a modified control-state
    # If a is True, the quasi-toffoli receives the flipped value of the ancilla
    
    
    # if n > 1:
    with control(n > 1):
        
        i = 0
        gidney_anc = QuantumVariable(n-1, name = "gidney_anc*")
        
        # Initial Toffoli
        with control(extract_boolean_digit(a, i)):
            if ctrl is None:
                cx(b[i], gidney_anc[i])
            else:
                mcx([ctrl, b[i]], gidney_anc[i], method = "gidney")
    
        
        # Left part of the V shape
        for j in jrange(n-2):
            
            i = j+1
        
            cx(gidney_anc[i-1], b[i])
            
            with control(extract_boolean_digit(a, i)):
                if ctrl is None:
                    x(gidney_anc[i-1])
                else:
                    cx(ctrl, gidney_anc[i-1])
            
            mcx([gidney_anc[i-1], b[i]], gidney_anc[i], method = "gidney")
            
            with control(extract_boolean_digit(a, i)):
                if ctrl is None:
                    x(gidney_anc[i-1])
                else:
                    cx(ctrl, gidney_anc[i-1])
                    
            cx(gidney_anc[i-1], gidney_anc[i])
        
        
        # Tip of the V shape
        cx(gidney_anc[n-2], b[n-1])
        
        # Right part of the V shape
        for j in jrange(n-2):
            
            i = n-j-2
            cx(gidney_anc[i-1], gidney_anc[i])        
            
            if ctrl is not None:
                with control(extract_boolean_digit(a, i)):
                    cx(ctrl, gidney_anc[i-1])
                mcx([gidney_anc[i-1], b[i]], gidney_anc[i], method = "gidney")
                with control(extract_boolean_digit(a, i)):
                    cx(ctrl, gidney_anc[i-1])
                    
            else:
                with control(extract_boolean_digit(a, i)):
                    x(gidney_anc[i-1])
                mcx([gidney_anc[i-1], b[i]], gidney_anc[i], method = "gidney_inv")
                with control(extract_boolean_digit(a, i)):
                    x(gidney_anc[i-1])
                
        # Final Toffoli
        with control(extract_boolean_digit(a, 0)):
            if ctrl is None:
                cx(b[0], gidney_anc[0])
            else:
                mcx([ctrl, b[0]], gidney_anc[0], method = "gidney_inv")
        gidney_anc.delete()
    
    # CX gates at the right side of the circuit
    for i in jrange(n):
        with control(extract_boolean_digit(a, i)):
            if ctrl is not None:
                cx(ctrl, b[i])
            else:
                x(b[i])
