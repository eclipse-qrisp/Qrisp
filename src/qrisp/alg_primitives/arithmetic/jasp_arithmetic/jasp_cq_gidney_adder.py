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
import jax

from qrisp.jasp import qache, jrange
from qrisp.core import x, cx, QuantumVariable, mcx
from qrisp.environments import control, custom_control


@jax.jit
def extract_boolean_digit(integer, digit):
    # return bool(integer>>digit & 1)
    return jnp.bool((integer>>digit & 1))

@custom_control
def jasp_cq_gidney_adder(a, b, ctrl = None):
    
    
    if isinstance(b, list):
        n = len(b)
    else:
        n = b.size
    
    # if n > 1:
    with control(n > 1):
        
        i = 0
        
        gidney_anc = QuantumVariable(n-1, name = "gidney_anc*")
        
        with control(extract_boolean_digit(a, i)):
            if ctrl is None:
                cx(b[i], gidney_anc[i])
            else:
                mcx([ctrl, b[i]], gidney_anc[i], method = "gidney")
    
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
        
        cx(gidney_anc[n-2], b[n-1])
        
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
                
        
        with control(extract_boolean_digit(a, 0)):
            if ctrl is None:
                cx(b[0], gidney_anc[0])
            else:
                mcx([ctrl, b[0]], gidney_anc[0], method = "gidney_inv")
        gidney_anc.delete()
    
    for i in jrange(n):
        with control(extract_boolean_digit(a, i)):
            if ctrl is not None:
                cx(ctrl, b[i])
            else:
                x(b[i])
