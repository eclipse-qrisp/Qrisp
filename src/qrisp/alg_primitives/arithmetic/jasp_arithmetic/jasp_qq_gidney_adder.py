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

from qrisp.jasp import qache, jrange, AbstractQubit, make_jaspr, Jaspr
from qrisp.core import x, h, cx, t, t_dg, s, measure, cz, mcx, QuantumVariable
from qrisp.qtypes import QuantumBool
from qrisp.environments import control, custom_control
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_cq_gidney_adder import jasp_cq_gidney_adder

@custom_control
def jasp_qq_gidney_adder(a, b, ctrl = None):
    
    n = jnp.min(jnp.array([a.size, b.size]))
    perform_incrementation = n < b.size
    
    gidney_anc = QuantumVariable(n-1, name = "gidney_anc*")
    
    if ctrl is not None:
        ctrl_anc = QuantumBool(name = "gidney_anc_2*")
    
    i = 0
    mcx([a[i], b[i]], gidney_anc[i], method = "gidney")
    
    for j in jrange(n-2):
        i = j+1
    
        cx(gidney_anc[i-1], a[i])
        cx(gidney_anc[i-1], b[i])
        
        mcx([a[i], b[i]], gidney_anc[i], method = "gidney")
        cx(gidney_anc[i-1], gidney_anc[i])
        
    with control(perform_incrementation):
        
        cx(gidney_anc[n-2], a[n-1])
        cx(gidney_anc[n-2], b[n-1])
        
        temp_carry_value = QuantumBool()
        
        mcx([a[n-1], b[n-1]], temp_carry_value[0], method = "gidney")
        
        cx(gidney_anc[n-2], temp_carry_value[0])
        
        ctrl_list = [temp_carry_value]
        
        if ctrl is not None:
            ctrl_list.append(ctrl)
        
        with control(ctrl_list):
            jasp_cq_gidney_adder(1, b[n:])
        
        cx(gidney_anc[n-2], temp_carry_value[0])
        
        mcx([a[n-1], b[n-1]], temp_carry_value[0], method = "gidney_inv")
        
        temp_carry_value.delete()
        
        cx(gidney_anc[n-2], a[n-1])
        cx(gidney_anc[n-2], b[n-1])
    
    cx(gidney_anc[n-2], b[n-1])
    
    if ctrl is not None:
        mcx([ctrl, a[n-1]], ctrl_anc[0], method = "gidney")
        cx(ctrl_anc[0], b[n-1])
        mcx([ctrl, a[n-1]], ctrl_anc[0], method = "gidney_inv")
    else:
        cx(a[n-1], b[n-1])
    
    
    for j in jrange(n-2):
        i = n-j-2
        
        cx(gidney_anc[i-1], gidney_anc[i])
        mcx([a[i], b[i]], gidney_anc[i], method = "gidney_inv")
        
        if ctrl is not None:
            
            mcx([ctrl, a[i]], ctrl_anc[0], method = "gidney")
            cx(ctrl_anc[0], b[i])
            mcx([ctrl, a[i]], ctrl_anc[0], method = "gidney_inv")
            
            cx(gidney_anc[i-1], a[i])
            cx(gidney_anc[i-1], b[i])
        else:
            cx(gidney_anc[i-1], a[i])
            cx(a[i], b[i])

    mcx([a[0], b[0]], gidney_anc[0], method = "gidney_inv")
    if ctrl is not None:
        mcx([ctrl, a[0]], ctrl_anc[0], method = "gidney")
        cx(ctrl_anc[0], b[0])
        mcx([ctrl, a[0]], ctrl_anc[0], method = "gidney_inv")
    else:
        cx(a[0], b[0])
    
    gidney_anc.delete()
    
    if ctrl is not None:
        ctrl_anc.delete()        

