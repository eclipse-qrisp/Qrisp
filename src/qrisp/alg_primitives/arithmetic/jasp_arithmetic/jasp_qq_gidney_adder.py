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

from qrisp.jasp import qache, jrange, AbstractQubit, make_jaspr, Jaspr
from qrisp.core import x, h, cx, t, t_dg, s, measure, cz, mcx, QuantumVariable
from qrisp.qtypes import QuantumBool
from qrisp.environments import control, custom_control


@custom_control
def jasp_qq_gidney_adder(a, b, ctrl = None):
    
    gidney_anc = QuantumVariable(a.size-1, name = "gidney_anc*")
    
    if ctrl is not None:
        ctrl_anc = QuantumBool(name = "gidney_anc_2*")
    
    i = 0
    mcx([a[i], b[i]], gidney_anc[i], method = "gidney")
    
    for j in jrange(a.size-2):
        i = j+1
    
        cx(gidney_anc[i-1], a[i])
        cx(gidney_anc[i-1], b[i])
        mcx([a[i], b[i]], gidney_anc[i], method = "gidney")
        cx(gidney_anc[i-1], gidney_anc[i])
        
    cx(gidney_anc[a.size-2], b[b.size-1])
    
    for j in jrange(a.size-2):
        i = a.size-j-2
        
        cx(gidney_anc[i-1], gidney_anc[i])
        mcx([a[i], b[i]], gidney_anc[i], method = "gidney_inv")
        
        if ctrl is not None:
            
            mcx([ctrl, a[i]], ctrl_anc[0], method = "gidney")
            # mcx([ctrl, a[i]], ctrl_anc[0], method = "gray")
            cx(ctrl_anc[0], b[i])
            mcx([ctrl, a[i]], ctrl_anc[0], method = "gidney_inv")
            # mcx([ctrl, a[i]], ctrl_anc[0], method = "gray")
            
            cx(gidney_anc[i-1], a[i])
            cx(gidney_anc[i-1], b[i])
        else:
            cx(gidney_anc[i-1], a[i])
            cx(a[i], b[i])
    
    mcx([a[0], b[0]], gidney_anc[0], method = "gidney_inv")
    cx(a[0], b[0])
    
    gidney_anc.delete()
    
    if ctrl is not None:
        ctrl_anc.delete()        

