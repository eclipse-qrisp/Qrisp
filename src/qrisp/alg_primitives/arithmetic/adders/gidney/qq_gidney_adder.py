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

from qrisp.core import QuantumVariable
from qrisp.qtypes import QuantumBool
from qrisp.environments import invert
from qrisp.alg_primitives import mcx, cx

# This function performs the Gidney adder from https://arxiv.org/pdf/1709.06648.pdf
def qq_gidney_adder(a, b, c_in = None, c_out = None):
    
    if len(a) != len(b):
        raise Exception("Tried to call Gidney adder with inputs of unequal length")

    if c_out is not None:
        # Convert to qubit if neccessary
        if isinstance(c_out, QuantumBool):
            c_out = c_out[0]
        
        
        b = list(b) + [c_out]
    
    if len(b) == 1:
        cx(a[0],b[0])
        if c_in is not None:
            if isinstance(c_in, QuantumBool):
                c_in = c_in[0]
            
            
            cx(c_in, b[0])
        return
    
    gidney_anc = QuantumVariable(len(b) - 1, name ="gidney_anc*", qs = b[0].qs())
    
    for i in range(len(b)-1):
        
        if i != 0:
            mcx([a[i], b[i]], gidney_anc[i], method = "gidney")
            cx(gidney_anc[i-1], gidney_anc[i])
        elif c_in is not None:
            cx(c_in, b[i])
            cx(c_in, a[i])
            mcx([a[i], b[i]], gidney_anc[i], method = "gidney")
            cx(c_in, gidney_anc[i])
        else:
            mcx([a[i], b[i]], gidney_anc[i], method = "gidney")
            
        if i != len(b) -2:
            cx(gidney_anc[i], b[i+1])
            cx(gidney_anc[i], a[i+1])
    
    cx(gidney_anc[-1], b[-1])
    
    with invert():
        
        for i in range(len(b)-1):
            
            if i != 0:
                mcx([a[i], b[i]], gidney_anc[i], method = "gidney")
                cx(gidney_anc[i-1], gidney_anc[i])
            elif c_in is not None:
                cx(c_in, a[i])
                mcx([a[i], b[i]], gidney_anc[i], method = "gidney")
                cx(c_in, gidney_anc[i])
            else:
                mcx([a[i], b[i]], gidney_anc[i], method = "gidney")
                
             
            if i != len(b) -2:
                cx(gidney_anc[i], a[i+1])
                
    for i in range(len(a)):
        cx(a[i], b[i])
    
    gidney_anc.delete(verify = False)
