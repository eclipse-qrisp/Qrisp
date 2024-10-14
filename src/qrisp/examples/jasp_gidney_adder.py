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

from qrisp import *
from qrisp.jasp import *


# Implement Gidney-Adder from https://arxiv.org/abs/1709.06648
# including the real-time computation part
@qache
def gidney_mcx(a, b, c):
    
    h(c)
    t(c)
    
    cx(a, c)
    cx(b, c)
    cx(c, a)
    cx(c, b)
    
    t_dg(a)
    t_dg(b)
    t(c)
    
    cx(c,a)
    cx(c,b)
    
    h(c)
    s(c)

@qache
def gidney_mcx_inv(a, b, c):
    h(c)
    bl = measure(c)
    
    with control(bl):
        cz(a,b)

@qache
def jasp_gidney_adder(a, b):
    
    gidney_anc = QuantumVariable(a.size-1, name = "gidney_anc*")
    
    i = 0
    gidney_mcx(a[i], b[i], gidney_anc[i])
    
    for j in jrange(a.size-2):
        i = j+1
    
        cx(gidney_anc[i-1], a[i])
        cx(gidney_anc[i-1], b[i])
        gidney_mcx(a[i], b[i], gidney_anc[i])
        cx(gidney_anc[i-1], gidney_anc[i])
        
    cx(gidney_anc[a.size-2], b[b.size-1])
    
    for j in jrange(a.size-2):
        i = a.size-j-2
        
        cx(gidney_anc[i-1], gidney_anc[i])
        gidney_mcx_inv(a[i], b[i], gidney_anc[i])
        cx(gidney_anc[i-1], a[i])
        cx(a[i], b[i])
    
    gidney_mcx_inv(a[0], b[0], gidney_anc[0])
    cx(a[0], b[0])
    
    gidney_anc.delete()
    

def call_gidney_adder(i):
    
    a = QuantumFloat(i)
    a[:] = 2
    b = QuantumFloat(i)
    b[:] = 3
    
    # Performs b += a
    jasp_gidney_adder(a, b)
    jasp_gidney_adder(a, b)
    jasp_gidney_adder(a, b)
    return measure(b)

jaspr = make_jaspr(call_gidney_adder)(1)

print(jaspr(5))
print(qjit(call_gidney_adder)(5))