"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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
********************************************************************************
"""

import numpy as np

from qrisp import QuantumFloat, gidney_adder, h, cx, multi_measurement, QuantumBool, control


def test_sc_gidney_adder():
    
    
    for n in range(1, 7):
        for j in range(2**n):
                
            b = QuantumFloat(n)
            c = QuantumFloat(n)
            h(b)
            cx(b,c)
    
            gidney_adder(j, b)
            
            mes_res = multi_measurement([b,c])
            
            for k in mes_res.keys():
                assert (k[1] + j)%(2**n) == k[0]
    
    
        for j in range(2**n):
            b = QuantumFloat(n)
            c = QuantumFloat(n)
            ctrl = QuantumBool()
            h(ctrl)
            h(b)
            cx(b,c)
            with ctrl:
                gidney_adder(j, b)
            
            mes_res = multi_measurement([b, c, ctrl])
            
            for k in mes_res.keys():
                
                if k[2]:
                    assert (k[1] + j)%(2**n) == k[0]
                else:
                    assert k[1] == k[0]
                    
    
    n = 6
    b = QuantumFloat(n)
    d = b.duplicate()

    h(b)
    cx(b,d)
    ctrl = QuantumBool()
    h(ctrl)

    c_in = QuantumBool()
    h(c_in)

    c_out = QuantumBool()

    with control(ctrl):
        gidney_adder(7, b, c_in = c_in, c_out = c_out)
    
    mes_res = multi_measurement([b,d,ctrl,c_in,c_out])
    for b, d, ctrl, c_in, c_out in mes_res.keys():
        
        if ctrl:
            assert (d + 7 + int(c_in))%(2**n) == b
            assert ((d + 7 + int(c_in)) >= 2**n) == c_out
        else:
            assert d == b
            assert c_out == False
            
    
    
