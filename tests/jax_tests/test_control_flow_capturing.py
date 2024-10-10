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

def test_control_flow_capturing():
    
    def test_f(k):
        
        qv = QuantumFloat(k)
        h(qv[0])
        
        for i in jrange(k-1):
            cx(qv[0], qv[i+1])
        
        return measure(qv)
        
    jaspr = make_jaspr(test_f)(1)
    
    for i in range(1, 10):
        assert jaspr(i) in [0, 2**i-1]


    ############
    
    def test_f(i):

        a = QuantumFloat(3)
        a[:] = i
        b = measure(a)
    
        with control(b == 4):
            x(a[0])
    
        return measure(a)

    jaspr = make_jaspr(test_f)(1)
    
    for i in range(8):
        assert jaspr(i) == i + int(i==4)
    
    	
    ################
    @qache
    def int_encoder(qv, encoding_int):
        for i in jrange(qv.size):
            with control(encoding_int & (1<<i)):
                x(qv[i])

    def test_f(a, b):
        
        qv = QuantumFloat(a)
        
        int_encoder(qv, b+1)
        
        return measure(qv)

    jaspr = make_jaspr(test_f)(1,1)
    
    assert jaspr(3, 2) == 3
    assert jaspr(3, 3) == 4
    assert jaspr(3, 5) == 6
