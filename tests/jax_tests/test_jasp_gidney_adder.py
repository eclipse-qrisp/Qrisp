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

from qrisp import *
from qrisp.jasp import *

def old_test_jasp_gidney_adder():
    
    def call_qq_gidney_adder(i, j, k):
        a = QuantumFloat(i)
        b = QuantumFloat(i)
        b[:] = j
        a[:] = k
        jasp_qq_gidney_adder(a, b)
        return measure(b)
    
    jaspr = make_jaspr(call_qq_gidney_adder)(1, 1, 1)
    
    import jax.numpy as jnp
    for i in range(1, 5):
        for j in range(2**i):
            for k in range(2**i):
                assert jaspr(i,j,k) == (j+k)%(2**i)
    
    # Test adder with differing input sizes
    def call_controlled_gidney_adder(i, j, k):
        
        a = QuantumFloat(i+k)
        a[:] = 5
        b = QuantumFloat(i+j)
        b[:] = 2**i - 1
        
        ctrl_qbl = QuantumBool()
        
        jasp_qq_gidney_adder(a, b)
    
        return measure(b)
    
    jaspr = make_jaspr(call_controlled_gidney_adder)(1,1,1)
    
    for i in range(3, 5):
        for j in range(3):
            for k in range(3):
                assert jaspr(i, j, k) == (5 + 2**i - 1)%2**(i+j)
    
    
    # Test multiple subsequent calls
    def call_qq_gidney_adder(i):
        
        a = QuantumFloat(i)
        a[:] = 2
        b = QuantumFloat(i)
        b[:] = 5
        
        jasp_qq_gidney_adder(a, b)
        jasp_qq_gidney_adder(a, b)
        jasp_qq_gidney_adder(a, b)
    
        return measure(b)
    
    jaspr = make_jaspr(call_qq_gidney_adder)(3)
    jaspr = jaspr.flatten_environments()
    assert jaspr(4) == 11
    
    # Ensure that the Qaching mechanisms keeps only on reference of each function
    # gidney_mcx_jaspr_1 = jaspr.eqns[-5].params["jaxpr"].jaxpr.eqns[-4].params["branches"][1].eqns[-2].params["jaxpr"].jaxpr
    # gidney_mcx_jaspr_2 = jaspr.eqns[-5].params["jaxpr"].jaxpr.eqns[-4].params["branches"][1].eqns[-3].params["body_jaxpr"].eqns[-6].params["jaxpr"].jaxpr
    
    # assert id(gidney_mcx_jaspr_1) == id(gidney_mcx_jaspr_2)
    
    # Test controlled version
    def call_controlled_gidney_adder(i):
        
        a = QuantumFloat(i)
        a[:] = 2
        b = QuantumFloat(i)
        b[:] = 5
        
        
        ctrl_qbl = QuantumBool()
        x(ctrl_qbl[0])
        
        with control(ctrl_qbl[0]):
            jasp_qq_gidney_adder(a, b)
            jasp_qq_gidney_adder(a, b)
        jasp_qq_gidney_adder(a, b)
    
        return measure(b)

    jaspr = make_jaspr(call_controlled_gidney_adder)(3)
    
    assert jaspr(4) == 11
    
    ##############
    # Test semi classical gidney adder
    
    def call_cq_gidney_adder(i, j, k):
        b = QuantumFloat(i)
        qbl = QuantumBool()
        b[:] = j
        a = k
        jasp_cq_gidney_adder(a, b)
        return measure(b)
    
    
    jaspr = make_jaspr(call_cq_gidney_adder)(1, 1, 1)
    
    import jax.numpy as jnp
    for i in range(1, 5):
        for j in range(2**i):
            for k in range(2**i):
                assert jaspr(i,j,k) == (j+k)%(2**i)

    
    def call_cq_gidney_adder(i, ctrl_true):
        b = QuantumFloat(5)
        qbl = QuantumBool()
        with control(ctrl_true):
            qbl.flip()
        b[:] = i
        a = 6
        with control(qbl[0]):
            jasp_cq_gidney_adder(a, b)
        
        return measure(b)
    
    import jax.numpy as jnp
    jaspr = make_jaspr(call_cq_gidney_adder)(1, True)

    for i in range(2**5):
        assert jaspr(i, True) == (i+6)%32    
        assert jaspr(i, False) == i
    
        
    try:
        import catalyst
    except ModuleNotFoundError:
        return
    assert qjit(call_qq_gidney_adder)(4) == 11
    
def test_jasp_gidney_adder():

    ### quantum-quantum
    @boolean_simulation
    def qq(i, j, asize, bsize):
        a = QuantumFloat(asize)
        a[:] = i
        b = QuantumFloat(bsize)
        b[:] = j
        jasp_qq_gidney_adder(a, b)
        return measure(a), measure(b)
    
    for i in range(0, 128, 11):
        for j in range(0, 128, 13):
            for asize in range(2, 6):
                for bsize in range(2, 6):
                    a, b = qq(i, j, asize, bsize)
                    assert a == i%(2**asize)
                    assert b == ((i%(2**asize))+j)%(2**bsize)

    ### quantum-quantum controlled
    @boolean_simulation
    def qq_controlled(i, j, asize, bsize, c):
        a = QuantumFloat(asize)
        a[:] = i
        b = QuantumFloat(bsize)
        b[:] = j
        cq = QuantumFloat(1)
        cq[:] = c
        with control(cq[0]):
            jasp_qq_gidney_adder(a, b)
        return measure(a), measure(b), measure(cq)
    
    for i in range(0, 128, 11):
        for j in range(0, 128, 13):
            for asize in range(2, 6):
                for bsize in range(2, 6):
                    for c in [0, 1]:
                        a, b, cq = qq_controlled(i, j, asize, bsize, c)
                        assert a == i%(2**asize)
                        assert b == ((i%(2**asize))*c+j)%(2**bsize)
                        assert cq == c

    ### classical-quantum
    @boolean_simulation
    def cq(i, j, bsize):
        b = QuantumFloat(bsize)
        b[:] = j
        jasp_cq_gidney_adder(i, b)
        return measure(b)
    
    for i in range(0, 128, 11):
        for j in range(0, 128, 13):
            for bsize in range(2, 6):
                b = cq(i, j, bsize)
                assert b == (i+j)%(2**bsize)

    ### classical-quantum controlled
    @boolean_simulation
    def cq_controlled(i, j, bsize, c):
        b = QuantumFloat(bsize)
        b[:] = j
        cq = QuantumFloat(1)
        cq[:] = c
        with control(cq[0]):
            jasp_cq_gidney_adder(i, b)
        return measure(b), measure(cq)
    
    for i in range(0, 128, 11):
        for j in range(0, 128, 13):
            for bsize in range(2, 6):
                for c in [0, 1]:
                    b, cq = cq_controlled(i, j, bsize, c)
                    assert b == (i*c+j)%(2**bsize)
                    assert cq == c

                        