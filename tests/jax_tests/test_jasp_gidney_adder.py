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

def test_jasp_gidney_adder():
    
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
    
    