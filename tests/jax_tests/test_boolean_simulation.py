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

from qrisp import QuantumFloat, measure
from qrisp.jasp import boolean_simulation, jrange
from jax import lax
import jax.numpy as jnp

def test_boolean_simulation():
    
    @boolean_simulation
    def main(i, j):
        
        a = QuantumFloat(10)
        
        b = QuantumFloat(10)
        
        a[:] = i
        b[:] = j
        
        c = QuantumFloat(30)
        
        for i in jrange(150): 
            c += a*b
        
        return measure(c)
    
    for i in range(5):
        for j in range(5):
            assert main(i, j) == 150*i*j
            
    # Test multi switch
    
    @boolean_simulation
    def main():

        def case0(x):
            return x + 1

        def case1(x):
            return x + 2

        def case2(x):
            return x + 3
        
        def case3(x):
            return x + 4

        def compute(index, x):
            return lax.switch(index, [case0, case1, case2, case3], x)


        qf = QuantumFloat(2)
        qf[:] = 3
        ind = jnp.int8(measure(qf))

        res = compute(ind,jnp.int32(0))

        return ind, res


    assert main() == (3,4)