"""
********************************************************************************
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
********************************************************************************
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
    
    ## Test qubit array fusion
    
    @boolean_simulation
    def main():
        
        a = QuantumFloat(3)
        b = QuantumFloat(3)
        a[:] = 7
        b[:] = 7
        
        return measure(a.reg + b.reg)
    
    assert main() == 63
        
    @boolean_simulation
    def main():
        
        a = QuantumFloat(3)
        b = QuantumFloat(3)
        a[:] = 7
        b[:] = 7
        
        return measure(a.reg + b[0])
    
    assert main() == 15
    
    @boolean_simulation
    def main():
        
        a = QuantumFloat(3)
        b = QuantumFloat(3)
        a[:] = 7
        b[:] = 7
        
        return measure(a[0] + b.reg)
    
    assert main() == 15
    
    @boolean_simulation
    def main():
        
        a = QuantumFloat(3)
        b = QuantumFloat(3)
        a[:] = 7
        b[:] = 7
        
        return measure(a[0] + b[0])
    
    assert main() == 3


def test_boolean_simulation_pytree():
    """Test that boolean_simulation preserves PyTree structure in return values."""
    
    # Test 1: Returning a dictionary
    @boolean_simulation
    def dict_return():
        a = QuantumFloat(4)
        b = QuantumFloat(4)
        a[:] = 5
        b[:] = 3
        return {"first": measure(a), "second": measure(b)}
    
    result = dict_return()
    assert isinstance(result, dict), "Should return a dict"
    assert "first" in result and "second" in result, "Should have correct keys"
    assert result["first"] == 5, "first should be 5"
    assert result["second"] == 3, "second should be 3"
    
    # Test 2: Returning a list
    @boolean_simulation
    def list_return():
        a = QuantumFloat(4)
        b = QuantumFloat(4)
        c = QuantumFloat(4)
        a[:] = 1
        b[:] = 2
        c[:] = 3
        return [measure(a), measure(b), measure(c)]
    
    result = list_return()
    assert isinstance(result, list), "Should return a list"
    assert len(result) == 3, "Should have 3 elements"
    assert result == [1, 2, 3], "Should have correct values"
    
    # Test 3: Returning a tuple (should still work as before)
    @boolean_simulation
    def tuple_return():
        a = QuantumFloat(4)
        b = QuantumFloat(4)
        a[:] = 7
        b[:] = 9
        return (measure(a), measure(b))
    
    result = tuple_return()
    assert isinstance(result, tuple), "Should return a tuple"
    assert result == (7, 9), "Should have correct values"
    
    # Test 4: Returning nested structure
    @boolean_simulation
    def nested_return():
        a = QuantumFloat(4)
        b = QuantumFloat(4)
        c = QuantumFloat(4)
        a[:] = 1
        b[:] = 2
        c[:] = 3
        return {
            "pair": (measure(a), measure(b)),
            "single": measure(c)
        }
    
    result = nested_return()
    assert isinstance(result, dict), "Should return a dict"
    assert isinstance(result["pair"], tuple), "Nested tuple should be preserved"
    assert result["pair"] == (1, 2), "pair should be (1, 2)"
    assert result["single"] == 3, "single should be 3"
    
    # Test 5: Single return value (scalar)
    @boolean_simulation
    def single_return():
        a = QuantumFloat(4)
        a[:] = 11
        return measure(a)
    
    result = single_return()
    # Single values should be returned as-is
    assert result == 11, "Should return the scalar value"
    
    # Test 6: No return value
    @boolean_simulation
    def no_return():
        a = QuantumFloat(4)
        a[:] = 5
        a.delete()
    
    result = no_return()
    assert result is None, "Should return None"
    
    # Test 7: Returning dict with computed values
    @boolean_simulation
    def computed_dict(x, y):
        a = QuantumFloat(8)
        b = QuantumFloat(8)
        a[:] = x
        b[:] = y
        c = a + b
        d = a * b
        return {"sum": measure(c), "product": measure(d)}
    
    result = computed_dict(3, 4)
    assert result["sum"] == 7, "sum should be 7"
    assert result["product"] == 12, "product should be 12"