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

from qrisp import *
from qrisp.jasp import *
from jax import make_jaxpr, jit

def test_jasp_simulation():
    
    @jaspify
    def main():
        
        qbl = QuantumBool()
        qf = QuantumFloat(4)
        
        # Bring qbl into superposition
        h(qbl)
        
        # Perform a measure
        cl_bl = measure(qbl)
        
        # Perform a conditional operation based on the measurement outcome
        with control(cl_bl):
            qf[:] = 1
            h(qf[2])
        
        return measure(qf), measure(qbl)

    assert main() in [(1.0, True), (5.0, True), (0.0, False)]
    
    @jaspify
    def main(i, j):
        qf = QuantumFloat(3)
        a = QuantumFloat(3)
        qbl = QuantumBool()
        h(qf[i])
        cx(qf[i], a[j])
        cx(qf[i], qbl[0])
        return measure(qf), measure(a), measure(qbl)

    for i in range(3):
        for j in range(3):
            assert main(i, j) in [(0.0, 0.0, False), (2**i, 2**j, True)]
            
    
    
    @jit
    def cl_inner_function(x):
        return 2*x + jax.numpy.array([1,2,3])[0]

    @jaspify
    def main(i):
        qv = QuantumFloat(4)
        
        qv += cl_inner_function(i)
        
        return measure(qv)


    assert main(4) == 9


def test_jaspify_pytree():
    """Test that jaspify preserves PyTree structure in return values."""
    
    # Test 1: Returning a dictionary
    @jaspify
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
    @jaspify
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
    @jaspify
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
    @jaspify
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
    @jaspify
    def single_return():
        a = QuantumFloat(4)
        a[:] = 11
        return measure(a)
    
    result = single_return()
    assert result == 11, "Should return the scalar value"
    
    # Test 6: Returning dict with superposition (values can vary)
    @jaspify
    def superposition_dict():
        a = QuantumFloat(2)
        h(a[0])  # Creates superposition of 0 and 1
        return {"value": measure(a)}
    
    result = superposition_dict()
    assert isinstance(result, dict), "Should return a dict"
    assert "value" in result, "Should have 'value' key"
    assert result["value"] in [0, 1], "Should be 0 or 1"
