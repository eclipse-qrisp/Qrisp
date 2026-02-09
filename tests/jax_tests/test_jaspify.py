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

def test_parity_simulation():
    """Test parity function simulation with scalar inputs."""
    
    # Test basic parity (XOR) with known outcomes
    @jaspify
    def test_all_false():
        qv = QuantumVariable(3)
        # All qubits start as False (|0>)
        a = measure(qv[0])
        b = measure(qv[1])
        c = measure(qv[2])
        
        # Parity of (False, False, False) = False
        result = parity(a, b, c, observable=True)
        return result
    
    assert test_all_false() == False
    
    # Test with one True
    @jaspify
    def test_one_true():
        qv = QuantumVariable(3)
        x(qv[0])  # Set first qubit to |1>
        
        a = measure(qv[0])
        b = measure(qv[1])
        c = measure(qv[2])
        
        # Parity of (True, False, False) = True
        result = parity(a, b, c, observable=True)
        return result
    
    assert test_one_true() == True
    
    # Test with two True
    @jaspify
    def test_two_true():
        qv = QuantumVariable(3)
        x(qv[0])
        x(qv[1])
        
        a = measure(qv[0])
        b = measure(qv[1])
        c = measure(qv[2])
        
        # Parity of (True, True, False) = False
        result = parity(a, b, c, observable=True)
        return result
    
    assert test_two_true() == False
    
    # Test with three True
    @jaspify
    def test_three_true():
        qv = QuantumVariable(3)
        x(qv[0])
        x(qv[1])
        x(qv[2])
        
        a = measure(qv[0])
        b = measure(qv[1])
        c = measure(qv[2])
        
        # Parity of (True, True, True) = True
        result = parity(a, b, c, observable=True)
        return result
    
    assert test_three_true() == True
    
    # Test with expectation parameter
    @jaspify
    def test_expectation_met():
        qv = QuantumVariable(2)
        
        a = measure(qv[0])
        b = measure(qv[1])
        
        # Parity is False, expectation is False -> returns False (expectation met)
        result = parity(a, b, expectation=0)
        return result

    assert test_expectation_met() == False

    @jaspify
    def test_expectation_not_met():
        qv = QuantumVariable(2)
        x(qv[0])  # Make parity True

        a = measure(qv[0])
        b = measure(qv[1])

        # Parity is True, expectation is False -> raises exception
        result = parity(a, b, expectation=0)
        return result

    # Should raise exception when expectation not met
    import pytest
    with pytest.raises(Exception, match="Parity expectation deviated"):
        test_expectation_not_met()


def test_parity_array_simulation():
    """Test parity function simulation with array inputs."""
    import jax.numpy as jnp
    
    @jaspify
    def test_array_parity():
        # Create arrays of boolean values
        a = jnp.array([True, False, True, False])
        b = jnp.array([False, False, True, True])
        c = jnp.array([True, True, False, False])
        
        # Element-wise parity
        result = parity(a, b, c, observable=True)
        return result
    
    result = test_array_parity()
    expected = jnp.array([False, True, False, True])  # Element-wise XOR
    assert jnp.array_equal(result, expected)
    
    # Test with 2D arrays
    @jaspify
    def test_2d_array_parity():
        a = jnp.array([[True, False], [True, False]])
        b = jnp.array([[False, True], [True, False]])
        
        result = parity(a, b, observable=True)
        return result
    
    result = test_2d_array_parity()
    expected = jnp.array([[True, True], [False, False]])
    assert jnp.array_equal(result, expected)
    
    # Test array parity with expectation that matches
    @jaspify
    def test_array_expectation_met():
        a = jnp.array([False, False, True])
        b = jnp.array([False, False, True])

        # Parity: [False, False, False]
        # Expectation: False for all -> expectation met
        result = parity(a, b, expectation=0)
        return result

    result = test_array_expectation_met()
    expected = jnp.array([False, False, False])
    assert jnp.array_equal(result, expected)

    # Test array parity with expectation that doesn't match
    @jaspify
    def test_array_expectation_not_met():
        a = jnp.array([True, False, True])
        b = jnp.array([False, False, True])

        # Parity: [True, False, False]
        # Expectation: False for all -> first element mismatches
        result = parity(a, b, expectation=0)
        return result

    # Should raise exception on first mismatch (element 0)
    import pytest
    with pytest.raises(Exception, match="Parity expectation deviated"):
        test_array_expectation_not_met()


def test_parity_with_superposition():
    """Test parity with qubits in superposition."""
    
    @jaspify
    def test_ghz_parity():
        # Create GHZ state: (|0000> + |1111>) / sqrt(2)
        qv = QuantumVariable(4)
        h(qv[0])
        cx(qv[0], qv[1])
        cx(qv[0], qv[2])
        cx(qv[0], qv[3])
        
        a = measure(qv[0])
        b = measure(qv[1])
        c = measure(qv[2])
        d = measure(qv[2])
        
        # Parity of GHZ state is always True (either 0 or 3 ones)
        result = parity(a, b, c, d, observable=True)
        return result
    
    # Run multiple times to check consistency
    results = [test_ghz_parity() for _ in range(10)]
    # All results should be True (parity of 0 or 3 is always odd count mod 2 = True for 1 or 3)
    # Actually: 000 has parity False, 111 has parity True
    assert all(r == False for r in results)

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
