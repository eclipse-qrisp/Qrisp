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
from qrisp.jasp import make_jaspr
from qrisp.jasp.interpreter_tools.interpreters import extract_post_processing
import numpy as np
import jax.numpy as jnp


def test_basic_post_processing():
    """
    Test basic post-processing extraction with simple arithmetic operations.
    """
    @make_jaspr
    def simple_function(i):
        qv = QuantumFloat(5)
        meas_res = measure(qv[i])
        
        result = meas_res + 1
        return result
    
    jaspr = simple_function(2)
    
    # Extract post-processing function
    post_proc = jaspr.extract_post_processing(2)
    
    # Test with different measurement results (now as an array)
    assert post_proc(jnp.array([False])) == 1  # 0 + 1 = 1
    assert post_proc(jnp.array([True])) == 2   # 1 + 1 = 2


def test_multiple_measurements():
    """
    Test post-processing with multiple measurements and operations.
    """
    @make_jaspr
    def multi_meas_function(index, offset):
        qf = QuantumFloat(5)
        
        meas_1 = measure(qf[index])
        meas_2 = measure(qf[2])
        
        result_1 = meas_1 + offset
        result_2 = meas_2 & (result_1 > 0)
        
        return result_1, result_2
    
    jaspr = multi_meas_function(1, 5)
    
    # Extract post-processing function
    post_proc = jaspr.extract_post_processing(1, 5)
    
    # Test all combinations
    result = post_proc(jnp.array([False, False]))
    assert result[0] == 5 and result[1] == False
    
    result = post_proc(jnp.array([False, True]))
    assert result[0] == 5 and result[1] == True
    
    result = post_proc(jnp.array([True, False]))
    assert result[0] == 6 and result[1] == False
    
    result = post_proc(jnp.array([True, True]))
    assert result[0] == 6 and result[1] == True


def test_boolean_post_processing():
    """
    Test post-processing with boolean operations.
    """
    @make_jaspr
    def boolean_function():
        qb_1 = QuantumBool()
        qb_2 = QuantumBool()
        
        h(qb_1[0])
        
        meas_1 = measure(qb_1)
        meas_2 = measure(qb_2)
        
        # Boolean operations
        result_and = meas_1 & meas_2
        result_or = meas_1 | meas_2
        result_not = ~meas_1
        
        qb_1.delete()
        qb_2.delete()
        
        return result_and, result_or, result_not
    
    jaspr = boolean_function()
    
    # Extract post-processing function
    post_proc = jaspr.extract_post_processing()
    
    # Test truth table
    and_result, or_result, not_result = post_proc(jnp.array([False, False]))
    assert and_result == False and or_result == False and not_result == True
    
    and_result, or_result, not_result = post_proc(jnp.array([False, True]))
    assert and_result == False and or_result == True and not_result == True
    
    and_result, or_result, not_result = post_proc(jnp.array([True, False]))
    assert and_result == False and or_result == True and not_result == False
    
    and_result, or_result, not_result = post_proc(jnp.array([True, True]))
    assert and_result == True and or_result == True and not_result == False


def test_arithmetic_post_processing():
    """
    Test post-processing with various arithmetic operations.
    """
    @make_jaspr
    def arithmetic_function(multiplier):
        qf = QuantumFloat(4)
        
        meas = measure(qf[0])
        
        from jax.lax import convert_element_type
        meas_int = convert_element_type(meas, int)
        
        result_add = meas_int + multiplier
        result_sub = meas_int - multiplier
        result_mul = meas_int * multiplier
        
        return result_add, result_sub, result_mul
    
    jaspr = arithmetic_function(3)
    
    # Extract post-processing function
    post_proc = jaspr.extract_post_processing(3)
    
    # Test with False (0)
    add_res, sub_res, mul_res = post_proc(jnp.array([False]))
    assert add_res == 3 and sub_res == -3 and mul_res == 0
    
    # Test with True (1)
    add_res, sub_res, mul_res = post_proc(jnp.array([True]))
    assert add_res == 4 and sub_res == -2 and mul_res == 3


def test_comparison_post_processing():
    """
    Test post-processing with comparison operations.
    """
    @make_jaspr
    def comparison_function(threshold):
        qf = QuantumFloat(4)
        
        meas = measure(qf[0])
        
        from jax.lax import convert_element_type
        meas_int = convert_element_type(meas, int)
        
        greater = meas_int > threshold
        less = meas_int < threshold
        equal = meas_int == threshold
        
        return greater, less, equal
    
    jaspr = comparison_function(0)
    
    # Extract post-processing function
    post_proc = jaspr.extract_post_processing(0)
    
    # Test with False (0)
    gt, lt, eq = post_proc(jnp.array([False]))
    assert gt == False and lt == False and eq == True
    
    # Test with True (1)
    gt, lt, eq = post_proc(jnp.array([True]))
    assert gt == True and lt == False and eq == False


def test_nested_operations():
    """
    Test post-processing with nested operations.
    """
    @make_jaspr
    def nested_function(a, b):
        qf = QuantumFloat(5)
        
        meas_1 = measure(qf[0])
        meas_2 = measure(qf[1])
        
        from jax.lax import convert_element_type
        m1 = convert_element_type(meas_1, int)
        m2 = convert_element_type(meas_2, int)
        
        # Nested operations
        result = (m1 + a) * (m2 + b)
        
        return result
    
    jaspr = nested_function(2, 3)
    
    # Extract post-processing function
    post_proc = jaspr.extract_post_processing(2, 3)
    
    # Test combinations
    assert post_proc(jnp.array([False, False])) == (0 + 2) * (0 + 3)  # 2 * 3 = 6
    assert post_proc(jnp.array([False, True])) == (0 + 2) * (1 + 3)   # 2 * 4 = 8
    assert post_proc(jnp.array([True, False])) == (1 + 2) * (0 + 3)   # 3 * 3 = 9
    assert post_proc(jnp.array([True, True])) == (1 + 2) * (1 + 3)    # 3 * 4 = 12


def test_single_measurement_multiple_uses():
    """
    Test that a single measurement result can be used multiple times in post-processing.
    """
    @make_jaspr
    def reuse_function(offset):
        qf = QuantumFloat(4)
        
        meas = measure(qf[0])
        
        from jax.lax import convert_element_type
        meas_int = convert_element_type(meas, int)
        
        # Use the same measurement multiple times
        result_1 = meas_int + offset
        result_2 = meas_int * 2
        result_3 = meas_int - offset
        
        return result_1, result_2, result_3
    
    jaspr = reuse_function(10)
    
    # Extract post-processing function
    post_proc = jaspr.extract_post_processing(10)
    
    # Test with False (0)
    r1, r2, r3 = post_proc(jnp.array([False]))
    assert r1 == 10 and r2 == 0 and r3 == -10
    
    # Test with True (1)
    r1, r2, r3 = post_proc(jnp.array([True]))
    assert r1 == 11 and r2 == 2 and r3 == -9


def test_consistency_with_to_qc():
    """
    Test that extract_post_processing produces consistent results with to_qc.
    The quantum circuit and post-processing function should use the same static arguments.
    """
    @make_jaspr
    def consistent_function(index, value):
        qf = QuantumFloat(5)
        
        meas = measure(qf[index])
        
        from jax.lax import convert_element_type
        meas_int = convert_element_type(meas, int)
        
        result = meas_int + value
        
        return result
    
    jaspr = consistent_function(2, 7)
    
    # Both should use the same arguments
    result, qc = jaspr.to_qc(2, 7)
    post_proc = jaspr.extract_post_processing(2, 7)
    
    # Verify quantum circuit has measurements
    has_measurements = any('measure' in str(op).lower() for op in qc.data)
    assert has_measurements
    
    # Verify post-processing works
    assert post_proc(jnp.array([False])) == 7
    assert post_proc(jnp.array([True])) == 8


def test_no_measurements():
    """
    Test that post-processing extraction handles functions with no measurements.
    """
    @make_jaspr
    def no_meas_function(value):
        # Just return a static value, no measurements
        return value * 2
    
    jaspr = no_meas_function(5)
    
    # Extract post-processing function
    post_proc = jaspr.extract_post_processing(5)
    
    # Should return the computed value (no measurement arguments needed)
    result = post_proc(jnp.array([]))
    assert result == 10


def test_direct_function_call():
    """
    Test using extract_post_processing function directly (not as a method).
    """
    @make_jaspr
    def direct_function(a):
        qf = QuantumFloat(3)
        meas = measure(qf[0])
        
        from jax.lax import convert_element_type
        meas_int = convert_element_type(meas, int)
        
        return meas_int + a
    
    jaspr = direct_function(100)
    
    # Use the function directly instead of the method
    post_proc = extract_post_processing(jaspr, 100)
    
    assert post_proc(jnp.array([False])) == 100
    assert post_proc(jnp.array([True])) == 101


def test_multiple_static_arguments():
    """
    Test post-processing extraction with many static arguments.
    """
    @make_jaspr
    def many_args_function(a, b, c, d):
        qf = QuantumFloat(4)
        
        meas = measure(qf[0])
        
        from jax.lax import convert_element_type
        meas_int = convert_element_type(meas, int)
        
        result = meas_int * a + b * c - d
        
        return result
    
    jaspr = many_args_function(2, 3, 4, 5)
    
    # Extract post-processing function
    post_proc = jaspr.extract_post_processing(2, 3, 4, 5)
    
    # Test: meas_int * 2 + 3 * 4 - 5 = meas_int * 2 + 12 - 5 = meas_int * 2 + 7
    assert post_proc(jnp.array([False])) == 0 * 2 + 7  # 7
    assert post_proc(jnp.array([True])) == 1 * 2 + 7   # 9


def test_quantum_gates_before_measurement():
    """
    Test that quantum gates before measurements are properly excluded from post-processing.
    """
    @make_jaspr
    def gates_function(value):
        qb = QuantumBool()
        
        # Apply quantum gates
        h(qb[0])
        x(qb[0])
        h(qb[0])
        
        # Measure
        meas = measure(qb)
        
        # Post-processing
        from jax.lax import convert_element_type
        meas_int = convert_element_type(meas, int)
        result = meas_int * value
        
        qb.delete()
        
        return result
    
    jaspr = gates_function(10)
    
    # Extract post-processing
    post_proc = jaspr.extract_post_processing(10)
    
    # Post-processing should only contain the multiplication
    assert post_proc(jnp.array([False])) == 0
    assert post_proc(jnp.array([True])) == 10


def test_get_size_in_post_processing():
    """
    Test that get_size operations work correctly in post-processing.
    """
    @make_jaspr
    def size_function(i):
        qv = QuantumFloat(i)
        
        # Get the size of the quantum variable
        size = qv.size
        
        # Measure a qubit
        meas = measure(qv[0])
        
        # Use both in post-processing
        from jax.lax import convert_element_type
        meas_int = convert_element_type(meas, int)
        result = meas_int + size
        
        return result, size
    
    jaspr = size_function(1)
    
    # Extract post-processing
    post_proc = jaspr.extract_post_processing(5)
    
    # Test with measurement results
    result, size = post_proc(jnp.array([False]))
    assert result == 5  # 0 + 5
    assert size == 5
    
    result, size = post_proc(jnp.array([True]))
    assert result == 6  # 1 + 5
    assert size == 5
