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
    
    # Extract post-processing function (bitstring input by default)
    post_proc = jaspr.extract_post_processing(2)
    
    # Test with bitstring input
    assert post_proc("0") == 1  # 0 + 1 = 1
    assert post_proc("1") == 2   # 1 + 1 = 2


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
    result = post_proc("00")  # qf[1]=0, qf[2]=0
    assert result[0] == 5 and result[1] == False
    
    result = post_proc("10")  # qf[1]=1, qf[2]=0
    assert result[0] == 6 and result[1] == False
    
    result = post_proc("01")  # qf[1]=0, qf[2]=1
    assert result[0] == 5 and result[1] == True
    
    result = post_proc("11")  # qf[1]=1, qf[2]=1
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
    and_result, or_result, not_result = post_proc("00")
    assert and_result == False and or_result == False and not_result == True
    
    and_result, or_result, not_result = post_proc("01")  # qb_1=0, qb_2=1
    assert and_result == False and or_result == True and not_result == True
    
    and_result, or_result, not_result = post_proc("10")  # qb_1=1, qb_2=0
    assert and_result == False and or_result == True and not_result == False
    
    and_result, or_result, not_result = post_proc("11")
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
    add_res, sub_res, mul_res = post_proc("0")
    assert add_res == 3 and sub_res == -3 and mul_res == 0
    
    # Test with True (1)
    add_res, sub_res, mul_res = post_proc("1")
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
    gt, lt, eq = post_proc("0")
    assert gt == False and lt == False and eq == True
    
    # Test with True (1)
    gt, lt, eq = post_proc("1")
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
    assert post_proc("00") == (0 + 2) * (0 + 3)  # 2 * 3 = 6
    assert post_proc("10") == (1 + 2) * (0 + 3)   # 3 * 3 = 9
    assert post_proc("01") == (0 + 2) * (1 + 3)   # 2 * 4 = 8
    assert post_proc("11") == (1 + 2) * (1 + 3)    # 3 * 4 = 12


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
    r1, r2, r3 = post_proc("0")
    assert r1 == 10 and r2 == 0 and r3 == -10
    
    # Test with True (1)
    r1, r2, r3 = post_proc("1")
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
    assert post_proc("0") == 7
    assert post_proc("1") == 8


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
    result = post_proc("")
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
    
    assert post_proc("0") == 100
    assert post_proc("1") == 101


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
    assert post_proc("0") == 0 * 2 + 7  # 7
    assert post_proc("1") == 1 * 2 + 7   # 9


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
    assert post_proc("0") == 0
    assert post_proc("1") == 10


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
    result, size = post_proc("0")
    assert result == 5  # 0 + 5
    assert size == 5
    
    result, size = post_proc("1")
    assert result == 6  # 1 + 5
    assert size == 5


def test_slice_in_post_processing():
    """
    Test that QubitArray slicing works correctly in post-processing.
    """
    @make_jaspr
    def slice_function():
        qv = QuantumFloat(10)
        
        # Slice the quantum variable
        sliced = qv[2:7]  # Size should be 7 - 2 = 5
        size = sliced.size
        
        # Measure a qubit
        meas = measure(qv[0])
        
        # Use size in post-processing
        from jax.lax import convert_element_type
        meas_int = convert_element_type(meas, int)
        result = meas_int * size
        
        return result, size
    
    jaspr = slice_function()
    
    # Extract post-processing
    post_proc = jaspr.extract_post_processing()
    
    # Test with measurement results
    result, size = post_proc("0")
    assert result == 0  # 0 * 5
    assert size == 5
    
    result, size = post_proc("1")
    assert result == 5  # 1 * 5
    assert size == 5


def test_fuse_in_post_processing():
    """
    Test that QubitArray fusing works correctly in post-processing.
    """
    @make_jaspr
    def fuse_function():
        qv1 = QuantumFloat(3)
        qv2 = QuantumFloat(4)
        
        # Fuse the quantum variables
        fused = qv1[:] + qv2[:]  # Size should be 3 + 4 = 7
        size = fused.size
        
        # Measure qubits
        meas1 = measure(qv1[0])
        meas2 = measure(qv2[0])
        
        # Use size in post-processing
        from jax.lax import convert_element_type
        meas1_int = convert_element_type(meas1, int)
        meas2_int = convert_element_type(meas2, int)
        result = (meas1_int + meas2_int) * size
        
        return result, size
    
    jaspr = fuse_function()
    
    # Extract post-processing
    post_proc = jaspr.extract_post_processing()
    
    # Test with measurement results
    result, size = post_proc("00")
    assert result == 0  # (0 + 0) * 7
    assert size == 7
    
    result, size = post_proc("01")
    assert result == 7  # (1 + 0) * 7
    assert size == 7
    
    result, size = post_proc("10")
    assert result == 7  # (0 + 1) * 7
    assert size == 7
    
    result, size = post_proc("11")
    assert result == 14  # (1 + 1) * 7
    assert size == 7


def test_bitstring_input():
    """
    Test that bitstring input works correctly (default behavior).
    """
    @make_jaspr
    def bitstring_function(i, j):
        qv = QuantumFloat(5)
        return measure(qv[i]) + 1, measure(qv[j])
    
    jaspr = bitstring_function(1, 2)
    
    # Extract post-processing (bitstring input is default)
    post_proc = jaspr.extract_post_processing(1, 2)
    
    result = post_proc("00")  # qv[1]=0, qv[2]=0
    assert result == (1, False)  # 0+1=1, False
    
    result = post_proc("10")  # qv[1]=1, qv[2]=0
    assert result == (2, False)  # 1+1=2, False
    
    result = post_proc("01")  # qv[1]=0, qv[2]=1
    assert result == (1, True)  # 0+1=1, True
    
    result = post_proc("11")  # qv[1]=1, qv[2]=1
    assert result == (2, True)  # 1+1=2, True
    
    # Test with bitstring "00"
    result = post_proc("00")
    assert result == (1, False)  # meas[0]=False -> 0+1=1, meas[1]=False


def test_bitstring_vs_array_equivalence():
    """
    Test that bitstring and array inputs produce the same results.
    """
    @make_jaspr
    def test_function():
        qv = QuantumFloat(3)
        meas1 = measure(qv[0])
        meas2 = measure(qv[1])
        meas3 = measure(qv[2])
        
        from jax.lax import convert_element_type
        m1_int = convert_element_type(meas1, int)
        m2_int = convert_element_type(meas2, int)
        m3_int = convert_element_type(meas3, int)
        
        return m1_int + 2*m2_int + 4*m3_int
    
    jaspr = test_function()
    
    # Create both versions
    post_proc_bitstring = jaspr.extract_post_processing()  # Works with both string and array
    post_proc_array = jaspr.extract_post_processing()
    
    # Test all combinations
    for bits in ["000", "001", "010", "011", "100", "101", "110", "111"]:
        array_input = jnp.array([c == '1' for c in bits], dtype=bool)
        result_bitstring = post_proc_bitstring(bits)
        result_array = post_proc_array(array_input)
        assert result_array == result_bitstring, f"Mismatch for {bits}: {result_array} != {result_bitstring}"


def test_array_input_jittable():
    """
    Test that the post-processing function works with JAX jit.
    """
    import jax
    
    @make_jaspr
    def test_function():
        qv = QuantumFloat(3)
        meas1 = measure(qv[0])
        meas2 = measure(qv[1])
        
        from jax.lax import convert_element_type
        m1_int = convert_element_type(meas1, int)
        m2_int = convert_element_type(meas2, int)
        
        return m1_int + 2*m2_int
    
    jaspr = test_function()
    
    # Extract with array input
    post_proc = jaspr.extract_post_processing()
    
    # Jit the post-processing function
    jitted_post_proc = jax.jit(post_proc)
    
    # Test that jitted version works
    test_input = jnp.array([False, True])
    result = jitted_post_proc(test_input)
    expected = post_proc(test_input)
    assert result == expected
    
    # Test with different input
    
    test_input2 = jnp.array([True, False])
    result2 = jitted_post_proc(test_input2)
    assert result2 == 1  # 1 + 2*0 = 1


def test_jit_post_processor_with_jitted_subroutines():
    """
    Test that jitting a post-processor works when it contains jitted subroutines.
    This is important because QuantumFloat decoders use jax.jit internally for
    non-trivial exponents.
    """
    import jax
    
    @make_jaspr
    def test_with_jitted_decoder():
        # QuantumFloat with exponent=-1 uses a jitted decoder internally
        qf = QuantumFloat(3, exponent=-1)
        h(qf[0])
        h(qf[1])
        
        result = measure(qf)
        return result
    
    jaspr = test_with_jitted_decoder()
    
    # Extract post-processing
    post_proc = jaspr.extract_post_processing()
    
    # Jit the post-processing function
    jitted_post_proc = jax.jit(post_proc)
    
    # Test various measurement results
    # Note: bit order is bit[0]=MSB, bit[1]=middle, bit[2]=LSB
    # value = (4*bit[0] + 2*bit[1] + bit[2]) * 0.5
    
    # "000" -> 0 * 0.5 = 0.0
    test_input = jnp.array([False, False, False])
    result = jitted_post_proc(test_input)
    expected = 0.0
    assert abs(float(result) - expected) < 0.01, f"Expected {expected}, got {result}"
    
    # "100" -> 4 * 0.5 = 2.0
    test_input = jnp.array([False, False, True])
    result = jitted_post_proc(test_input)
    expected = 2.0
    assert abs(float(result) - expected) < 0.01, f"Expected {expected}, got {result}"
    
    # "010" -> 2 * 0.5 = 1.0
    test_input = jnp.array([False, True, False])
    result = jitted_post_proc(test_input)
    expected = 1.0
    assert abs(float(result) - expected) < 0.01, f"Expected {expected}, got {result}"
    
    # "001" -> 1 * 0.5 = 0.5
    test_input = jnp.array([True, False, False])
    result = jitted_post_proc(test_input)
    expected = 0.5
    assert abs(float(result) - expected) < 0.01, f"Expected {expected}, got {result}"
    
    # "111" -> 7 * 0.5 = 3.5
    test_input = jnp.array([True, True, True])
    result = jitted_post_proc(test_input)
    expected = 3.5
    assert abs(float(result) - expected) < 0.01, f"Expected {expected}, got {result}"


def test_jit_post_processor_with_qached_subroutines():
    """
    Test that jitting a post-processor works when it contains qached subroutines.
    Qached functions are the quantum equivalent of jit - they cache quantum circuits
    and are called with pjit in the Jaspr.
    """
    import jax
    from qrisp.jasp import qache
    
    # Create a qached subroutine
    @qache
    def qached_operation(qv):
        h(qv[0])
        cx(qv[0], qv[1])
        result = measure(qv[0])
        return result
    
    @make_jaspr
    def test_with_qached():
        qv1 = QuantumVariable(2)
        qv2 = QuantumVariable(2)
        
        # Call qached function multiple times
        res1 = qached_operation(qv1)
        res2 = qached_operation(qv2)
        
        # Combine results
        from jax.lax import convert_element_type
        r1_int = convert_element_type(res1, int)
        r2_int = convert_element_type(res2, int)
        
        return r1_int + 2*r2_int
    
    jaspr = test_with_qached()
    
    # Extract post-processing
    post_proc = jaspr.extract_post_processing()
    
    # Jit the post-processing function
    jitted_post_proc = jax.jit(post_proc)
    
    # Test various combinations
    # Circuit measures: res1 (from qv1[0]) then res2 (from qv2[0])
    # Measurement order: [res1, res2] - chronological order (first measurement first)
    
    # Both measurements False: array[0]=res1=False, array[1]=res2=False
    test_input = jnp.array([False, False])
    result = jitted_post_proc(test_input)
    assert result == 0, f"Expected 0, got {result}"
    
    # res1=True, res2=False: array[0]=res1=True, array[1]=res2=False
    test_input = jnp.array([True, False])
    result = jitted_post_proc(test_input)
    assert result == 1, f"Expected 1, got {result}"
    
    # res1=False, res2=True: array[0]=res1=False, array[1]=res2=True
    test_input = jnp.array([False, True])
    result = jitted_post_proc(test_input)
    assert result == 2, f"Expected 2, got {result}"
    
    # Both True: array[0]=res1=True, array[1]=res2=True
    test_input = jnp.array([True, True])
    result = jitted_post_proc(test_input)
    assert result == 3, f"Expected 3, got {result}"


def test_jit_post_processor_with_mixed_subroutines():
    """
    Test jitting a post-processor with both jitted (QuantumFloat decoder) and
    qached subroutines. This is the most comprehensive test case.
    """
    import jax
    from qrisp.jasp import qache
    
    # Create a qached subroutine
    @qache
    def prepare_state(qv):
        h(qv[0])
        return measure(qv[0])
    
    @make_jaspr
    def test_mixed():
        # Use qached function
        qv = QuantumVariable(2)
        bit_result = prepare_state(qv)
        
        # Use QuantumFloat with non-trivial exponent (uses jitted decoder)
        qf = QuantumFloat(2, exponent=-1)
        h(qf[0])
        float_result = measure(qf)
        
        # Combine both results
        from jax.lax import convert_element_type
        bit_int = convert_element_type(bit_result, int)
        
        return bit_int + float_result
    
    jaspr = test_mixed()
    
    # Extract post-processing
    post_proc = jaspr.extract_post_processing()
    
    # Jit the post-processing function
    jitted_post_proc = jax.jit(post_proc)
    
    # Test various combinations
    # Measurement order: qv[0] (bit), qf[0], qf[1] - chronological order
    # qf value = (qf[0] << 0 | qf[1] << 1) * 0.5

    # array[0]=qv[0]=False (bit=0), array[1]=qf[0]=False, array[2]=qf[1]=False: qf_val=0.0 -> 0 + 0.0 = 0.0
    test_input = jnp.array([False, False, False, False])
    result = jitted_post_proc(test_input)
    expected = 0.0
    assert abs(float(result) - expected) < 0.01, f"Expected {expected}, got {result}"
    
    # array[0]=qv[0]=True (bit=1), array[1]=qf[0]=False, array[2]=qf[1]=False: qf_val=0.0 -> 1 + 0.0 = 1.0
    test_input = jnp.array([True, False, False, False])
    result = jitted_post_proc(test_input)
    expected = 1.0
    assert abs(float(result) - expected) < 0.01, f"Expected {expected}, got {result}"
    
    # array[0]=qv[0]=False (bit=0), array[1]=qf[0]=True, array[2]=qf[1]=False: qf_val=0.5 -> 0 + 0.5 = 0.5
    test_input = jnp.array([False, True, False, False])
    result = jitted_post_proc(test_input)
    expected = 0.5
    assert abs(float(result) - expected) < 0.01, f"Expected {expected}, got {result}"
    
    # array[0]=qv[0]=False (bit=0), array[1]=qf[0]=False, array[2]=qf[1]=True: qf_val=1.0 -> 0 + 1.0 = 1.0
    test_input = jnp.array([False, False, True, False])
    result = jitted_post_proc(test_input)
    expected = 1.0
    assert abs(float(result) - expected) < 0.01, f"Expected {expected}, got {result}"
    
    # array[0]=qv[0]=True (bit=1), array[1]=qf[0]=True, array[2]=qf[1]=True: qf_val=1.5 -> 1 + 1.5 = 2.5
    test_input = jnp.array([True, True, True, False])
    result = jitted_post_proc(test_input)
    expected = 2.5
    assert abs(float(result) - expected) < 0.01, f"Expected {expected}, got {result}"


def test_cond_primitive():
    """
    Test that the cond (conditional) primitive works with post-processing.
    """
    import jax
    
    @make_jaspr
    def test_with_cond():
        qv = QuantumFloat(2)
        h(qv[0])
        
        # Measure to get a boolean
        meas = measure(qv[0])
        
        # Use conditional based on measurement
        from jax.lax import cond
        
        def true_branch(x):
            return x + 10
        
        def false_branch(x):
            return x + 20
        
        result = cond(meas, true_branch, false_branch, 5)
        
        return result
    
    jaspr = test_with_cond()
    
    # Extract post-processing
    post_proc = jaspr.extract_post_processing()
    
    # Circuit measures: qv[0] (single measurement)
    # QuantumFloat(2) has 2 qubits but only qv[0] is measured
    # Measurement order: qv[0] at array[0] (additional array elements are ignored)
    
    # Test with meas=False -> false_branch(5) = 25
    result = post_proc(jnp.array([False, False]))  # array[0]=qv[0]=False (extra elements ignored)
    assert result == 25, f"Expected 25, got {result}"
    
    # Test with meas=True -> true_branch(5) = 15
    result = post_proc(jnp.array([True, False]))  # array[0]=qv[0]=True (extra elements ignored)
    assert result == 15, f"Expected 15, got {result}"
    
    # Test that it can be jitted
    jitted_post_proc = jax.jit(post_proc)
    result = jitted_post_proc(jnp.array([False, False]))
    assert result == 25, f"Jitted: Expected 25, got {result}"
    
    result = jitted_post_proc(jnp.array([True, False]))
    assert result == 15, f"Jitted: Expected 15, got {result}"


def test_while_primitive():
    """
    Test that the while loop primitive works with post-processing.
    """
    import jax
    
    @make_jaspr
    def test_with_while():
        qv = QuantumFloat(3)
        h(qv[0])
        h(qv[1])
        
        # Measure two qubits
        meas1 = measure(qv[0])
        meas2 = measure(qv[1])
        
        # Convert to int
        from jax.lax import convert_element_type, while_loop
        m1 = convert_element_type(meas1, int)
        m2 = convert_element_type(meas2, int)
        
        # Count how many bits are set using while loop
        def cond_fun(carry):
            counter, remaining = carry
            return remaining > 0
        
        def body_fun(carry):
            counter, remaining = carry
            return (counter + 1, remaining - 1)
        
        # Start with sum of measurements
        initial = (0, m1 + m2)
        final_counter, _ = while_loop(cond_fun, body_fun, initial)
        
        return final_counter
    
    jaspr = test_with_while()
    
    # Extract post-processing
    post_proc = jaspr.extract_post_processing()
    
    result = post_proc(jnp.array([False, False, False]))
    assert result == 0, f"Expected 0, got {result}"
    
    result = post_proc(jnp.array([True, False, False]))
    assert result == 1, f"Expected 1, got {result}"
    
    result = post_proc(jnp.array([False, True, False]))
    assert result == 1, f"Expected 1, got {result}"
    
    result = post_proc(jnp.array([True, True, False]))
    assert result == 2, f"Expected 2, got {result}"
    
    # Test that it can be jitted
    jitted_post_proc = jax.jit(post_proc)
    result = jitted_post_proc(jnp.array([True, True, False]))
    assert result == 2, f"Jitted: Expected 2, got {result}"


def test_terminal_sampling_comparison():
    """
    Compare post-processing extraction with terminal sampling across various algorithms.
    
    This test verifies that post-processing extraction produces equivalent results
    to terminal sampling by:
    1. Running algorithms with terminal_sampling decorator
    2. Running algorithms with make_jaspr, extracting circuit, and executing
    3. Applying post-processing to circuit outputs
    4. Comparing the resulting distributions
    """
    
    # Helper function to convert bitstring to bool array
    def bitstring_to_array(bitstring):
        """Convert bitstring to JAX array of booleans.
        
        Converts each character in the bitstring to a boolean value.
        The post_processing_func consumes measurements in chronological order
        (first measurement at array[0], second at array[1], etc.).
        """
        return jnp.array([c == '1' for c in bitstring])
    
    # Helper function to compare post-processed distributions
    def compare_post_processed_distributions(qc_results, post_proc, terminal_results, tolerance=0.1):
        """
        Compare post-processed circuit results with terminal sampling results.
        
        This applies post-processing to each circuit output and builds a distribution
        of post-processed results, then compares with terminal sampling.
        """
        total_shots = sum(qc_results.values())
        
        # Build distribution of post-processed results
        post_proc_dist = {}
        for bitstring, count in qc_results.items():
            bit_array = bitstring_to_array(bitstring)
            result = post_proc(bit_array)
            
            # Convert result to hashable key
            # Handle single values, tuples, and arrays
            if isinstance(result, (tuple, list)):
                # Convert arrays to basic types for hashing
                key = tuple(float(x) if hasattr(x, 'item') else x for x in result)
            elif hasattr(result, 'item'):
                key = float(result)
            else:
                key = result
                
            if key not in post_proc_dist:
                post_proc_dist[key] = 0
            post_proc_dist[key] += count
        
        # Normalize to probabilities
        for key in post_proc_dist:
            post_proc_dist[key] /= total_shots
        
        # Compare distributions
        # Check all post-processed outcomes appear in terminal results
        for key, prob in post_proc_dist.items():
            # For single value results, terminal might have it as first element of tuple
            terminal_key = key
            if terminal_key not in terminal_results:
                # Try extracting first element if key is tuple
                if isinstance(key, tuple) and len(key) > 0:
                    terminal_key = key[0]
            
            if terminal_key in terminal_results:
                terminal_prob = terminal_results[terminal_key]
                diff = abs(prob - terminal_prob)
                if diff > tolerance:
                    return False, f"Mismatch for {key}: post_proc={prob:.3f}, terminal={terminal_prob:.3f}"
            elif prob > tolerance:
                return False, f"Unexpected post-processed outcome {key} with prob {prob:.3f}"
        
        return True, "Distributions match"
    
    # Test 1: Superposition State Analysis
    def superposition_algorithm():
        qf = QuantumFloat(2, signed=False)
        h(qf)
        return qf
    
    superposition_terminal = terminal_sampling(superposition_algorithm)
    
    @make_jaspr
    def superposition_post_proc():
        qf = superposition_algorithm()
        m = measure(qf)
        
        from jax.lax import convert_element_type
        val = convert_element_type(m, int)
        is_even = (val % 2 == 0)
        
        return val, is_even
    
    # Get terminal sampling results
    terminal_results = superposition_terminal()
    
    # Get circuit and run it
    jaspr = superposition_post_proc()
    result = jaspr.to_qc()
    qc = result[-1]
    qc_results = qc.run(shots=1000)
    
    # Extract post-processing function
    post_proc = jaspr.extract_post_processing()
    
    # Compare distributions
    match, msg = compare_post_processed_distributions(qc_results, post_proc, terminal_results)
    assert match, f"Superposition test - {msg}"
    
    # Test 2: Bell State Parity Check
    def bell_state_algorithm():
        q1 = QuantumBool()
        q2 = QuantumBool()
        h(q1)
        cx(q1, q2)
        return q1, q2
    
    bell_state_terminal = terminal_sampling(bell_state_algorithm)
    
    @make_jaspr
    def bell_state_post_proc():
        q1, q2 = bell_state_algorithm()
        m1 = measure(q1)
        m2 = measure(q2)
        return m1, m2
    
    terminal_results = bell_state_terminal()
    jaspr = bell_state_post_proc()
    result = jaspr.to_qc()
    qc = result[-1]
    qc_results = qc.run(shots=1000)
    post_proc = jaspr.extract_post_processing()
    
    match, msg = compare_post_processed_distributions(qc_results, post_proc, terminal_results)
    assert match, f"Bell state test - {msg}"
    
    # Verify Bell state property: both qubits should be same
    for bitstring in list(qc_results.keys())[:2]:
        bits = bitstring_to_array(bitstring)
        result = post_proc(bits)
        assert result[0] == result[1], f"Bell state violation: {result}"
    
    # Test 3: CNOT Gate
    def cnot_algorithm():
        ctrl = QuantumBool()
        target = QuantumBool()
        h(ctrl)
        cx(ctrl, target)
        return ctrl, target
    
    cnot_terminal = terminal_sampling(cnot_algorithm)
    
    @make_jaspr
    def cnot_post_proc():
        ctrl, target = cnot_algorithm()
        m_ctrl = measure(ctrl)
        m_target = measure(target)
        return m_ctrl, m_target
    
    terminal_results = cnot_terminal()
    jaspr = cnot_post_proc()
    result = jaspr.to_qc()
    qc = result[-1]
    qc_results = qc.run(shots=1000)
    post_proc = jaspr.extract_post_processing()
    
    match, msg = compare_post_processed_distributions(qc_results, post_proc, terminal_results)
    assert match, f"CNOT test - {msg}"
    
    # Test 4: GHZ State
    def ghz_algorithm():
        q1 = QuantumBool()
        q2 = QuantumBool()
        q3 = QuantumBool()
        h(q1)
        cx(q1, q2)
        cx(q1, q3)
        return q1, q2, q3
    
    ghz_terminal = terminal_sampling(ghz_algorithm)
    
    @make_jaspr
    def ghz_post_proc():
        q1, q2, q3 = ghz_algorithm()
        m1 = measure(q1)
        m2 = measure(q2)
        m3 = measure(q3)
        return m1, m2, m3
    
    terminal_results = ghz_terminal()
    jaspr = ghz_post_proc()
    result = jaspr.to_qc()
    qc = result[-1]
    qc_results = qc.run(shots=1000)
    post_proc = jaspr.extract_post_processing()
    
    match, msg = compare_post_processed_distributions(qc_results, post_proc, terminal_results)
    assert match, f"GHZ state test - {msg}"
    
    # Verify GHZ state property: all qubits should be same
    for bitstring in list(qc_results.keys())[:2]:
        bits = bitstring_to_array(bitstring)
        result = post_proc(bits)
        assert result[0] == result[1] == result[2], f"GHZ state violation: {result}"
    
    # Test 5: Mixed Size QuantumFloats
    def mixed_size_algorithm():
        qf1 = QuantumFloat(2, signed=False)
        qf2 = QuantumFloat(3, signed=False)
        qf1[:] = 2
        qf2[:] = 5
        return qf1, qf2
    
    mixed_size_terminal = terminal_sampling(mixed_size_algorithm)
    
    @make_jaspr
    def mixed_size_post_proc():
        qf1, qf2 = mixed_size_algorithm()
        m1 = measure(qf1)
        m2 = measure(qf2)
        return m1, m2
    
    terminal_results = mixed_size_terminal()
    terminal_key = list(terminal_results.keys())[0]
    jaspr = mixed_size_post_proc()
    result = jaspr.to_qc()
    qc = result[-1]
    qc_results = qc.run(shots=1000)
    post_proc = jaspr.extract_post_processing()
    
    match, msg = compare_post_processed_distributions(qc_results, post_proc, terminal_results)
    assert match, f"Mixed size QuantumFloats test - {msg}"
    
    # Verify deterministic outcome
    assert len(qc_results) == 1, "Expected deterministic outcome"
    bitstring = list(qc_results.keys())[0]
    bits = bitstring_to_array(bitstring)
    result = post_proc(bits)
    assert result[0] == terminal_key[0], f"qf1 mismatch: expected {terminal_key[0]}, got {result[0]}"
    assert result[1] == terminal_key[1], f"qf2 mismatch: expected {terminal_key[1]}, got {result[1]}"
    
    # Test 6: Same-sized QuantumFloats
    def arithmetic_algorithm():
        qf1 = QuantumFloat(3, signed=False)
        qf2 = QuantumFloat(3, signed=False)
        qf1[:] = 3
        qf2[:] = 5
        return qf1, qf2
    
    arithmetic_terminal = terminal_sampling(arithmetic_algorithm)
    
    @make_jaspr
    def arithmetic_post_proc():
        qf1, qf2 = arithmetic_algorithm()
        m1 = measure(qf1)
        m2 = measure(qf2)
        return m1, m2
    
    terminal_results = arithmetic_terminal()
    terminal_key = list(terminal_results.keys())[0]
    jaspr = arithmetic_post_proc()
    result = jaspr.to_qc()
    qc = result[-1]
    qc_results = qc.run(shots=1000)
    post_proc = jaspr.extract_post_processing()
    
    match, msg = compare_post_processed_distributions(qc_results, post_proc, terminal_results)
    assert match, f"Same-sized QuantumFloats test - {msg}"
    
    # Verify deterministic outcome
    assert len(qc_results) == 1, "Expected deterministic outcome"
    bitstring = list(qc_results.keys())[0]
    bits = bitstring_to_array(bitstring)
    result = post_proc(bits)
    assert result[0] == terminal_key[0], f"qf1 mismatch: expected {terminal_key[0]}, got {result[0]}"
    assert result[1] == terminal_key[1], f"qf2 mismatch: expected {terminal_key[1]}, got {result[1]}"

def test_parity_post_processing():
    """Test parity primitive in post-processing extraction."""
    from qrisp.jasp import parity
    
    # Test basic parity computation
    @make_jaspr
    def parity_test():
        qv = QuantumVariable(3)
        x(qv[0])
        x(qv[2])
        
        m1 = measure(qv[0])
        m2 = measure(qv[1])
        m3 = measure(qv[2])
        
        # Parity of (True, False, True) = False (even number of Trues)
        result = parity(m1, m2, m3)
        return result
    
    jaspr = parity_test()
    post_proc = jaspr.extract_post_processing()
    
    # Test with bitstring
    result = post_proc("101")  # m1=True, m2=False, m3=True -> parity=False
    assert result == 0, f"Expected 0 (False), got {result}"
    
    # Test with array
    import jax.numpy as jnp
    result = post_proc(jnp.array([True, False, True], dtype=bool))
    assert result == 0, f"Expected 0 (False), got {result}"
    
    # Test with expectation parameter
    @make_jaspr
    def parity_with_expectation():
        qv = QuantumVariable(2)
        x(qv[0])
        
        m1 = measure(qv[0])
        m2 = measure(qv[1])
        
        # Parity is True, expectation is False
        # Result should be 1 (indicating mismatch)
        result = parity(m1, m2, expectation=False)
        return result
    
    jaspr = parity_with_expectation()
    post_proc = jaspr.extract_post_processing()
    
    result = post_proc("10")  # m1=True, m2=False -> parity=True, expectation=False -> result=1
    assert result == 1, f"Expected 1 (mismatch indicator), got {result}"
    
    # Test matching expectation
    result = post_proc("00")  # m1=False, m2=False -> parity=False, expectation=False -> result=0
    assert result == 0, f"Expected 0 (match indicator), got {result}"
    
    # Test parity in control flow
    from jax.lax import cond
    
    @make_jaspr
    def parity_in_control():
        qv = QuantumVariable(2)
        x(qv[0])
        
        m1 = measure(qv[0])
        m2 = measure(qv[1])
        
        par = parity(m1, m2)
        
        return cond(par, lambda: 10, lambda: 20)
    
    jaspr = parity_in_control()
    post_proc = jaspr.extract_post_processing()
    
    result = post_proc("10")  # parity=True -> return 10
    assert result == 10, f"Expected 10, got {result}"
    
    result = post_proc("00")  # parity=False -> return 20
    assert result == 20, f"Expected 20, got {result}"    
    # Test that post-processing function can be jitted
    from jax import jit
    jitted_post_proc = jit(post_proc)
    
    result = jitted_post_proc(jnp.array([True, False], dtype=bool))
    assert result == 10, f"Expected 10 from jitted function, got {result}"
    
    result = jitted_post_proc(jnp.array([False, False], dtype=bool))
    assert result == 20, f"Expected 20 from jitted function, got {result}"