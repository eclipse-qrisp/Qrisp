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
    result = post_proc("00")
    assert result[0] == 5 and result[1] == False
    
    result = post_proc("01")
    assert result[0] == 5 and result[1] == True
    
    result = post_proc("10")
    assert result[0] == 6 and result[1] == False
    
    result = post_proc("11")
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
    
    and_result, or_result, not_result = post_proc("01")
    assert and_result == False and or_result == True and not_result == True
    
    and_result, or_result, not_result = post_proc("10")
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
    assert post_proc("01") == (0 + 2) * (1 + 3)   # 2 * 4 = 8
    assert post_proc("10") == (1 + 2) * (0 + 3)   # 3 * 3 = 9
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
    
    result, size = post_proc("10")
    assert result == 7  # (1 + 0) * 7
    assert size == 7
    
    result, size = post_proc("01")
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
    
    # Test with bitstring "01"
    result = post_proc("01")
    assert result == (1, True)  # meas[0]=False -> 0+1=1, meas[1]=True
    
    # Test with bitstring "10"
    result = post_proc("10")
    assert result == (2, False)  # meas[0]=True -> 1+1=2, meas[1]=False
    
    # Test with bitstring "11"
    result = post_proc("11")
    assert result == (2, True)  # meas[0]=True -> 1+1=2, meas[1]=True
    
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
    post_proc_bitstring = jaspr.extract_post_processing()  # Default is bitstring
    post_proc_array = jaspr.extract_post_processing(array_input=True)
    
    # Test all combinations
    for bits in ["000", "001", "010", "011", "100", "101", "110", "111"]:
        array_input = jnp.array([c == '1' for c in bits], dtype=bool)
        result_bitstring = post_proc_bitstring(bits)
        result_array = post_proc_array(array_input)
        assert result_array == result_bitstring, f"Mismatch for {bits}: {result_array} != {result_bitstring}"


def test_array_input_jittable():
    """
    Test that array_input=True produces JAX-jittable functions.
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
    post_proc = jaspr.extract_post_processing(array_input=True)
    
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
    
    # Extract post-processing with array input (required for jitting)
    post_proc = jaspr.extract_post_processing(array_input=True)
    
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
    test_input = jnp.array([True, False, False])
    result = jitted_post_proc(test_input)
    expected = 2.0
    assert abs(float(result) - expected) < 0.01, f"Expected {expected}, got {result}"
    
    # "010" -> 2 * 0.5 = 1.0
    test_input = jnp.array([False, True, False])
    result = jitted_post_proc(test_input)
    expected = 1.0
    assert abs(float(result) - expected) < 0.01, f"Expected {expected}, got {result}"
    
    # "001" -> 1 * 0.5 = 0.5
    test_input = jnp.array([False, False, True])
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
    
    # Extract post-processing with array input (required for jitting)
    post_proc = jaspr.extract_post_processing(array_input=True)
    
    # Jit the post-processing function
    jitted_post_proc = jax.jit(post_proc)
    
    # Test various combinations
    # Both measurements False -> 0 + 2*0 = 0
    test_input = jnp.array([False, False])
    result = jitted_post_proc(test_input)
    assert result == 0, f"Expected 0, got {result}"
    
    # First True, second False -> 1 + 2*0 = 1
    test_input = jnp.array([True, False])
    result = jitted_post_proc(test_input)
    assert result == 1, f"Expected 1, got {result}"
    
    # First False, second True -> 0 + 2*1 = 2
    test_input = jnp.array([False, True])
    result = jitted_post_proc(test_input)
    assert result == 2, f"Expected 2, got {result}"
    
    # Both True -> 1 + 2*1 = 3
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
    
    # Extract post-processing with array input
    post_proc = jaspr.extract_post_processing(array_input=True)
    
    # Jit the post-processing function
    jitted_post_proc = jax.jit(post_proc)
    
    # Test various combinations
    # Note: For QuantumFloat(2, exponent=-1), bit order is MSB, LSB
    # qf_value = (2*bit[qf_msb] + bit[qf_lsb]) * 0.5
    
    # bit=False (0), qf="00" (0.0) -> 0 + 0.0 = 0.0
    test_input = jnp.array([False, False, False])
    result = jitted_post_proc(test_input)
    expected = 0.0
    assert abs(float(result) - expected) < 0.01, f"Expected {expected}, got {result}"
    
    # bit=True (1), qf="00" (0.0) -> 1 + 0.0 = 1.0
    test_input = jnp.array([True, False, False])
    result = jitted_post_proc(test_input)
    expected = 1.0
    assert abs(float(result) - expected) < 0.01, f"Expected {expected}, got {result}"
    
    # bit=False (0), qf="10" (2*0.5=1.0) -> 0 + 1.0 = 1.0
    test_input = jnp.array([False, True, False])
    result = jitted_post_proc(test_input)
    expected = 1.0
    assert abs(float(result) - expected) < 0.01, f"Expected {expected}, got {result}"
    
    # bit=False (0), qf="01" (1*0.5=0.5) -> 0 + 0.5 = 0.5
    test_input = jnp.array([False, False, True])
    result = jitted_post_proc(test_input)
    expected = 0.5
    assert abs(float(result) - expected) < 0.01, f"Expected {expected}, got {result}"
    
    # bit=True (1), qf="11" ((2+1)*0.5=1.5) -> 1 + 1.5 = 2.5
    test_input = jnp.array([True, True, True])
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
    post_proc = jaspr.extract_post_processing(array_input=True)
    
    # Test with meas=False -> false_branch(5) = 25
    result = post_proc(jnp.array([False, False]))
    assert result == 25, f"Expected 25, got {result}"
    
    # Test with meas=True -> true_branch(5) = 15
    result = post_proc(jnp.array([True, False]))
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
    post_proc = jaspr.extract_post_processing(array_input=True)
    
    # Test with both False -> 0 + 0 = 0, loop runs 0 times, counter=0
    result = post_proc(jnp.array([False, False, False]))
    assert result == 0, f"Expected 0, got {result}"
    
    # Test with one True -> 1 + 0 = 1, loop runs 1 time, counter=1
    result = post_proc(jnp.array([True, False, False]))
    assert result == 1, f"Expected 1, got {result}"
    
    # Test with other True -> 0 + 1 = 1, loop runs 1 time, counter=1
    result = post_proc(jnp.array([False, True, False]))
    assert result == 1, f"Expected 1, got {result}"
    
    # Test with both True -> 1 + 1 = 2, loop runs 2 times, counter=2
    result = post_proc(jnp.array([True, True, False]))
    assert result == 2, f"Expected 2, got {result}"
    
    # Test that it can be jitted
    jitted_post_proc = jax.jit(post_proc)
    result = jitted_post_proc(jnp.array([True, True, False]))
    assert result == 2, f"Jitted: Expected 2, got {result}"
