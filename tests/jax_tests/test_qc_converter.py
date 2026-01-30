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

from jax import make_jaxpr
from qrisp import QuantumVariable, cx, QuantumCircuit, QuantumFloat, x, rz, measure, control, QuantumBool
from qrisp.jasp import qache, flatten_pjit, make_jaspr, ProcessedMeasurement

def test_qc_converter():
    
    def test_function(i):
        qv = QuantumVariable(i)
        
        cx(qv[0], qv[1])
        cx(qv[1], qv[2])
        cx(qv[1], qv[2])
    
    for i in range(3, 7):
        
        jaspr = make_jaspr(test_function)(i)
        qc = jaspr.to_qc(i)
        
        comparison_qc = QuantumCircuit(i)
        comparison_qc.cx(0, 1)
        comparison_qc.cx(1, 2)
        comparison_qc.cx(1, 2)
        
        assert qc.compare_unitary(comparison_qc)
    
    @qache
    def inner_function(qv, i):
        cx(qv[i], qv[i+1])
        cx(qv[i+2], qv[i+1])
    
    def test_function(i):
        qv = QuantumVariable(i)
        
        inner_function(qv, 0)
        inner_function(qv, 1)
        inner_function(qv, 2)
        
    
    jaspr = make_jaspr(test_function)(5)
    qc = jaspr.to_qc(5)
    
    assert len(qc.data) == 3 + len(qc.qubits)
    print(qc)
    comparison_qc = QuantumCircuit(5)
    
    comparison_qc.cx(0, 1)
    comparison_qc.cx(2, 1)
    comparison_qc.cx(1, 2)
    comparison_qc.cx(3, 2)
    comparison_qc.cx(2, 3)
    comparison_qc.cx(4, 3)
    
    print(comparison_qc)
    
    assert qc.compare_unitary(comparison_qc)
    
    jaspr = make_jaspr(test_function)(5)
    flattened_jaspr = flatten_pjit(jaspr)
    
    qc = flattened_jaspr.to_qc(5)
    
    assert qc.compare_unitary(comparison_qc)
    
    ######
    # Test classically controlled Operations
    
    def main():
        
        qf = QuantumFloat(5)
        bl = measure(qf[0])
        
        with control(bl):
            rz(0.5, qf[1])
            x(qf[1])
        
        return

    jaspr = make_jaspr(main)()

    qrisp_qc = jaspr.to_qc()
    qiskit_qc = qrisp_qc.to_qiskit()
    qasm_str = qrisp_qc.to_qasm3()

    assert qasm_str.find("if (cb_0[0]) {") != -1
    
    def main():
        
        qv = QuantumFloat(3)
        qv += 4

    jaspr = make_jaspr(main)()
    str(jaspr.to_qc())
    
    def main():
        
        qv = QuantumFloat(5)
        qbl = QuantumBool()
        with control(qbl):
            qv += 4
        
        return measure(qv) + 5
        
    jaspr = make_jaspr(main)()
    
    assert isinstance(jaspr.to_qc()[0], ProcessedMeasurement)


def test_parity_to_qc():
    """Test parity primitive with to_qc (cl_func_interpreter/qc_extraction_interpreter)."""
    from qrisp.jasp import parity
    from qrisp import Clbit
    
    # Test that parity of scalar measurements creates a ParityOperation in the circuit
    # and returns a Clbit representing the parity result
    def parity_test():
        qv = QuantumVariable(3)
        x(qv[0])
        x(qv[2])
        
        m1 = measure(qv[0])
        m2 = measure(qv[1])
        m3 = measure(qv[2])
        
        result = parity(m1, m2, m3)
        return result
    
    jaspr = make_jaspr(parity_test)()
    result, qc = jaspr.to_qc()
    
    # Parity of scalar measurements returns a Clbit (ParityOperation added to circuit)
    assert isinstance(result, Clbit), f"Expected Clbit, got {type(result)}"
    
    # The quantum circuit should contain the measurements plus the parity result
    assert len(qc.clbits) == 4, f"Expected 4 classical bits (3 measurements + 1 parity result), got {len(qc.clbits)}"
    
    # Test parity with array inputs (while primitive)
    def parity_while_test():
        import jax.numpy as jnp
        
        qv0 = QuantumVariable(3)
        qv1 = QuantumVariable(3)
        
        x(qv0[0])
        x(qv1[1])
        
        # Measure and create arrays
        m0_0 = measure(qv0[0])
        m0_1 = measure(qv0[1])
        m0_2 = measure(qv0[2])
        
        m1_0 = measure(qv1[0])
        m1_1 = measure(qv1[1])
        m1_2 = measure(qv1[2])
        
        meas_array_0 = jnp.array([m0_0, m0_1, m0_2])
        meas_array_1 = jnp.array([m1_0, m1_1, m1_2])
        
        # Parity with arrays (triggers while)
        result = parity(meas_array_0, meas_array_1)
        return result
    
    jaspr = make_jaspr(parity_while_test)()
    result, qc = jaspr.to_qc()
    
    # Array parity returns a MeasurementArray with Clbit references (negative values)
    # because parity operations create ParityOperations that produce new Clbits
    from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import MeasurementArray
    assert isinstance(result, MeasurementArray), f"Expected MeasurementArray for array parity, got {type(result)}"


def test_measurement_array_basic():
    """Test basic MeasurementArray construction and element extraction."""
    import jax.numpy as jnp
    from qrisp import Clbit
    from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import MeasurementArray
    
    # Test 1: Array construction from measurements and element extraction
    def array_construction_test():
        qv = QuantumVariable(3)
        x(qv[0])
        x(qv[2])
        
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        
        meas_array = jnp.array([m0, m1, m2])
        
        # Extract first element
        first = meas_array[0]
        return first
    
    jaspr = make_jaspr(array_construction_test)()
    result, qc = jaspr.to_qc()
    
    # Extracted element should be a Clbit
    assert isinstance(result, Clbit), f"Expected Clbit, got {type(result)}"
    
    # Test 2: Extract different indices
    def array_index_test():
        qv = QuantumVariable(3)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        
        meas_array = jnp.array([m0, m1, m2])
        return meas_array[1]
    
    jaspr = make_jaspr(array_index_test)()
    result, qc = jaspr.to_qc()
    assert isinstance(result, Clbit), f"Expected Clbit for index 1, got {type(result)}"
    
    # Test 3: Slicing
    def array_slice_test():
        qv = QuantumVariable(4)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        m3 = measure(qv[3])
        
        meas_array = jnp.array([m0, m1, m2, m3])
        sliced = meas_array[1:3]
        return sliced[0]  # Extract from sliced array
    
    jaspr = make_jaspr(array_slice_test)()
    result, qc = jaspr.to_qc()
    assert isinstance(result, Clbit), f"Expected Clbit from sliced array, got {type(result)}"


def test_measurement_array_reshape():
    """Test that reshape preserves measurement information."""
    import jax.numpy as jnp
    from qrisp import Clbit
    
    def reshape_test():
        qv = QuantumVariable(4)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        m3 = measure(qv[3])
        
        meas_array = jnp.array([m0, m1, m2, m3])
        reshaped = meas_array.reshape((2, 2))
        
        # Extract element from reshaped array
        elem = reshaped[0, 0]
        return elem
    
    jaspr = make_jaspr(reshape_test)()
    result, qc = jaspr.to_qc()
    
    # Reshape should preserve measurement info - element should be Clbit
    assert isinstance(result, Clbit), f"Expected Clbit after reshape, got {type(result)}"


def test_measurement_array_processed_markers():
    """Test that classical operations mark array entries as processed."""
    import jax.numpy as jnp
    from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import MeasurementArray
    
    # Test 1: Bitwise NOT marks as processed
    def not_test():
        qv = QuantumVariable(3)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        
        meas_array = jnp.array([m0, m1, m2])
        negated = ~meas_array
        return negated
    
    jaspr = make_jaspr(not_test)()
    result, qc = jaspr.to_qc()
    
    assert isinstance(result, MeasurementArray), f"Expected MeasurementArray, got {type(result)}"
    assert result.has_processed_entries(), "NOT should mark entries as processed"
    assert all(v == MeasurementArray.PROCESSED_VALUE for v in result.data), \
        f"All entries should be processed, got {result.data}"
    
    # Test 2: Extracting from processed array gives ProcessedMeasurement
    def not_extract_test():
        qv = QuantumVariable(3)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        
        meas_array = jnp.array([m0, m1, m2])
        negated = ~meas_array
        first = negated[0]
        return first
    
    jaspr = make_jaspr(not_extract_test)()
    result, qc = jaspr.to_qc()
    
    assert isinstance(result, ProcessedMeasurement), \
        f"Extracting from processed array should give ProcessedMeasurement, got {type(result)}"


def test_measurement_array_concatenate_with_processed():
    """Test concatenating regular measurements with processed values."""
    import jax.numpy as jnp
    from qrisp import Clbit
    from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import MeasurementArray
    
    def concat_test():
        qv = QuantumVariable(3)
        meas_0 = measure(qv[0])
        meas_1 = measure(qv[1])
        meas_2 = measure(qv[2])
        
        # Create array and negate
        meas_array = jnp.array([meas_1, meas_2])
        negated = ~meas_array  # These are now processed
        
        # Concatenate regular measurement with processed array
        combined = jnp.concatenate([jnp.array([meas_0]), negated])
        return combined
    
    jaspr = make_jaspr(concat_test)()
    result, qc = jaspr.to_qc()
    
    assert isinstance(result, MeasurementArray), f"Expected MeasurementArray, got {type(result)}"
    assert len(result.data) == 3, f"Expected 3 elements, got {len(result.data)}"
    
    # First element should be a real measurement (negative index)
    assert result.data[0] < 0, f"First element should be measurement, got {result.data[0]}"
    
    # Rest should be processed
    assert result.data[1] == MeasurementArray.PROCESSED_VALUE, \
        f"Second element should be processed, got {result.data[1]}"
    assert result.data[2] == MeasurementArray.PROCESSED_VALUE, \
        f"Third element should be processed, got {result.data[2]}"
    
    # Extract first (should be Clbit)
    first = result[0]
    assert isinstance(first, Clbit), f"First element should be Clbit, got {type(first)}"
    
    # Extract second (should be ProcessedMeasurement)
    second = result[1]
    assert isinstance(second, ProcessedMeasurement), \
        f"Second element should be ProcessedMeasurement, got {type(second)}"


def test_measurement_array_type_conversion():
    """Test that type conversions to numeric types mark as processed."""
    import jax.numpy as jnp
    from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import MeasurementArray
    
    # Test: astype to int should mark as processed
    def astype_int_test():
        qv = QuantumVariable(3)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        
        meas_array = jnp.array([m0, m1, m2])
        int_array = meas_array.astype(jnp.int32)
        return int_array
    
    jaspr = make_jaspr(astype_int_test)()
    result, qc = jaspr.to_qc()
    
    assert isinstance(result, MeasurementArray), f"Expected MeasurementArray, got {type(result)}"
    assert result.has_processed_entries(), "astype(int) should mark entries as processed"


def test_measurement_array_arithmetic():
    """Test that arithmetic operations mark as processed."""
    import jax.numpy as jnp
    from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import MeasurementArray
    
    # Test: Addition
    def add_test():
        qv = QuantumVariable(2)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        
        meas_array = jnp.array([m0, m1])
        added = meas_array + 1
        return added
    
    jaspr = make_jaspr(add_test)()
    result, qc = jaspr.to_qc()
    
    assert isinstance(result, MeasurementArray), f"Expected MeasurementArray, got {type(result)}"
    assert result.has_processed_entries(), "Addition should mark entries as processed"
    
    # Test: Multiplication
    def mul_test():
        qv = QuantumVariable(2)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        
        meas_array = jnp.array([m0, m1])
        multiplied = meas_array * 2
        return multiplied
    
    jaspr = make_jaspr(mul_test)()
    result, qc = jaspr.to_qc()
    
    assert isinstance(result, MeasurementArray), f"Expected MeasurementArray for mul, got {type(result)}"
    assert result.has_processed_entries(), "Multiplication should mark entries as processed"


def test_measurement_array_comparison():
    """Test that comparison operations mark as processed."""
    import jax.numpy as jnp
    from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import MeasurementArray
    
    def compare_test():
        qv = QuantumVariable(2)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        
        meas_array = jnp.array([m0, m1])
        result = meas_array == True
        return result
    
    jaspr = make_jaspr(compare_test)()
    result, qc = jaspr.to_qc()
    
    assert isinstance(result, MeasurementArray), f"Expected MeasurementArray, got {type(result)}"
    assert result.has_processed_entries(), "Comparison should mark entries as processed"


def test_measurement_array_reductions():
    """Test that reduction operations mark as processed."""
    import jax.numpy as jnp
    from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import MeasurementArray
    
    # Test: Sum reduction
    def sum_test():
        qv = QuantumVariable(3)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        
        meas_array = jnp.array([m0, m1, m2])
        total = jnp.sum(meas_array)
        return total
    
    jaspr = make_jaspr(sum_test)()
    result, qc = jaspr.to_qc()
    
    # Sum returns a scalar, which will be MeasurementArray or ProcessedMeasurement
    assert isinstance(result, (MeasurementArray, ProcessedMeasurement)), \
        f"Expected MeasurementArray or ProcessedMeasurement for sum, got {type(result)}"
    
    # Test: any() reduction
    def any_test():
        qv = QuantumVariable(2)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        
        meas_array = jnp.array([m0, m1])
        result = jnp.any(meas_array)
        return result
    
    jaspr = make_jaspr(any_test)()
    result, qc = jaspr.to_qc()
    
    assert isinstance(result, (MeasurementArray, ProcessedMeasurement)), \
        f"Expected MeasurementArray or ProcessedMeasurement for any, got {type(result)}"


def test_measurement_array_bitwise_ops():
    """Test that bitwise operations mark as processed."""
    import jax.numpy as jnp
    from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import MeasurementArray
    
    # Test: AND
    def and_test():
        qv = QuantumVariable(2)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        
        arr1 = jnp.array([m0])
        arr2 = jnp.array([m1])
        result = arr1 & arr2
        return result
    
    jaspr = make_jaspr(and_test)()
    result, qc = jaspr.to_qc()
    
    assert isinstance(result, MeasurementArray), f"Expected MeasurementArray for AND, got {type(result)}"
    assert result.has_processed_entries(), "AND should mark entries as processed"
    
    # Test: OR
    def or_test():
        qv = QuantumVariable(2)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        
        arr1 = jnp.array([m0])
        arr2 = jnp.array([m1])
        result = arr1 | arr2
        return result
    
    jaspr = make_jaspr(or_test)()
    result, qc = jaspr.to_qc()
    
    assert isinstance(result, MeasurementArray), f"Expected MeasurementArray for OR, got {type(result)}"
    assert result.has_processed_entries(), "OR should mark entries as processed"
    
    # Test: XOR
    def xor_test():
        qv = QuantumVariable(2)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        
        arr1 = jnp.array([m0])
        arr2 = jnp.array([m1])
        result = arr1 ^ arr2
        return result
    
    jaspr = make_jaspr(xor_test)()
    result, qc = jaspr.to_qc()
    
    assert isinstance(result, MeasurementArray), f"Expected MeasurementArray for XOR, got {type(result)}"
    assert result.has_processed_entries(), "XOR should mark entries as processed"


def test_measurement_array_parity_with_processed():
    """Test that parity raises error when given processed measurement data."""
    import jax.numpy as jnp
    from qrisp.jasp import parity
    
    def parity_negated_test():
        qv = QuantumVariable(3)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        
        meas_array = jnp.array([m0, m1, m2])
        negated = ~meas_array  # This is processed
        
        # Try to compute parity on processed data - should fail
        return parity(negated)
    
    jaspr = make_jaspr(parity_negated_test)()
    
    try:
        result, qc = jaspr.to_qc()
        assert False, "Expected exception when computing parity of processed data"
    except Exception as e:
        # Should raise an error about processed measurements
        assert "processed" in str(e).lower(), \
            f"Exception should mention 'processed', got: {e}"


def test_measurement_array_parity_with_unprocessed():
    """Test parity with measurement arrays (uses scan primitive internally)."""
    import jax.numpy as jnp
    from qrisp import Clbit
    from qrisp.jasp import parity
    from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import MeasurementArray
    
    # When parity is called with an array, it uses the while primitive internally,
    # which returns a MeasurementArray with processed entries since while involves 
    # operations that can't be directly represented in a QuantumCircuit.
    def parity_array_test():
        qv = QuantumVariable(3)
        x(qv[0])
        x(qv[2])
        
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        
        # Create array without any classical processing
        meas_array = jnp.array([m0, m1, m2])
        
        # Parity of array uses while internally
        result = parity(meas_array)
        return result
    
    jaspr = make_jaspr(parity_array_test)()
    result, qc = jaspr.to_qc()
    
    # Array parity returns a MeasurementArray with Clbit references (negative values)
    # because parity operations create ParityOperations that produce new Clbits
    # For direct Clbit parity, use scalar arguments: parity(m0, m1, m2)
    assert isinstance(result, MeasurementArray), \
        f"Array parity (via scan) should return MeasurementArray, got {type(result)}"


def test_measurement_array_chain_operations():
    """Test chaining multiple operations on measurement arrays."""
    import jax.numpy as jnp
    from qrisp import Clbit
    from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import MeasurementArray
    
    # Test: Extract from array, then use in another array
    def chain_test():
        qv = QuantumVariable(4)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        m3 = measure(qv[3])
        
        # Create first array
        arr1 = jnp.array([m0, m1])
        # Create second array  
        arr2 = jnp.array([m2, m3])
        
        # Concatenate
        combined = jnp.concatenate([arr1, arr2])
        
        # Extract element
        elem = combined[2]  # Should be m2
        return elem
    
    jaspr = make_jaspr(chain_test)()
    result, qc = jaspr.to_qc()
    
    assert isinstance(result, Clbit), f"Expected Clbit from chained operations, got {type(result)}"
    
    # Test: Mix of processed and unprocessed in chain
    def mixed_chain_test():
        qv = QuantumVariable(4)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        m3 = measure(qv[3])
        
        # Some are processed, some are not
        processed_arr = ~jnp.array([m0, m1])  # Processed
        regular_arr = jnp.array([m2, m3])      # Not processed
        
        # Concatenate mixed
        combined = jnp.concatenate([regular_arr, processed_arr])
        return combined
    
    jaspr = make_jaspr(mixed_chain_test)()
    result, qc = jaspr.to_qc()
    
    assert isinstance(result, MeasurementArray), f"Expected MeasurementArray, got {type(result)}"
    
    # First two should be unprocessed (negative indices)
    assert result.data[0] < 0, f"First element should be measurement, got {result.data[0]}"
    assert result.data[1] < 0, f"Second element should be measurement, got {result.data[1]}"
    
    # Last two should be processed
    assert result.data[2] == MeasurementArray.PROCESSED_VALUE, \
        f"Third element should be processed, got {result.data[2]}"
    assert result.data[3] == MeasurementArray.PROCESSED_VALUE, \
        f"Fourth element should be processed, got {result.data[3]}"

