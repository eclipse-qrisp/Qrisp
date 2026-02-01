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
from qrisp import QuantumVariable, cx, QuantumCircuit, QuantumFloat, x, rz, measure, control, QuantumBool, QuantumArray
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
    
    # 6 cx operations (2 per inner_function call × 3 calls) + 5 qubit allocations
    assert len(qc.data) == 6 + len(qc.qubits)
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
    import numpy as np
    from qrisp.jasp import parity
    from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import ParityHandle
    from qrisp.jasp.primitives.parity_primitive import ParityOperation
    
    # Test that parity of scalar measurements creates a ParityOperation in the circuit
    # and returns a ParityHandle containing the parity result clbits
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
    
    # Parity of scalar measurements returns a ParityHandle
    assert isinstance(result, ParityHandle), f"Expected ParityHandle, got {type(result)}"
    
    # The quantum circuit should contain ONLY the measurements (parity tracked separately)
    assert len(qc.clbits) == 3, f"Expected 3 classical bits (3 measurements), got {len(qc.clbits)}"
    
    # Check the ParityHandle has 3 input clbits (stored directly)
    assert len(result.clbits) == 3, f"Expected 3 input clbits in parity entry, got {len(result.clbits)}"
    
    # Verify ParityOperation was added to circuit data
    parity_ops = [op for op in qc.data if isinstance(op.op, ParityOperation)]
    assert len(parity_ops) == 1, f"Expected 1 ParityOperation in circuit, got {len(parity_ops)}"
    
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
    
    # Array parity now returns a numpy array with dtype=object containing ParityHandles
    assert isinstance(result, np.ndarray), f"Expected numpy array for array parity, got {type(result)}"
    assert result.dtype == object, f"Expected dtype=object, got {result.dtype}"
    assert all(isinstance(item, ParityHandle) for item in result.flat), \
        f"All elements should be ParityHandle objects"
    
    # Should have 3 ParityOperations in the circuit (one for each element-wise parity)
    parity_ops = [op for op in qc.data if isinstance(op.op, ParityOperation)]
    assert len(parity_ops) == 3, f"Expected 3 ParityOperations, got {len(parity_ops)}"


def test_measurement_array_basic():
    """Test basic measurement array construction and element extraction."""
    import jax.numpy as jnp
    from qrisp import Clbit
    
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
    import numpy as np
    import jax.numpy as jnp
    
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
    
    # Result is now a numpy array with dtype=object containing ProcessedMeasurement objects
    assert isinstance(result, np.ndarray), f"Expected numpy array, got {type(result)}"
    assert result.dtype == object, f"Expected dtype=object, got {result.dtype}"
    assert all(isinstance(item, ProcessedMeasurement) for item in result.flat), \
        f"NOT should mark all entries as ProcessedMeasurement"
    
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
    import numpy as np
    import jax.numpy as jnp
    from qrisp import Clbit
    
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
    
    # Result is now a numpy array with dtype=object
    assert isinstance(result, np.ndarray), f"Expected numpy array, got {type(result)}"
    assert result.dtype == object, f"Expected dtype=object, got {result.dtype}"
    assert len(result) == 3, f"Expected 3 elements, got {len(result)}"
    
    # First element should be a Clbit (real measurement)
    assert isinstance(result[0], Clbit), f"First element should be Clbit, got {type(result[0])}"
    
    # Rest should be ProcessedMeasurement
    assert isinstance(result[1], ProcessedMeasurement), \
        f"Second element should be ProcessedMeasurement, got {type(result[1])}"
    assert isinstance(result[2], ProcessedMeasurement), \
        f"Third element should be ProcessedMeasurement, got {type(result[2])}"


def test_measurement_array_type_conversion():
    """Test that type conversions to float types mark as processed, but int conversions pass through."""
    import numpy as np
    import jax.numpy as jnp
    from qrisp import Clbit
    
    # Test: astype to int should pass through (needed for cond primitive)
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
    
    # Result is now a numpy array with dtype=object
    assert isinstance(result, np.ndarray), f"Expected numpy array, got {type(result)}"
    # bool->int conversions pass through - all elements should be Clbit (not ProcessedMeasurement)
    assert all(isinstance(item, Clbit) for item in result.flat), \
        "astype(int) should NOT mark entries as processed (needed for cond)"


def test_measurement_array_arithmetic():
    """Test that arithmetic operations mark as processed."""
    import numpy as np
    import jax.numpy as jnp
    
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
    
    # Result is now a numpy array with dtype=object containing ProcessedMeasurement
    assert isinstance(result, np.ndarray), f"Expected numpy array, got {type(result)}"
    assert all(isinstance(item, ProcessedMeasurement) for item in result.flat), \
        "Addition should mark entries as ProcessedMeasurement"
    
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
    
    assert isinstance(result, np.ndarray), f"Expected numpy array for mul, got {type(result)}"
    assert all(isinstance(item, ProcessedMeasurement) for item in result.flat), \
        "Multiplication should mark entries as ProcessedMeasurement"


def test_measurement_array_comparison():
    """Test that comparison operations mark as processed."""
    import numpy as np
    import jax.numpy as jnp
    
    def compare_test():
        qv = QuantumVariable(2)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        
        meas_array = jnp.array([m0, m1])
        result = meas_array == True
        return result
    
    jaspr = make_jaspr(compare_test)()
    result, qc = jaspr.to_qc()
    
    assert isinstance(result, np.ndarray), f"Expected numpy array, got {type(result)}"
    assert all(isinstance(item, ProcessedMeasurement) for item in result.flat), \
        "Comparison should mark entries as ProcessedMeasurement"


def test_measurement_array_reductions():
    """Test that reduction operations mark as processed."""
    import numpy as np
    import jax.numpy as jnp
    
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
    
    # Sum returns either a scalar ProcessedMeasurement or numpy array with ProcessedMeasurement
    if isinstance(result, np.ndarray):
        assert all(isinstance(item, ProcessedMeasurement) for item in result.flat), \
            f"Expected ProcessedMeasurement entries for sum, got {result}"
    else:
        assert isinstance(result, ProcessedMeasurement), \
            f"Expected ProcessedMeasurement for sum, got {type(result)}"
    
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
    
    # any() returns either a scalar ProcessedMeasurement or numpy array with ProcessedMeasurement
    if isinstance(result, np.ndarray):
        assert all(isinstance(item, ProcessedMeasurement) for item in result.flat), \
            f"Expected ProcessedMeasurement entries for any, got {result}"
    else:
        assert isinstance(result, ProcessedMeasurement), \
            f"Expected ProcessedMeasurement for any, got {type(result)}"


def test_measurement_array_bitwise_ops():
    """Test that bitwise operations mark as processed."""
    import numpy as np
    import jax.numpy as jnp
    
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
    
    assert isinstance(result, np.ndarray), f"Expected numpy array for AND, got {type(result)}"
    assert all(isinstance(item, ProcessedMeasurement) for item in result.flat), \
        "AND should mark entries as ProcessedMeasurement"
    
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
    
    assert isinstance(result, np.ndarray), f"Expected numpy array for OR, got {type(result)}"
    assert all(isinstance(item, ProcessedMeasurement) for item in result.flat), \
        "OR should mark entries as ProcessedMeasurement"
    
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
    
    assert isinstance(result, np.ndarray), f"Expected numpy array for XOR, got {type(result)}"
    assert all(isinstance(item, ProcessedMeasurement) for item in result.flat), \
        "XOR should mark entries as ProcessedMeasurement"


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
    import numpy as np
    import jax.numpy as jnp
    from qrisp import Clbit
    from qrisp.jasp import parity
    from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import ParityHandle
    
    # When parity is called with a single array, it creates a parity for each element
    # (each element is XORed with identity, essentially creating ParityHandles wrapping each clbit)
    def parity_array_test():
        qv = QuantumVariable(3)
        x(qv[0])
        x(qv[2])
        
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        
        # Create array without any classical processing
        meas_array = jnp.array([m0, m1, m2])
        
        # Parity of single array creates element-wise ParityHandles
        result = parity(meas_array)
        return result
    
    jaspr = make_jaspr(parity_array_test)()
    result, qc = jaspr.to_qc()
    
    # Parity of array returns numpy array of ParityHandles (one per element)
    assert isinstance(result, np.ndarray), \
        f"Array parity should return numpy array, got {type(result)}"
    assert result.dtype == object, f"Expected dtype=object, got {result.dtype}"
    assert all(isinstance(item, ParityHandle) for item in result.flat), \
        f"All elements should be ParityHandle objects"


def test_measurement_array_chain_operations():
    """Test chaining multiple operations on measurement arrays."""
    import numpy as np
    import jax.numpy as jnp
    from qrisp import Clbit
    
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
    
    # Result is now a numpy array with dtype=object
    assert isinstance(result, np.ndarray), f"Expected numpy array, got {type(result)}"
    
    # First two should be Clbit (unprocessed measurements)
    assert isinstance(result[0], Clbit), f"First element should be Clbit, got {type(result[0])}"
    assert isinstance(result[1], Clbit), f"Second element should be Clbit, got {type(result[1])}"
    
    # Last two should be ProcessedMeasurement
    assert isinstance(result[2], ProcessedMeasurement), \
        f"Third element should be ProcessedMeasurement, got {type(result[2])}"
    assert isinstance(result[3], ProcessedMeasurement), \
        f"Fourth element should be ProcessedMeasurement, got {type(result[3])}"


def test_scatter_with_clbit_updates():
    """Test scatter primitive handling with Clbit updates (used by QuantumArray measurement)."""
    import numpy as np
    import jax.numpy as jnp
    from qrisp import Clbit
    
    # Test 1: Simple QuantumArray measurement (uses scatter internally via while loop)
    def simple_quantum_array_measure():
        qa = QuantumArray(qtype=QuantumBool(), shape=(2,))
        meas = measure(qa)
        return meas
    
    jaspr = make_jaspr(simple_quantum_array_measure)()
    result, qc = jaspr.to_qc()
    
    # Result is now a numpy array with dtype=object containing Clbit objects
    assert isinstance(result, np.ndarray), \
        f"QuantumArray measurement should return numpy array, got {type(result)}"
    assert result.shape == (2,), f"Expected shape (2,), got {result.shape}"
    # Should contain Clbit objects
    assert all(isinstance(item, Clbit) for item in result.flat), \
        f"All values should be Clbit objects"
    
    # Test 2: QuantumArray with different shape (stored flattened internally)
    def shaped_quantum_array_measure():
        qa = QuantumArray(qtype=QuantumBool(), shape=(2, 3))
        meas = measure(qa)
        return meas
    
    jaspr = make_jaspr(shaped_quantum_array_measure)()
    result, qc = jaspr.to_qc()
    
    assert isinstance(result, np.ndarray), \
        f"QuantumArray measurement should return numpy array, got {type(result)}"
    # Result should be flattened or have proper shape
    assert result.size == 6, f"Expected 6 elements, got {result.size}"
    # Should have 6 unique Clbit references
    unique_refs = set(item for item in result.flat)
    assert len(unique_refs) == 6, f"Expected 6 unique Clbit refs, got {len(unique_refs)}"
    
    # Test 3: Multiple QuantumArray measurements
    def multiple_quantum_array_measure():
        qa_0 = QuantumArray(qtype=QuantumBool(), shape=(2,))
        qa_1 = QuantumArray(qtype=QuantumBool(), shape=(3,))
        meas_0 = measure(qa_0)
        meas_1 = measure(qa_1)
        return meas_0, meas_1
    
    jaspr = make_jaspr(multiple_quantum_array_measure)()
    result_0, result_1, qc = jaspr.to_qc()
    
    assert isinstance(result_0, np.ndarray), f"First result should be numpy array"
    assert isinstance(result_1, np.ndarray), f"Second result should be numpy array"
    assert result_0.shape == (2,), f"Expected shape (2,), got {result_0.shape}"
    assert result_1.shape == (3,), f"Expected shape (3,), got {result_1.shape}"
    
    # All Clbit references should be unique across both arrays
    all_refs = list(result_0.flatten()) + list(result_1.flatten())
    assert len(set(all_refs)) == 5, f"Expected 5 unique Clbit refs total, got {len(set(all_refs))}"


def test_quantum_array_measurement_with_parity():
    """Test the user's example: QuantumArray measurement with parity operation."""
    import numpy as np
    import jax.numpy as jnp
    from qrisp.jasp import parity
    from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import ParityHandle
    from qrisp.jasp.primitives.parity_primitive import ParityOperation
    
    # User's exact example
    @make_jaspr
    def array_test():
        qa_0 = QuantumArray(qtype=QuantumBool(), shape=(2, 1))
        qa_1 = QuantumArray(qtype=QuantumBool(), shape=(2, 1))
        meas_0 = measure(qa_0)
        meas_1 = measure(qa_1)
        return parity(meas_0, meas_1)
    
    jaspr = array_test()
    result, qc = jaspr.to_qc()
    
    # Result should be a numpy array with dtype=object containing ParityHandle objects
    assert isinstance(result, np.ndarray), \
        f"Parity of QuantumArray measurements should return numpy array, got {type(result)}"
    assert result.dtype == object, f"Expected dtype=object, got {result.dtype}"
    assert all(isinstance(item, ParityHandle) for item in result.flat), \
        f"All elements should be ParityHandle objects"
    
    # The circuit should have classical bits ONLY for the measurements
    # 4 measurements (2 from each QuantumArray)
    assert len(qc.clbits) == 4, \
        f"Expected 4 classical bits (4 measurements), got {len(qc.clbits)}"
    
    # Count ParityOperations in circuit data
    parity_ops = [op for op in qc.data if isinstance(op.op, ParityOperation)]
    assert len(parity_ops) == 2, \
        f"Expected 2 ParityOperations, got {len(parity_ops)}"
    
    # Test with single-dimensional arrays
    @make_jaspr  
    def array_test_1d():
        qa_0 = QuantumArray(qtype=QuantumBool(), shape=(3,))
        qa_1 = QuantumArray(qtype=QuantumBool(), shape=(3,))
        meas_0 = measure(qa_0)
        meas_1 = measure(qa_1)
        return parity(meas_0, meas_1)
    
    jaspr = array_test_1d()
    result, qc = jaspr.to_qc()
    
    assert isinstance(result, np.ndarray), \
        f"Parity result should be numpy array, got {type(result)}"
    assert result.shape == (3,), f"Expected shape (3,), got {result.shape}"
    
    # 6 measurements (parity tracked separately)
    assert len(qc.clbits) == 6, \
        f"Expected 6 classical bits (6 measurements), got {len(qc.clbits)}"
    
    # 3 ParityOperations
    parity_ops = [op for op in qc.data if isinstance(op.op, ParityOperation)]
    assert len(parity_ops) == 3, \
        f"Expected 3 ParityOperations, got {len(parity_ops)}"


def test_quantum_array_measurement_operations():
    """Test various operations on QuantumArray measurement results."""
    import numpy as np
    import jax.numpy as jnp
    
    # Test: Element access from QuantumArray measurement
    def element_access_test():
        qa = QuantumArray(qtype=QuantumBool(), shape=(3,))
        meas = measure(qa)
        return meas[1]  # Access second element
    
    jaspr = make_jaspr(element_access_test)()
    result, qc = jaspr.to_qc()
    
    from qrisp import Clbit
    assert isinstance(result, Clbit), \
        f"Element access should return Clbit, got {type(result)}"
    
    # Test: Concatenate QuantumArray measurements
    def concat_test():
        qa_0 = QuantumArray(qtype=QuantumBool(), shape=(2,))
        qa_1 = QuantumArray(qtype=QuantumBool(), shape=(3,))
        meas_0 = measure(qa_0)
        meas_1 = measure(qa_1)
        combined = jnp.concatenate([meas_0, meas_1])
        return combined
    
    jaspr = make_jaspr(concat_test)()
    result, qc = jaspr.to_qc()
    
    assert isinstance(result, np.ndarray), \
        f"Concatenated measurements should be numpy array, got {type(result)}"
    assert result.shape == (5,), f"Expected shape (5,), got {result.shape}"
    # All should be Clbit objects
    assert all(isinstance(item, Clbit) for item in result.flat), \
        f"All values should be Clbit objects"


def test_processed_parity_returns_processed_measurement():
    """Test that parity on processed measurements returns ProcessedMeasurement."""
    import jax.numpy as jnp
    from qrisp.jasp import parity
    
    def parity_on_processed():
        qv = QuantumVariable(2)
        
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        
        # Put measurements in array and process with bitwise ops
        arr = jnp.array([m0, m1])
        processed = arr & True  # Bitwise AND - marks as processed
        
        # Use processed values in parity - should return ProcessedMeasurement
        p = parity(processed[0], processed[1])
        
        return p
    
    jaspr = make_jaspr(parity_on_processed)()
    result, qc = jaspr.to_qc()
    
    # Should return ProcessedMeasurement, not raise an error
    assert isinstance(result, ProcessedMeasurement), \
        f"Parity on processed measurements should return ProcessedMeasurement, got {type(result)}"
    
    # Circuit should still have the measurements
    measure_ops = [instr for instr in qc.data if instr.op.name == "measure"]
    assert len(measure_ops) == 2, f"Expected 2 measurement operations, got {len(measure_ops)}"


def test_processed_measurement_in_control_raises_error():
    """Test that using processed measurements for control raises an error."""
    import jax.numpy as jnp
    import pytest
    
    def control_with_processed():
        qv = QuantumVariable(2)
        
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        
        # Put measurements in array and process
        arr = jnp.array([m0, m1])
        processed = arr & True  # Bitwise - marks as processed
        
        # Try to use processed value for control (circuit-influencing)
        with control(processed[0]):
            x(qv[0])
        
        return qv
    
    jaspr = make_jaspr(control_with_processed)()
    
    # Should raise an error about real-time feedback
    try:
        jaspr.to_qc()
        assert False, "Expected an exception for control with processed measurement"
    except Exception as e:
        # Should mention something about real-time feedback or processed measurements
        assert "real-time feedback" in str(e).lower() or "processed" in str(e).lower(), \
            f"Expected error about real-time feedback or processed measurements, got: {e}"


def test_nested_qache_with_measurements():
    """Test that nested @qache functions with measurements work correctly.
    
    This tests the scenario where a @qache function measures qubits and returns
    the results, which are then collected in a loop by an outer function.
    This is a common pattern for syndrome measurements in error correction.
    """
    import jax.numpy as jnp
    from qrisp import reset, cx
    
    N = 4
    
    def measure_stabilizer(data_qb_0, data_qb_1, ancilla):
        reset(ancilla)
        cx(data_qb_0, ancilla)
        cx(data_qb_1, ancilla)
    
    @qache
    def syndrome_round(data, ancilla):
        for i in range(ancilla.size):
            measure_stabilizer(data[i][0], data[i+1][0], ancilla[i][0])
        return measure(ancilla)
    
    def multi_round(data, ancilla, amount):
        syndrome_list = []
        for i in range(amount):
            syndrome_list.append(syndrome_round(data, ancilla))
        return jnp.vstack(syndrome_list)
    
    def main():
        data = QuantumArray(qtype=QuantumBool(), shape=(N,))
        ancilla = QuantumArray(qtype=QuantumBool(), shape=(N-1,))
        return multi_round(data, ancilla, 3)
    
    # This should not raise an error
    jaspr = make_jaspr(main)()
    result, qc = jaspr.to_qc()
    
    # jnp.vstack processes the MeasurementArrays, so we get ProcessedMeasurements
    # The result is an array of 3 ProcessedMeasurement objects (one per round)
    assert hasattr(result, 'shape'), f"Expected result with shape, got {type(result)}"
    assert result.shape == (3,), f"Expected shape (3,), got {result.shape}"
    
    # All elements should be ProcessedMeasurement
    for elem in result:
        assert isinstance(elem, ProcessedMeasurement), \
            f"Expected ProcessedMeasurement, got {type(elem)}"
    
    # Verify the circuit has measurements: 3 rounds * (N-1) ancilla qubits = 9
    measure_ops = [instr for instr in qc.data if instr.op.name == "measure"]
    assert len(measure_ops) == 3 * (N-1), \
        f"Expected {3 * (N-1)} measurements, got {len(measure_ops)}"
    
    # Verify the circuit has reset operations
    reset_ops = [instr for instr in qc.data if instr.op.name == "reset"]
    assert len(reset_ops) == 3 * (N-1), \
        f"Expected {3 * (N-1)} resets, got {len(reset_ops)}"

