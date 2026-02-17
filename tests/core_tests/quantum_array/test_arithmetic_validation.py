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

import pytest
import numpy as np
from qrisp import QuantumArray, QuantumFloat, QuantumBool, QuantumChar


class TestArithmeticValidation:
    """Tests for the _validate_arithmetic method and arithmetic operations."""
    
    def test_validate_qtype_requirement(self):
        """Test that operations require QuantumFloat qtype."""
        # Create a QuantumArray with non-QuantumFloat qtype
        qa_char = QuantumArray(QuantumChar(), shape=(2, 2))
        qa_float = QuantumArray(QuantumFloat(5), shape=(2, 2))
        
        # Should raise TypeError for non-QuantumFloat qtype
        with pytest.raises(TypeError, match="Element-wise operations require qtype 'QuantumFloat'"):
            qa_char + qa_float
        
        with pytest.raises(TypeError, match="Element-wise operations require qtype 'QuantumFloat'"):
            qa_char - qa_float
            
        with pytest.raises(TypeError, match="Element-wise operations require qtype 'QuantumFloat'"):
            qa_char * qa_float
    
    def test_validate_array_vs_array_qtype(self):
        """Test that array-vs-array operations require matching QuantumFloat qtypes."""
        qa_float = QuantumArray(QuantumFloat(5), shape=(2, 2))
        qa_char = QuantumArray(QuantumChar(), shape=(2, 2))
        
        # Should raise TypeError when other array has wrong qtype
        with pytest.raises(TypeError, match="Element-wise operations require both arrays to have qtype 'QuantumFloat'"):
            qa_float + qa_char
    
    def test_validate_array_vs_array_shape(self):
        """Test that array-vs-array operations require matching shapes."""
        qa1 = QuantumArray(QuantumFloat(5), shape=(2, 2))
        qa2 = QuantumArray(QuantumFloat(5), shape=(3, 3))
        
        # Should raise ValueError for shape mismatch
        with pytest.raises(ValueError, match="Shape mismatch"):
            qa1 + qa2
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            qa1 - qa2
            
        with pytest.raises(ValueError, match="Shape mismatch"):
            qa1 * qa2
    
    def test_validate_array_vs_numpy_shape(self):
        """Test that array-vs-numpy operations require matching shapes."""
        qa = QuantumArray(QuantumFloat(5), shape=(2, 2))
        np_arr_wrong_shape = np.ones((3, 3))
        
        # Should raise ValueError for shape mismatch with numpy array
        with pytest.raises(ValueError, match="Shape mismatch"):
            qa + np_arr_wrong_shape
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            qa - np_arr_wrong_shape
    
    def test_scalar_operations_no_validation_error(self):
        """Test that scalar operations pass validation."""
        qa = QuantumArray(QuantumFloat(5), shape=(2, 2))
        qa[:] = np.eye(2)
        
        # These should not raise validation errors (scalar broadcasting)
        # We won't test execution here, just that validation passes
        try:
            # Validation happens before execution
            qa._validate_arithmetic(5)
            qa._validate_arithmetic(3.14)
        except (TypeError, ValueError):
            pytest.fail("Scalar validation should not raise errors")
    
    def test_matching_shapes_pass_validation(self):
        """Test that matching shapes pass validation."""
        qa1 = QuantumArray(QuantumFloat(5), shape=(2, 3))
        qa2 = QuantumArray(QuantumFloat(5), shape=(2, 3))
        
        # Should not raise
        qa1._validate_arithmetic(qa2)
        
        # With numpy array
        np_arr = np.ones((2, 3))
        qa1._validate_arithmetic(np_arr)
    
    def test_in_place_operations_use_validation(self):
        """Test that in-place operations use _validate_arithmetic."""
        qa_float = QuantumArray(QuantumFloat(5), shape=(2, 2))
        qa_char = QuantumArray(QuantumChar(), shape=(2, 2))
        
        # Should raise TypeError for wrong qtype
        with pytest.raises(TypeError, match="Element-wise operations require"):
            qa_float += qa_char
        
        # Should raise ValueError for wrong shape
        qa_wrong_shape = QuantumArray(QuantumFloat(5), shape=(3, 3))
        with pytest.raises(ValueError, match="Shape mismatch"):
            qa_float += qa_wrong_shape
    
    def test_comparison_operations_use_validation(self):
        """Test that comparison operations use _validate_arithmetic."""
        qa_float = QuantumArray(QuantumFloat(5), shape=(2, 2))
        qa_char = QuantumArray(QuantumChar(), shape=(2, 2))
        
        # Should raise TypeError for wrong qtype
        with pytest.raises(TypeError, match="Element-wise operations require"):
            qa_float == qa_char
        
        with pytest.raises(TypeError, match="Element-wise operations require"):
            qa_float < qa_char


class TestArithmeticExecution:
    """Tests for actual arithmetic operations execution."""
    
    def test_array_plus_array(self):
        """Test element-wise addition of two QuantumArrays."""
        qa1 = QuantumArray(QuantumFloat(5), shape=(2, 2))
        qa2 = QuantumArray(QuantumFloat(5), shape=(2, 2))
        qa1[:] = np.ones((2, 2))
        qa2[:] = 2 * np.ones((2, 2))
        
        result = qa1 + qa2
        measured = result.most_likely()
        
        assert np.allclose(measured, 3 * np.ones((2, 2)))
    
    def test_array_plus_scalar(self):
        """Test element-wise addition with scalar."""
        qa = QuantumArray(QuantumFloat(5), shape=(2, 2))
        qa[:] = np.ones((2, 2))
        
        result = qa + 5
        measured = result.most_likely()
        
        assert np.allclose(measured, 6 * np.ones((2, 2)))
    
    def test_array_plus_numpy(self):
        """Test element-wise addition with numpy array."""
        qa = QuantumArray(QuantumFloat(5), shape=(2, 2))
        qa[:] = np.ones((2, 2))
        np_arr = 2 * np.ones((2, 2))
        
        result = qa + np_arr
        measured = result.most_likely()
        
        assert np.allclose(measured, 3 * np.ones((2, 2)))
    
    def test_array_minus_array(self):
        """Test element-wise subtraction."""
        qa1 = QuantumArray(QuantumFloat(5, signed=True), shape=(2, 2))
        qa2 = QuantumArray(QuantumFloat(5, signed=True), shape=(2, 2))
        qa1[:] = 5 * np.ones((2, 2))
        qa2[:] = 2 * np.ones((2, 2))
        
        result = qa1 - qa2
        measured = result.most_likely()
        
        assert np.allclose(measured, 3 * np.ones((2, 2)))
    
    def test_array_multiply_array(self):
        """Test element-wise multiplication."""
        qa1 = QuantumArray(QuantumFloat(5), shape=(2, 2))
        qa2 = QuantumArray(QuantumFloat(5), shape=(2, 2))
        qa1[:] = 3 * np.ones((2, 2))
        qa2[:] = 2 * np.ones((2, 2))
        
        result = qa1 * qa2
        measured = result.most_likely()
        
        assert np.allclose(measured, 6 * np.ones((2, 2)))
    
    def test_in_place_addition(self):
        """Test in-place addition."""
        qa = QuantumArray(QuantumFloat(5), shape=(2, 2))
        qa[:] = np.ones((2, 2))
        
        qa += 3
        measured = qa.most_likely()
        
        assert np.allclose(measured, 4 * np.ones((2, 2)))
    
    def test_in_place_with_numpy_array(self):
        """Test in-place operations with numpy arrays."""
        qa = QuantumArray(QuantumFloat(5), shape=(2, 2))
        qa[:] = np.ones((2, 2))
        np_arr = 2 * np.ones((2, 2))
        
        qa += np_arr
        measured = qa.most_likely()
        
        assert np.allclose(measured, 3 * np.ones((2, 2)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
