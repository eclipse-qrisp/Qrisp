"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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
import operator
from qrisp import QuantumArray, QuantumFloat, QuantumBool, QuantumChar, QuantumModulus

ops = [
    operator.add,
    operator.sub,
    operator.mul,  # +, -, *
    operator.eq,
    operator.ne,  # ==, !=
    operator.gt,
    operator.ge,  # >, >=
    operator.lt,
    operator.le,  # <, <=
    operator.iadd,
    operator.isub,
    operator.imul,  # +=, -=, *=
]

logical_ops = [
    operator.and_,
    operator.or_,
    operator.xor,  # &, |, ^
]


class TestArithmeticValidation:
    """Tests for the _validate_arithmetic method and arithmetic operations."""

    def test_validation_errors(self):
        """Test that _validate_arithmetic raises errors for invalid operations."""
        qa_float = QuantumArray(QuantumFloat(5), shape=(2, 2))
        qa_bool = QuantumArray(QuantumBool(), shape=(2, 2))
        qa_char = QuantumArray(QuantumChar(), shape=(2, 2))

        # Should raise TypeError for non-QuantumFloat qtype of self
        with pytest.raises(TypeError, match="Element-wise operations require qtype 'QuantumFloat'"):
            qa_char._validate_arithmetic(qa_float)

        # Should raise TypeError for non-QuantumFloat qtype of other array
        with pytest.raises(TypeError, match="Element-wise operations require both arrays to have qtype 'QuantumFloat'"):
            qa_float._validate_arithmetic(qa_char)

        # Should raise TypeError for non-QuantumBool qtype of self
        with pytest.raises(TypeError, match="Element-wise operations require qtype 'QuantumBool'"):
            qa_float._validate_arithmetic(qa_bool, mode="bool")

        # Should raise TypeError for non-QuantumBool qtype of other array
        with pytest.raises(TypeError, match="Element-wise operations require both arrays to have qtype 'QuantumBool'"):
            qa_bool._validate_arithmetic(qa_float, mode="bool")

        # Should raise ValueError for shape mismatch
        qa_wrong_shape = QuantumArray(QuantumFloat(5), shape=(3, 3))
        with pytest.raises(ValueError, match="Shape mismatch"):
            qa_float._validate_arithmetic(qa_wrong_shape)

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

    @pytest.mark.parametrize("op", ops)
    def test_validate_qtype_requirement(self, op):
        """Test that operations require QuantumFloat qtype."""
        # Create a QuantumArray with non-QuantumFloat qtype
        qa_char = QuantumArray(QuantumChar(), shape=(2, 2))
        qa_float = QuantumArray(QuantumFloat(5), shape=(2, 2))

        # Should raise TypeError for non-QuantumFloat qtype
        with pytest.raises(TypeError, match="Element-wise operations require qtype 'QuantumFloat'"):
            op(qa_char, qa_float)

    @pytest.mark.parametrize("op", ops)
    def test_validate_array_vs_array_qtype(self, op):
        """Test that array-vs-array operations require matching QuantumFloat qtypes."""
        qa_float = QuantumArray(QuantumFloat(5), shape=(2, 2))
        qa_char = QuantumArray(QuantumChar(), shape=(2, 2))

        # Should raise TypeError when other array has wrong qtype
        with pytest.raises(TypeError, match="Element-wise operations require both arrays to have qtype 'QuantumFloat'"):
            op(qa_float, qa_char)

    @pytest.mark.parametrize("op", logical_ops)
    def test_validate_qtype_requirement_logical(self, op):
        """Test that logical operations require QuantumBool qtype."""
        # Create a QuantumArray with non-QuantumBool qtype
        qa_char = QuantumArray(QuantumChar(), shape=(2, 2))
        qa_bool = QuantumArray(QuantumBool(), shape=(2, 2))

        # Should raise TypeError for non-QuantumBool qtype
        with pytest.raises(TypeError, match="Element-wise operations require qtype 'QuantumBool'"):
            op(qa_char, qa_bool)

    @pytest.mark.parametrize("op", logical_ops)
    def test_validate_array_vs_array_qtype_logical(self, op):
        """Test that array-vs-array logical operations require matching QuantumBool qtypes."""
        qa_bool = QuantumArray(QuantumBool(), shape=(2, 2))
        qa_char = QuantumArray(QuantumChar(), shape=(2, 2))

        # Should raise TypeError when other array has wrong qtype
        with pytest.raises(TypeError, match="Element-wise operations require both arrays to have qtype 'QuantumBool'"):
            op(qa_bool, qa_char)

    @pytest.mark.parametrize("op", ops)
    def test_validate_array_vs_array_shape(self, op):
        """Test that array-vs-array operations require matching shapes."""
        qa1 = QuantumArray(QuantumFloat(5), shape=(2, 2))
        qa2 = QuantumArray(QuantumFloat(5), shape=(3, 3))

        # Should raise ValueError for shape mismatch
        with pytest.raises(ValueError, match="Shape mismatch"):
            op(qa1, qa2)

    @pytest.mark.parametrize("op", ops)
    def test_validate_array_vs_numpy_shape(self, op):
        """Test that array-vs-numpy operations require matching shapes."""
        qa = QuantumArray(QuantumFloat(5), shape=(2, 2))
        np_arr_wrong_shape = np.ones((3, 3))

        # Should raise ValueError for shape mismatch with numpy array
        with pytest.raises(ValueError, match="Shape mismatch"):
            op(qa, np_arr_wrong_shape)

    def test_reduction_validation(self):
        """Test that all and any methods validate qtype."""
        qa_bool = QuantumArray(QuantumBool(), shape=(2, 2))
        qa_float = QuantumArray(QuantumFloat(5), shape=(2, 2))

        # Should raise TypeError for wrong qtype
        with pytest.raises(TypeError, match="Reduction operation 'all' requires qtype 'QuantumBool'"):
            qa_float.all()

        with pytest.raises(TypeError, match="Reduction operation 'any' requires qtype 'QuantumBool'"):
            qa_float.any()

    def test_matmul_validation(self):
        """Test that matrix multiplication validates shapes and qtypes."""
        qa1 = QuantumArray(QuantumFloat(5), shape=(2, 3))
        qa2 = QuantumArray(QuantumFloat(5), shape=(4, 2))

        # Should raise ValueError for incompatible shapes
        with pytest.raises(ValueError, match="Incompatible shapes for matrix multiplication"):
            qa1 @ qa2

        # Should raise TypeError for non-QuantumFloat/QuantumModulus qtype of self
        with pytest.raises(TypeError, match="Matrix multiplication requires qtype 'QuantumFloat' or 'QuantumModulus'"):
            qa_char = QuantumArray(QuantumChar(), shape=(3, 2))
            qa_char @ qa1

        # Should raise TypeError for non-QuantumFloat/QuantumModulus qtype of other array
        with pytest.raises(
            TypeError,
            match="Matrix multiplication requires both arrays to have qtype 'QuantumFloat' or 'QuantumModulus'",
        ):
            qa_char = QuantumArray(QuantumChar(), shape=(3, 2))
            qa1 @ qa_char

        # Should raise NotImplementedError for QuantumModulus self with QuantumArray other
        qa_modulus = QuantumArray(QuantumModulus(5), shape=(3, 3))
        with pytest.raises(
            NotImplementedError,
            match="Matrix multiplication between a QuantumArray of QuantumModulus and another QuantumArray is not supported",
        ):
            qa_modulus @ qa_modulus


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
