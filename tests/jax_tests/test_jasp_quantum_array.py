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

import operator
import pytest
from qrisp import *
from qrisp.jasp import *


def test_jasp_quantum_array():

    @jaspify
    def main():
        qf = QuantumFloat(5)
        qa = QuantumArray(shape=(2, 5), qtype=qf)
        qa[0, :] = np.arange(5)
        return measure(qa)

    assert np.all(main() == np.array([[0, 1, 2, 3, 4], [0, 0, 0, 0, 0]]))

    @jaspify
    def main(k, l):

        qa = QuantumArray(qtype=QuantumFloat(5), shape=(5, 3))

        for i in range(qa.shape[0]):
            for j in range(qa.shape[1]):
                qa[i, j][:] = i * j

        return measure(qa[:4][k][l])

    assert main(3, 2) == 6

    # Test qaching behavior of quantum arrays
    @qache
    def inner(qa):
        for i in range(qa.shape[0]):
            for j in range(qa.shape[1]):
                qa[i, j][:] = i * j

    # test slicing
    @terminal_sampling
    def main(k, l):

        qa = QuantumArray(qtype=QuantumFloat(5), shape=(5, 3))

        inner(qa)

        return qa[:4][k][l]

    assert main(3, 2) == {6.0: 1.0}

    @jaspify
    def main():

        qa = QuantumArray(qtype=QuantumFloat(5), shape=(5, 5, 5))

        for i in range(qa.shape[0]):
            for j in range(qa.shape[1]):
                qa[0, i, j][:] = i * j

        return measure(qa[0])

    assert np.all(main() == np.tensordot(np.arange(5), np.arange(5), ((), ())))

    # test swapaxes/transpose
    @terminal_sampling
    def main(k, l):

        qa = QuantumArray(qtype=QuantumFloat(5), shape=(5, 5))

        qa.transpose()

        for i in range(qa.shape[0]):
            for j in range(qa.shape[1]):
                qa[i, j][:] = i * j

        qa = qa.swapaxes(0, 1)
        return qa[k, l]

    assert main(3, 3) == {9.0: 1.0}

    # test duplicate

    @jaspify
    def main():

        qtype = QuantumFloat(5, -2)
        q_array_0 = QuantumArray(qtype, (2, 2))

        q_array_0[:] = np.eye(2)
        q_array_1 = q_array_0.duplicate()
        q_array_2 = q_array_0.duplicate(init=True)

        return q_array_0.measure(), q_array_1.measure(), q_array_2.measure()

    res = main()
    assert np.all(res[0] == np.eye(2))
    assert np.all(res[1] == np.zeros((2, 2)))
    assert np.all(res[2] == np.eye(2))


def test_error_invalid_slice_step():
    """Test that an error is raised when slicing with a step different from 1"""

    @jaspify
    def main():
        qf = QuantumFloat(4)
        qf[::-1]

    with pytest.raises(NotImplementedError, match="Slicing with DynamicQubitArray only supports step=1"):
        main()


def test_error_matmul_invalid_qtype():
    """Test that an error is raised when trying to apply matrix multiplication to a QuantumArray of unsupported qtype"""

    @jaspify
    def main():
        qa_float = QuantumArray(QuantumFloat(2), shape=(2, 2))
        qa_float @ qa_float

    with pytest.raises(
        NotImplementedError,
        match="Matrix multiplication between QuantumArrays of QuantumFloat in tracing mode is not supported",
    ):
        main()


def test_injection():
    @jaspify
    def test():
        a_array = QuantumArray(QuantumFloat(4), shape=(4, 4))
        x(a_array)
        b_array = QuantumArray(QuantumFloat(4), shape=(4, 4))
        h(b_array)
        r_array = QuantumArray(QuantumBool(), shape=(4, 4))

        (r_array << (lambda a, b: a == b))(a_array, b_array)
        return measure(r_array), measure(a_array), measure(b_array)

    r, a, b = test()
    assert (r == (a == b)).all()


def test_element_wise_addition_injection_qm():
    @jaspify
    def test():
        I = np.eye(4, dtype=int)
        a_array = QuantumArray(QuantumModulus(7), shape=(4, 4))
        a_array[:] = I
        b_array = QuantumArray(QuantumModulus(7), shape=(4, 4))
        b_array[:] = I
        r_array = QuantumArray(QuantumModulus(7), shape=(4, 4))

        (r_array << (lambda a, b: a + b))(a_array, b_array)
        return measure(r_array), measure(a_array), measure(b_array)

    r, a, b = test()
    assert (r == (a + b) % 7).all()


def test_element_wise_addition_injection():
    @jaspify
    def test():
        a_array = QuantumArray(QuantumFloat(4), shape=(4, 4))
        x(a_array)
        b_array = QuantumArray(QuantumFloat(4), shape=(4, 4))
        h(b_array)
        r_array = QuantumArray(QuantumFloat(6), shape=(4, 4))

        (r_array << (lambda a, b: a + b))(a_array, b_array)
        return measure(r_array), measure(a_array), measure(b_array)

    r, a, b = test()
    assert (r == a + b).all()


#
# Element-wise arithmetic
#

# Define the set of operators to test
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
]
rhs_type = ["quantum", "classical"]
instances = [
    # Instances without overflow because of different overflow behavior of quantum and classical addition/subtraction for QuantumFloat
    pytest.param((np.array([[3, 4], [5, 6]]), np.array([[1, 2], [3, 4]]), 4), id="QuantumFloat; array RHS"),
    pytest.param((np.array([[1, 2], [2, 1]]), 1, 3), id="QuantumFloat; scalar RHS"),
]


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("rhs_type", rhs_type)
@pytest.mark.parametrize("instance", instances)
def test_quantum_array_element_wise_ops(op, rhs_type, instance):
    """Test element-wise operations on QuantumArrays of QuantumFloat against their classical counterparts."""

    a_c, b_c, size = instance

    @jaspify
    def main():

        # Initialize QuantumArrays
        qtype = QuantumFloat(size)
        a_array = QuantumArray(qtype, shape=(2, 2))
        a_array[:] = a_c

        if rhs_type == "quantum":
            if isinstance(b_c, (np.ndarray, jnp.ndarray)):
                b_array = QuantumArray(qtype, shape=(2, 2))
                b_array[:] = b_c
                rhs_operand = b_array
            else:
                b_qv = qtype.duplicate()
                b_qv[:] = b_c
                rhs_operand = b_qv
        else:
            rhs_operand = b_c

        # Execute quantum operation
        r_array = op(a_array, rhs_operand)
        return measure(r_array)

    # Validate measurements
    if op == operator.mul and rhs_type == "classical":
        with pytest.raises(
            NotImplementedError, match="Quantum-classical multiplication is not supported for non-QuantumModulus types"
        ):
            main()
    else:
        measured = main()
        expected = op(a_c, b_c)
        assert np.array_equal(measured, expected)


instances = [
    pytest.param((np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]]), 7), id="QuantumModulus; array RHS"),
    pytest.param((np.array([[0, 1], [2, 1]]), 2, 3), id="QuantumModulus; scalar RHS"),
]


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("rhs_type", rhs_type)
@pytest.mark.parametrize("instance", instances)
def test_quantum_array_element_wise_ops_qm(op, rhs_type, instance):
    """Test element-wise operations on QuantumArrays of QuantumModulus against their classical counterparts."""

    if op == operator.mul and rhs_type == "quantum":
        # qq multiplication fixed in separate pull request, but for now we skip this test to avoid CI failures
        pytest.skip("Quantum-quantum multiplication for QuantumModulus is currently unsupported in Jasp.")

    a_c, b_c, modulus = instance

    @jaspify
    def main():

        # Initialize QuantumArrays
        qtype = QuantumModulus(modulus)
        a_array = QuantumArray(qtype, shape=(2, 2))
        a_array[:] = a_c

        if rhs_type == "quantum":
            if isinstance(b_c, (np.ndarray, jnp.ndarray)):
                b_array = QuantumArray(qtype, shape=(2, 2))
                b_array[:] = b_c
                rhs_operand = b_array
            else:
                b_qv = qtype.duplicate()
                b_qv[:] = b_c
                rhs_operand = b_qv
        else:
            rhs_operand = b_c

        # Execute quantum operation
        r_array = op(a_array, rhs_operand)
        return measure(r_array)

    # Calculate classical reference
    expected_c = op(a_c, b_c) % modulus

    # Validate measurements
    r_array = main()

    assert np.array_equal(r_array, expected_c)


bool_ops = [
    operator.and_,
    operator.or_,
    operator.xor,  # &, |, ^
]


@pytest.mark.parametrize("op", bool_ops)
def test_quantum_array_element_wise_bool_ops(op):
    """Test element-wise boolean operations on QuantumArrays against their classical counterparts."""

    a_c = np.array([[True, False], [False, True]])
    b_c = np.array([[True, True], [False, False]])

    @jaspify
    def main():

        # Initialize QuantumArrays
        qtype = QuantumBool()
        a_array = QuantumArray(qtype, shape=(2, 2))
        b_array = QuantumArray(qtype, shape=(2, 2))

        a_array[:] = a_c
        b_array[:] = b_c

        # Execute quantum operation
        r_array = op(a_array, b_array)
        return measure(r_array)

    # Calculate classical reference
    expected_c = op(a_c, b_c)

    # Validate measurements
    r_array = main()

    assert np.array_equal(r_array, expected_c)


ops = [
    operator.iadd,
    operator.isub,
    operator.imul,  # +=, -=, *=
]
rhs_types = ["quantum", "classical"]
instances = [
    pytest.param((np.array([[3, 4], [5, 6]]), np.array([[1, 2], [3, 4]]), 4), id="QuantumFloat; array RHS"),
    pytest.param((np.array([[1, 2], [2, 1]]), 1, 3), id="QuantumFloat; scalar RHS"),
]


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("rhs_type", rhs_types)
@pytest.mark.parametrize("instance", instances)
def test_quantum_array_element_wise_inplace_ops(op, rhs_type, instance):
    """Test element-wise in-place operations on QuantumArrays of QuantumFloat against classical counterparts."""

    a_c_ref, b_c_ref, size = instance
    a_c = a_c_ref.copy()
    b_c = b_c_ref.copy() if isinstance(b_c_ref, np.ndarray) else b_c_ref

    def main_classical(a_c, b_c):
        op(a_c, b_c)
        return a_c

    @jaspify
    def main_quantum():

        # Initialize QuantumArrays
        qtype = QuantumFloat(size)
        a_array = QuantumArray(qtype, shape=(2, 2))
        a_array[:] = a_c

        if rhs_type == "quantum":
            if isinstance(b_c, (np.ndarray, jnp.ndarray)):
                b_array = QuantumArray(qtype, shape=(2, 2))
                b_array[:] = b_c
                rhs_operand = b_array
            else:
                b_qv = qtype.duplicate()
                b_qv[:] = b_c
                rhs_operand = b_qv
        else:
            rhs_operand = b_c

        # Execute quantum operation
        op(a_array, rhs_operand)
        return measure(a_array)

    # Validate measurements
    if op == operator.imul and rhs_type == "quantum":
        with pytest.raises(TypeError, match="Quantum-quantum in-place multiplication is not supported"):
            main_quantum()
    elif op == operator.imul and rhs_type == "classical":
        with pytest.raises(
            NotImplementedError,
            match="Quantum-classical in-place multiplication is not supported in tracing mode for non-QuantumModulus types",
        ):
            main_quantum()
    else:
        measured = main_quantum()
        expected = main_classical(a_c, b_c)
        assert np.array_equal(measured, expected)


instances = [
    pytest.param((np.array([[3, 4], [5, 6]]), np.array([[1, 2], [3, 4]]), 7), id="QuantumModulus; array RHS"),
    pytest.param((np.array([[1, 2], [2, 1]]), 1, 3), id="QuantumModulus; scalar RHS"),
]


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("rhs_type", rhs_types)
@pytest.mark.parametrize("instance", instances)
def test_quantum_array_element_wise_inplace_ops_qm(op, rhs_type, instance):
    """Test element-wise in-place operations on QuantumArrays of QuantumModulus against classical counterparts."""

    a_c_ref, b_c_ref, modulus = instance
    a_c = a_c_ref.copy()
    b_c = b_c_ref.copy() if isinstance(b_c_ref, np.ndarray) else b_c_ref

    def main_classical(a_c, b_c):
        op(a_c, b_c)
        a_c %= modulus
        return a_c

    @jaspify
    def main_quantum():

        # Initialize QuantumArrays
        qtype = QuantumModulus(modulus)
        a_array = QuantumArray(qtype, shape=(2, 2))
        a_array[:] = a_c

        if rhs_type == "quantum":
            if isinstance(b_c, (np.ndarray, jnp.ndarray)):
                b_array = QuantumArray(qtype, shape=(2, 2))
                b_array[:] = b_c
                rhs_operand = b_array
            else:
                b_qv = qtype.duplicate()
                b_qv[:] = b_c
                rhs_operand = b_qv
        else:
            rhs_operand = b_c

        # Execute quantum operation
        op(a_array, rhs_operand)
        return measure(a_array)

    # Validate measurements
    if op == operator.imul and rhs_type == "quantum":
        with pytest.raises(TypeError, match="Quantum-quantum in-place multiplication is not supported"):
            main_quantum()
    else:
        measured = main_quantum()
        expected = main_classical(a_c, b_c)
        assert np.array_equal(measured, expected)


@pytest.mark.parametrize(
    " a_c, axis",
    [
        pytest.param(np.array([[True, True], [True, True]]), 0, id="All True, axis 0"),
        pytest.param(np.array([[True, False], [True, True]]), 0, id="One False, axis 0"),
        pytest.param(np.array([[True, True], [True, True]]), 1, id="All True, axis 1"),
        pytest.param(np.array([[True, False], [True, True]]), 1, id="One False, axis 1"),
    ],
)
def test_quantum_array_all(a_c, axis):
    """Test the all() method on QuantumArrays of QuantumBool against their classical counterparts."""

    @jaspify
    def main():
        q_array = QuantumArray(QuantumBool(), shape=(2, 2))
        q_array[:] = a_c
        qbl = q_array.all(axis=axis)
        return measure(qbl)

    measured = main()
    assert np.array_equal(measured, a_c.all(axis=axis))


@pytest.mark.parametrize(
    " a_c, axis",
    [
        pytest.param(np.array([[True, True], [True, True]]), 0, id="All True, axis 0"),
        pytest.param(np.array([[True, False], [True, True]]), 0, id="One False, axis 0"),
        pytest.param(np.array([[True, True], [True, True]]), 1, id="All True, axis 1"),
        pytest.param(np.array([[True, False], [True, True]]), 1, id="One False, axis 1"),
    ],
)
def test_quantum_array_any(a_c, axis):
    """Test the any() method on QuantumArrays of QuantumBool against their classical counterparts."""

    @jaspify
    def main():
        q_array = QuantumArray(QuantumBool(), shape=(2, 2))
        q_array[:] = a_c
        qbl = q_array.any(axis=axis)
        return measure(qbl)

    measured = main()
    assert np.array_equal(measured, a_c.any(axis=axis))
