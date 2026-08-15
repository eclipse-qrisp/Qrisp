"""********************************************************************************
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
from collections.abc import Mapping

import numpy as np
import pytest

from qrisp import *
from qrisp.interface.measurement_result import DecodedMeasurementResult


def test_quantum_array():

    qf = QuantumFloat(5)
    qa = QuantumArray(shape=5, qtype=qf)
    for i in range(len(qa)):
        qa[i][:] = i

    assert qa.get_measurement() == {OutcomeArray([0, 1, 2, 3, 4]): 1.0}

    qf = QuantumFloat(5)
    qa = QuantumArray(shape=(2, 5), qtype=qf)
    qa[0, :] = np.arange(5)

    assert qa.most_likely()[0, :] == OutcomeArray([0, 1, 2, 3, 4])

    # test dynamic indexing

    @jaspify
    def main(k):

        qtype = QuantumFloat(8)
        q_array = QuantumArray(qtype, 10)
        q_array[:] = np.arange(10)

        for i in jrange(1, k):
            q_array[i] += q_array[i - 1]

        return measure(q_array)

    assert np.all(main(8) == np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 8.0, 9.0]))

    # Test the snippets from the documentation

    qtype = QuantumFloat(5, -2)
    q_array = QuantumArray(qtype=qtype, shape=(2, 2, 2))

    from qrisp import h

    qv = q_array[0, 0, 1]
    h(qv[0])

    assert q_array.get_measurement() == {
        OutcomeArray([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]): 0.5,
        OutcomeArray([[[0.0, 0.25], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]): 0.5,
    }

    q_array = q_array.reshape(2, 4)

    q_array_swap = np.swapaxes(q_array, 0, 1)

    assert q_array_swap.get_measurement() == {
        OutcomeArray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]): 0.5,
        OutcomeArray([[0.0, 0.0], [0.25, 0.0], [0.0, 0.0], [0.0, 0.0]]): 0.5,
    }

    q_array[1:, :] = 2 * np.ones((1, 4))

    assert q_array.get_measurement() == {
        OutcomeArray([[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]]): 0.5,
        OutcomeArray([[0.0, 0.25, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]]): 0.5,
    }

    # quantum indexing

    q_array = QuantumArray(QuantumFloat(1), shape=(4, 4))
    index_0 = QuantumFloat(2)
    index_1 = QuantumFloat(2)

    index_0[:] = 2
    index_1[:] = 1

    h(index_0[0])

    with q_array[index_0, index_1] as entry:
        x(entry)

    assert multi_measurement([index_0, index_1, q_array]) == {
        (
            2,
            1,
            OutcomeArray([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]),
        ): 0.5,
        (
            3,
            1,
            OutcomeArray([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]]),
        ): 0.5,
    }

    qtype = QuantumFloat(5, -2)
    q_array_1 = QuantumArray(qtype, (2, 2))
    q_array_2 = QuantumArray(qtype, (2, 2))
    q_array_1[:] = 2 * np.eye(2)
    q_array_2[:] = [[1, 2], [3, 4]]

    assert (q_array_1 @ q_array_2).get_measurement() == {OutcomeArray([[2.0, 4.0], [6.0, 0.0]]): 1.0}

    q_array = QuantumArray(qtype, (2, 2))
    q_array[:] = 3 * np.eye(2)
    cl_array = np.array([[1, 2], [3, 4]])
    assert (q_array @ cl_array).get_measurement() == {OutcomeArray([[3.0, 6.0], [1.0, 4.0]]): 1.0}

    # test duplicate
    q_array_0 = QuantumArray(qtype, (2, 2))
    q_array_0[:, :] = np.eye(2)
    q_array_1 = q_array_0.duplicate()
    q_array_2 = q_array_0.duplicate(init=True)

    assert q_array_1.get_measurement() == {OutcomeArray([[0.0, 0.0], [0.0, 0.0]]): 1.0}

    assert q_array_2.get_measurement() == {OutcomeArray([[1.0, 0.0], [0.0, 1.0]]): 1.0}


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
rhs_types = ["quantum", "classical"]
instances = [
    pytest.param(
        (np.array([[1, 2], [2, 1]]), np.array([[1, 1], [1, 1]]), QuantumFloat(3)), id="QuantumFloat; array RHS"
    ),
    pytest.param((np.array([[1, 2], [2, 1]]), 1, QuantumFloat(2)), id="QuantumFloat; scalar RHS"),
    pytest.param(
        (np.array([[1, 0.5], [0.5, 1]]), np.array([[1, 1], [1, 1]]), QuantumFloat(4, -1, signed=True)),
        id="Signed QuantumFloat; array RHS",
    ),
    pytest.param(
        (np.array([[1, 2], [3, 4]]), np.array([[4, 3], [2, 3]]), QuantumModulus(7)), id="QuantumModulus; array RHS"
    ),
    pytest.param((np.array([[0, 1], [2, 1]]), 2, QuantumModulus(3)), id="QuantumModulus; scalar RHS"),
]


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("rhs_type", rhs_types)
@pytest.mark.parametrize("instance", instances)
def test_quantum_array_element_wise_ops(op, rhs_type, instance):
    """Test element-wise operations on QuantumArrays of QuantumFloat/QuantumModulus against their classical counterparts."""
    a_c_ref, b_c_ref, qtype = instance
    a_c = a_c_ref.copy()
    b_c = b_c_ref.copy() if isinstance(b_c_ref, np.ndarray) else b_c_ref

    def main_classical(a_c, b_c):
        result = op(a_c, b_c)
        if isinstance(qtype, QuantumModulus):
            result = result % qtype.modulus
        return result

    def main_quantum(a_c, b_c):
        # Initialize QuantumArrays
        a_array = QuantumArray(qtype, shape=(2, 2))
        a_array[:] = a_c

        if rhs_type == "quantum":
            if isinstance(b_c, np.ndarray):
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
        return r_array.most_likely()

    # Validate measurements
    if op == operator.mul and rhs_type == "classical" and not isinstance(qtype, QuantumModulus):
        with pytest.raises(
            NotImplementedError, match="Quantum-classical multiplication is not supported for non-QuantumModulus types"
        ):
            main_quantum(a_c, b_c)
    else:
        measured = main_quantum(a_c, b_c)
        expected = main_classical(a_c, b_c)
        assert np.array_equal(measured, expected)


bool_ops = [
    operator.and_,
    operator.or_,
    operator.xor,  # &, |, ^
]


@pytest.mark.parametrize("op", bool_ops)
def test_quantum_array_element_wise_bool_ops(op):
    """Test element-wise boolean operations on QuantumArrays of QuantumBool against their classical counterparts."""
    a_c = np.array([[True, False], [False, True]])
    b_c = np.array([[True, True], [False, False]])

    # Initialize QuantumArrays
    qtype = QuantumBool()
    a_array = QuantumArray(qtype, shape=(2, 2))
    b_array = QuantumArray(qtype, shape=(2, 2))

    a_array[:] = a_c
    b_array[:] = b_c

    # Execute quantum operation
    r_array = op(a_array, b_array)

    # Calculate classical reference
    expected = op(a_c, b_c)

    # Validate measurements
    measured = r_array.most_likely()
    assert np.array_equal(measured, expected), f"Failed on operator {op.__name__}. Expected {expected}, got {measured}"


ops = [
    operator.iadd,
    operator.isub,
    operator.imul,  # +=, -=, *=
]
rhs_types = ["quantum", "classical"]
instances = [
    pytest.param(
        (np.array([[1.5, 2.0], [3.0, 4.0]]), np.array([[4.0, 3.0], [2.0, 1.0]]), QuantumFloat(8, -1, signed=True)),
        id="Signed QuantumFloat; array RHS",
    ),
    pytest.param((np.array([[1, 2], [2, 1]]), 1, QuantumFloat(3)), id="QuantumFloat; scalar RHS"),
    pytest.param(
        (np.array([[1, 2], [3, 4]]), np.array([[4, 3], [2, 1]]), QuantumModulus(7)), id="QuantumModulus; array RHS"
    ),
    pytest.param((np.array([[0, 1], [2, 1]]), 2, QuantumModulus(3)), id="QuantumModulus; scalar RHS"),
]


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("rhs_type", rhs_types)
@pytest.mark.parametrize("instance", instances)
def test_quantum_array_element_wise_inplace_ops(op, rhs_type, instance):
    """Test element-wise in-place operations on QuantumArrays of QuantumFloat/QuantumModulus against classical counterparts."""
    a_c_ref, b_c_ref, qtype = instance
    a_c = a_c_ref.copy()
    b_c = b_c_ref.copy() if isinstance(b_c_ref, np.ndarray) else b_c_ref

    def main_classical(a_c, b_c):
        op(a_c, b_c)
        if isinstance(qtype, QuantumModulus):
            a_c = a_c % qtype.modulus
        return a_c

    def main_quantum(a_c, b_c):
        # Initialize QuantumArrays
        a_array = QuantumArray(qtype, shape=(2, 2))
        a_array[:] = a_c

        if rhs_type == "quantum":
            if isinstance(b_c, np.ndarray):
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
        return a_array.most_likely()

    # Validate measurements
    if op == operator.imul and rhs_type == "quantum":
        with pytest.raises(TypeError, match="Quantum-quantum in-place multiplication is not supported"):
            main_quantum(a_c, b_c)
    else:
        measured = main_quantum(a_c, b_c)
        expected = main_classical(a_c, b_c)
        assert np.array_equal(measured, expected), (
            f"Failed on operator {op.__name__}. Expected {expected}, got {measured}"
        )


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
    q_array = QuantumArray(QuantumBool(), shape=(2, 2))
    q_array[:] = a_c
    qbl = q_array.all(axis=axis)
    measured = qbl.most_likely()
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
    q_array = QuantumArray(QuantumBool(), shape=(2, 2))
    q_array[:] = a_c
    qbl = q_array.any(axis=axis)
    measured = qbl.most_likely()
    assert np.array_equal(measured, a_c.any(axis=axis))


def test_quantum_array_element_eq():
    a_c = np.array(3 * [[0, 1, 2]])
    b_c = np.arange(0, 9).reshape((3, 3))
    a_array = QuantumArray(QuantumFloat(4), shape=(3, 3))
    a_array[:] = a_c
    b_array = QuantumArray(QuantumFloat(4), shape=(3, 3))
    b_array[:] = b_c
    r_array = a_array == b_array
    for k in r_array.get_measurement().keys():
        assert k == (a_c == b_c)


def test_quantum_array_inject_add():
    a_c = np.array(3 * [[0, 1, 2]])
    b_c = np.arange(0, 9).reshape((3, 3))
    a_array = QuantumArray(QuantumFloat(4), shape=(3, 3))
    a_array[:] = a_c
    b_array = QuantumArray(QuantumFloat(4), shape=(3, 3))
    b_array[:] = b_c
    # TODO: Result of QuantumFloat addition
    r_array = QuantumArray(QuantumFloat(6), shape=(3, 3))
    (r_array << (lambda a, b: a + b))(a_array, b_array)
    for k in r_array.get_measurement().keys():
        assert k == (a_c + b_c)

    # Test modular addition
    a_c = np.array(3 * [[0, 1, 2]])
    b_c = np.arange(0, 9).reshape((3, 3))
    a_array = QuantumArray(QuantumModulus(9), shape=(3, 3))
    a_array[:] = a_c
    b_array = QuantumArray(QuantumModulus(9), shape=(3, 3))
    b_array[:] = b_c
    r_array = QuantumArray(QuantumModulus(9), shape=(3, 3))
    (r_array << (lambda a, b: a + b))(a_array, b_array)

    for k in r_array.get_measurement().keys():
        assert k == ((a_c + b_c) % 9)


def test_modular_multiplication():
    # Test modular multiplication

    a_c = np.array(3 * [[0, 1, 2]])
    b_c = np.arange(0, 9).reshape((3, 3))
    a_array = QuantumArray(QuantumModulus(9), shape=(3, 3))
    a_array[:] = a_c
    b_array = QuantumArray(QuantumModulus(9), shape=(3, 3))
    b_array[:] = b_c
    r_array = a_array * b_array

    for k in r_array.get_measurement().keys():
        assert k == ((a_c * b_c) % 9)


def test_quantum_array_get_measurement_returns_mapping():
    """get_measurement() must return a Mapping with the correct decoded outcomes."""
    qtype = QuantumFloat(3)
    q_array = QuantumArray(qtype, shape=(2,))
    q_array[:] = [3, 5]

    result = q_array.get_measurement()

    assert isinstance(result, Mapping)
    assert isinstance(result, DecodedMeasurementResult)
    # Equality comparison with a plain dict must still work
    keys = list(result.keys())
    assert len(keys) == 1
    assert keys[0].tolist() == [3, 5]
    assert abs(result[keys[0]] - 1.0) < 1e-6


def test_quantum_array_get_measurement_compatible_with_batched_backend():
    """get_measurement() must be compatible with BatchedBackend (lazy evaluation)."""
    from qrisp.default_backend import QrispSimulatorBackend
    from qrisp.interface.measurement_result import LazyDict

    qtype = QuantumFloat(3)
    q_array = QuantumArray(qtype, shape=(2,))
    q_array[:] = [3, 5]

    bb = QrispSimulatorBackend().batched()
    res = q_array.get_measurement(backend=bb)

    # Result must be lazy before dispatch
    assert isinstance(res, LazyDict)
    assert not res._populated

    bb.dispatch()

    # After dispatch the decoded result must be correct
    keys = list(res.keys())
    assert len(keys) == 1
    assert keys[0].tolist() == [3, 5]


def test_quantum_array_injection():
    a_c = np.array(3 * [[0, 1, 2]])
    b_c = np.arange(0, 9).reshape((3, 3))
    a_array = QuantumArray(QuantumFloat(4), shape=(3, 3))
    a_array[:] = a_c
    b_array = QuantumArray(QuantumFloat(4), shape=(3, 3))
    b_array[:] = b_c
    r_array = QuantumArray(QuantumBool(), shape=(3, 3))
    (r_array << (lambda a, b: a == b))(a_array, b_array)
    for k in r_array.get_measurement().keys():
        assert k == (a_c == b_c)
