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

# Created by ann81984 at 06.05.2022
import numpy as np

from qrisp import QuantumArray, QuantumFloat, dot, tensordot

np.random.seed(42)  # Deterministic for reproducible test results


def test_matrix_multiplication_example():
    qf = QuantumFloat(3, 0, signed=False)

    test_1 = QuantumArray(qf, shape=(2,))
    test_2 = QuantumArray(qf, shape=(2,))

    test_1[:] = [2, 3]
    test_2[:] = [1, 1]
    # h(test_1)
    # h(test_2)
    print(test_1)
    assert (type(test_1[0]).__name__ and type(test_1[1]).__name__) == "QuantumFloat" and type(
        test_1
    ).__name__ == "QuantumArray"
    assert str(test_1[0]) == "{2: 1.0}" and str(test_1[1]) == "{3: 1.0}"
    # test_1 = QuantumArray(qf, 3)
    # test_1.init_state([(np.array([1,2,3]), 1), (np.array([1,2,2]), 1j)])

    # test_1[:] = np.array([[5,6,2],[3,4,1],[1,1,0],[2,3,5]])

    # test_2 = QuantumArray(qf)
    # test_2[:] = np.eye(3)

    # res = test_1 @ test_2
    # np.sin(test_1)
    res = dot(test_1, test_2)
    print(res)
    assert type(res).__name__ == "QuantumFloat"
    assert str(res) == "{5: 1.0}"
    # test = res.get_measurement(statevector = True, plot = True)

    qf = QuantumFloat(3, 0, signed=False)

    test_1 = QuantumArray(qf, shape=(4, 4))
    test_1.encode(np.eye(4))

    test_1 = test_1.reshape(2, 2, 2, 2)

    test_2 = QuantumArray(qf, shape=(2, 2))
    test_2.encode(np.eye(2))

    res = tensordot(test_1, test_2, (-1, 0))

    qtype = QuantumFloat(3, signed=False)

    test_0 = np.round(np.random.rand(3, 4) * 5)
    test_1 = np.round(np.random.rand(4, 2) * 5)

    q_array = QuantumArray(qtype=qtype, shape=test_1.shape)

    q_array[:] = test_1

    (test_0 @ q_array)

    # test = semi_classic_matrix_multiplication(test_0, q_array)


def _certain_outcome(res):
    """Return the single measured outcome of a deterministic result as an array.

    Works for both QuantumArray (-> OutcomeArray) and scalar QuantumFloat outcomes.
    """
    measurement = res.get_measurement()
    assert len(measurement) == 1
    ((outcome, probability),) = measurement.items()
    assert probability == 1.0
    return np.array(outcome)


def test_dot_1d_2d_regression():
    """Regression test for issue #638: dot() on a 1-D and a 2-D QuantumArray.

    This used to raise "TypeError: 'QuantumArrayIterator' object is not iterable":
    QuantumArray.reshape() rejected the ``order`` keyword that np.reshape forwards,
    forcing numpy into an array-conversion fallback. It mirrors the example in
    MatrixMultiplication.rst, which previously could not be executed.
    """
    qf = QuantumFloat(3)

    a = QuantumArray(qf, shape=2)
    b = QuantumArray(qf, shape=(2, 2))
    a[:] = [2, 3]
    b[:] = [[0, 2], [1, 0]]

    assert np.array_equal(_certain_outcome(dot(a, b)), np.array([[3, 4]]))


def test_dot_2d_2d_documented_example():
    """Cover the matrix-matrix example from the dot() docstring, which had no test."""
    qf = QuantumFloat(3, 0, signed=True)

    a = QuantumArray(qf, shape=(2, 2))
    b = QuantumArray(qf, shape=(2, 2))
    a[:] = [[0, 1], [1, 0]]
    b[:] = [[1, 0], [0, -1]]

    assert np.array_equal(_certain_outcome(dot(a, b)), np.array([[0, -1], [1, 0]]))


def test_quantum_array_iterator_is_iterable():
    """Iterating a QuantumArray must yield a protocol-compliant iterator.

    An iterator must return itself from __iter__; otherwise calling iter() on it
    (as numpy's array-conversion path does) raises "'QuantumArrayIterator' object
    is not iterable" (issue #638).
    """
    qf = QuantumFloat(3)
    q_array = QuantumArray(qf, shape=3)

    iterator = iter(q_array)
    assert iter(iterator) is iterator
