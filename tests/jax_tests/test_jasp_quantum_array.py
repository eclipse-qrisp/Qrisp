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

    with pytest.raises(
        NotImplementedError, match="Slicing with DynamicQubitArray only supports step=1"
    ):
        main()

def test_injection():
    @jaspify
    def test():
        a_array = QuantumArray(QuantumFloat(4), shape=(4,4))
        x(a_array)
        b_array = QuantumArray(QuantumFloat(4), shape=(4,4))
        h(b_array)
        r_array = QuantumArray(QuantumBool(), shape=(4,4))

        (r_array << (lambda a,b: a==b))(a_array, b_array)
        return measure(r_array), measure(a_array), measure(b_array)

    r, a, b = test()
    assert((r[0] == (a == b)).all())

def test_element_wise_addition_injection():
    @jaspify
    def test():
        a_array = QuantumArray(QuantumFloat(4), shape=(4,4))
        x(a_array)
        b_array = QuantumArray(QuantumFloat(4), shape=(4,4))
        h(b_array)
        r_array = QuantumArray(QuantumFloat(4), shape=(4,4))

        (r_array << (lambda a,b: a+b))(a_array, b_array)
        return measure(r_array), measure(a_array), measure(b_array)

    r, a, b = test()
    assert((r == (a+b)%16).all())