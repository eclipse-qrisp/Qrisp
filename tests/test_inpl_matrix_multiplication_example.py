"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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
********************************************************************************/
"""

# Created by ann81984 at 22.07.2022
from qrisp import QuantumSession, QuantumArray
from qrisp.arithmetic import QuantumFloat, inplace_matrix_app
import numpy as np


def test_inpl_matrix_multiplication_example():
    def generate_random_inv_matrix(n, bit):
        from qrisp.misc import is_inv
        import random

        found = False

        while found == False:
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    matrix[i, j] = random.randint(0, 2**bit - 1)

            det = np.round(np.linalg.det(matrix) % 2**bit)

            found = is_inv(det, bit)

        return matrix

    bit = 5
    n = 3
    qs = QuantumSession()

    qtype = QuantumFloat(bit, 0, qs=qs, signed=False)
    vector = QuantumArray(qtype, n)

    x_values = np.array([0, 2, 1])

    vector.encode(x_values)

    matrix = generate_random_inv_matrix(n, bit)

    inplace_matrix_app(vector, matrix)

    print("Matrix multiplication result:")

    expected_res = ((np.dot(matrix, x_values)) % (2**bit)).astype(int).transpose()

    print(expected_res)

    mes_res = vector.get_measurement()
    print("Circuit result:")
    print(mes_res)

    assert list(mes_res.keys())[0] == expected_res
