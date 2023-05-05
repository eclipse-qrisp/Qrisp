"""
/*********************************************************************
* Copyright (c) 2023 the Qrisp Authors
*
* This program and the accompanying materials are made
* available under the terms of the Eclipse Public License 2.0
* which is available at https://www.eclipse.org/legal/epl-2.0/
*
* SPDX-License-Identifier: EPL-2.0
**********************************************************************/
"""


import numpy as np

from qrisp import QuantumArray
from qrisp.arithmetic import QuantumFloat, inplace_matrix_app


def generate_random_inv_matrix(n, bit):
    import random

    from qrisp.misc import is_inv

    found = False

    while found is False:
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = random.randint(0, 2**bit - 1)

        det = np.round(np.linalg.det(matrix) % 2**bit)

        found = is_inv(det, bit)

    return matrix


bit = 5
n = 3

qtype = QuantumFloat(bit)
vector = QuantumArray(qtype, n)

vector[:] = [0, 2, 1]

x_values = [0, 2, 1]

matrix = generate_random_inv_matrix(n, bit)

inplace_matrix_app(vector, matrix)

print("Matrix multiplication result:")
print(((np.dot(matrix, x_values)) % (2**bit)).astype(int).transpose())

print("Circuit result:")
print(vector.get_measurement())
