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

import numpy as np
from qrisp import QuantumFloat, gphase, prepare, multi_measurement
from qrisp.alg_primitives import qswitch
from qrisp.algorithms.cks import CKS, inner_CKS
from qrisp.jasp import terminal_sampling

def rand_binary_with_forced_one(n):
    b = np.random.randint(0, 2, size=n)
    if not b.any():
        b[np.random.randint(n)] = 1
    return b

def test_hermitian_matrix():

    def hermitian_matrix_with_eigenvalues_in_range(n, low=0.45, high=1.0):
        # Generate eigenvalues uniformly in [low, high]
        eigenvalues = np.random.uniform(low, high, size=n)

        Q, _ = np.linalg.qr(np.random.randn(n, n))

        # Construct the Hermitian matrix
        A = Q @ np.diag(eigenvalues) @ Q.T
        return A
    
    n=4
    A = hermitian_matrix_with_eigenvalues_in_range(n)
    b = rand_binary_with_forced_one(n)

    @terminal_sampling
    def main():

        x = CKS(A, b, 0.01)
        return x

    res_dict = main()

    for k, v in res_dict.items():
        res_dict[k] = v**0.5

    q = np.array([res_dict.get(key, 0) for key in range(n)])
    c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)

    assert np.linalg.norm(q-c) < 5e-1

def test_3_sparse_matrix_8x8():

    def tridiagonal_shifted(n, mu=1.0, dtype=float):
        I = np.eye(n, dtype=dtype)
        return (2 + mu) * I - 2*np.eye(n, k=n//2, dtype=dtype) - 2*np.eye(n, k=-n//2, dtype=dtype)
    
    n = 8
    A = tridiagonal_shifted(n, mu=3)
    b = rand_binary_with_forced_one(n)

    @terminal_sampling
    def main():

        x = CKS(A, b, 0.01)
        return x

    res_dict = main()

    for k, v in res_dict.items():
        res_dict[k] = v**0.5

    q = np.array([res_dict.get(key, 0) for key in range(n)])
    c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)

    assert np.linalg.norm(q-c) < 1e-2

def test_A_block_encoding():

    def tridiagonal_shifted(n, mu=1.0, dtype=float):
        I = np.eye(n, dtype=dtype)
        return (2 + mu) * I - 2*np.eye(n, k=n//2, dtype=dtype) - 2*np.eye(n, k=-n//2, dtype=dtype)
    
    n=4
    A = tridiagonal_shifted(n, mu=3)
    b = rand_binary_with_forced_one(n)
    
    def U0(qv):
        pass

    def U1(qv):
        qv += n//2
        gphase(np.pi, qv[0])

    def U2(qv):
        qv -= n//2
        gphase(np.pi, qv[0])

    unitaries = [U0, U1, U2]

    coeffs = np.array([5,1,1])
    alpha = np.sum(coeffs)

    def U_func(case, operand):
        qswitch(operand, case, unitaries)

    def G_func(case):
        prepare(case, np.sqrt(coeffs/alpha))

    BE = (U_func, G_func, 2)
    
    @terminal_sampling
    def main():

        x = CKS(BE, b, 0.01, np.linalg.cond(A))
        return x

    res_dict = main()

    for k, v in res_dict.items():
        res_dict[k] = v**0.5

    q = np.array([res_dict.get(key, 0) for key in range(n)])
    c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)

    assert np.linalg.norm(q-c) < 1e-2

def test_A_block_encoding_b_callable():

    def tridiagonal_shifted(n, mu=1.0, dtype=float):
        I = np.eye(n, dtype=dtype)
        return (2 + mu) * I - 2*np.eye(n, k=n//2, dtype=dtype) - 2*np.eye(n, k=-n//2, dtype=dtype)
    
    n=4
    A = tridiagonal_shifted(n, mu=3)
    b = rand_binary_with_forced_one(n)

    def bprep():
        operand = QuantumFloat(int(np.log2(b.shape[0])))
        prepare(operand, b)
        return operand
    
    def U0(qv):
        pass

    def U1(qv):
        qv += n//2
        gphase(np.pi, qv[0])

    def U2(qv):
        qv -= n//2
        gphase(np.pi, qv[0])

    unitaries = [U0, U1, U2]

    coeffs = np.array([5,1,1])
    alpha = np.sum(coeffs)

    def U_func(case, operand):
        qswitch(operand, case, unitaries)

    def G_func(case):
        prepare(case, np.sqrt(coeffs/alpha))

    BE = (U_func, G_func, 2)
    
    @terminal_sampling
    def main():

        x = CKS(BE, bprep, 0.01, np.linalg.cond(A))
        return x

    res_dict = main()

    for k, v in res_dict.items():
        res_dict[k] = v**0.5

    q = np.array([res_dict.get(key, 0) for key in range(n)])
    c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)

    assert np.linalg.norm(q-c) < 1e-2

def test_hermitian_matrix_post_selection():

    A = np.array([[0.73255474, 0.14516978, -0.14510851, -0.0391581],
                [0.14516978, 0.68701415, -0.04929867, -0.00999921],
                [-0.14510851, -0.04929867, 0.76587818, -0.03420339],
                [-0.0391581, -0.00999921, -0.03420339, 0.58862043]])

    b = np.array([0, 1, 1, 1])

    operand, *ancillas = inner_CKS(A, b, 0.001)
    res_dict = multi_measurement([operand, *ancillas])

    new_dict = dict()
    success_prob = 0

    for key, prob in res_dict.items():
        if all(k == 0 for k in key[1:]):
            new_dict[key[0]] = prob
            success_prob += prob

    for key in new_dict.keys():
        new_dict[key] = new_dict[key]/success_prob

    for k, v in new_dict.items():
        new_dict[k] = v**0.5

    q = np.array([new_dict.get(key, 0) for key in range(len(b))])
    c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)
    
    assert np.linalg.norm(q-c) < 1e-2


