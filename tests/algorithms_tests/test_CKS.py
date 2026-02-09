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
from qrisp import QuantumFloat, gphase, multi_measurement, prepare
from qrisp.algorithms.cks import CKS
from qrisp.block_encodings import BlockEncoding
from qrisp.jasp import terminal_sampling


def rand_binary_with_forced_one(n):
    b = np.random.randint(0, 2, size=n)
    if not b.any():
        b[np.random.randint(n)] = 1
    return b


def test_cks_random_matrix():

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

    BE = BlockEncoding.from_array(A)
    BE_CKS = CKS(BE, 0.01, np.linalg.cond(A))

    def b_prep():
        qv = QuantumFloat(2)
        prepare(qv, b)
        return qv

    @terminal_sampling
    def main():
        qv = BE_CKS.apply_rus(b_prep)()
        return qv

    res_dict = main()
    amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])

    c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)
    assert np.linalg.norm(amps - np.abs(c)) < 1e-2


def test_cks_3_sparse_matrix_8x8():

    def tridiagonal_shifted(n, mu=1.0, dtype=float):
        I = np.eye(n, dtype=dtype)
        return (2 + mu) * I - 2*np.eye(n, k=n//2, dtype=dtype) - 2*np.eye(n, k=-n//2, dtype=dtype)
    
    n = 8
    A = tridiagonal_shifted(n, mu=3)
    b = rand_binary_with_forced_one(n)

    BE = BlockEncoding.from_array(A)
    BE_CKS = CKS(BE, 0.01, np.linalg.cond(A))

    def b_prep():
        qv = QuantumFloat(3)
        prepare(qv, b)
        return qv

    @terminal_sampling
    def main():
        qv = BE_CKS.apply_rus(b_prep)()
        return qv

    res_dict = main()
    amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])

    c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)
    assert np.linalg.norm(amps - np.abs(c)) < 1e-2


def test_cks_custom_block_encoding_hermitian():

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
    coeffs = np.array([5, 1, 1, 0])
    BE = BlockEncoding.from_lcu(coeffs, unitaries, is_hermitian=True)
    BE_CKS = CKS(BE, 0.01, np.linalg.cond(A))

    def b_prep():
        qv = QuantumFloat(2)
        prepare(qv, b)
        return qv

    @terminal_sampling
    def main():
        qv = BE_CKS.apply_rus(b_prep)()
        return qv

    res_dict = main()
    amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])

    c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)
    assert np.linalg.norm(amps - np.abs(c)) < 1e-2


# Discrete Laplace operator
def test_cks_custom_block_encoding_not_hermitian():

    def tridiagonal_shifted(n, mu=1.0, dtype=float):
        I = np.eye(n, dtype=dtype)
        A = (2 + mu) * I - np.eye(n, k=1, dtype=dtype) - np.eye(n, k=-1, dtype=dtype)
        A[0, n - 1] = -1
        A[n - 1, 0] = -1
        return A
    
    n=4
    A = tridiagonal_shifted(n, mu=3)
    b = rand_binary_with_forced_one(n)
    
    def U0(qv):
        pass

    def U1(qv):
        qv += 1
        gphase(np.pi, qv[0])

    def U2(qv):
        qv -= 1
        gphase(np.pi, qv[0])

    unitaries = [U0, U1, U2]
    coeffs = np.array([5, 1, 1, 0])
    BE = BlockEncoding.from_lcu(coeffs, unitaries, is_hermitian=False)
    BE_CKS = CKS(BE, 0.01, np.linalg.cond(A))

    def b_prep():
        qv = QuantumFloat(2)
        prepare(qv, b)
        return qv

    @terminal_sampling
    def main():
        qv = BE_CKS.apply_rus(b_prep)()
        return qv

    res_dict = main()
    amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])

    c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)
    assert np.linalg.norm(amps - np.abs(c)) < 1e-2


def test_cks_post_selection():

    A = np.array([[0.73255474, 0.14516978, -0.14510851, -0.0391581],
                [0.14516978, 0.68701415, -0.04929867, -0.00999921],
                [-0.14510851, -0.04929867, 0.76587818, -0.03420339],
                [-0.0391581, -0.00999921, -0.03420339, 0.58862043]])

    b = np.array([0, 1, 1, 1])

    BE = BlockEncoding.from_array(A)
    BE_CKS = CKS(BE, 0.01, np.linalg.cond(A))

    def b_prep():
        qv = QuantumFloat(2)
        prepare(qv, b)
        return qv
    
    def main():
        operand = b_prep()
        ancillas = BE_CKS.apply(operand)
        return operand, ancillas

    operand, ancillas = main()
    res_dict = multi_measurement([operand] + ancillas)

    # Post-selection on ancillas being in |0> state
    filtered_dict = {k[0]: p for k, p in res_dict.items() \
                    if all(x == 0 for x in k[1:])}
    success_prob = sum(filtered_dict.values())
    filtered_dict = {k: p / success_prob for k, p in filtered_dict.items()}
    amps = np.sqrt([filtered_dict.get(i, 0) for i in range(len(b))])

    c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)
    assert np.linalg.norm(amps - np.abs(c)) < 1e-2
