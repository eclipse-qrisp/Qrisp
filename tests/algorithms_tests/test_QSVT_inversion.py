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
from qrisp import QuantumVariable, QuantumFloat, gphase, conjugate
from qrisp.alg_primitives import qswitch, prepare
from qrisp.algorithms.qsvt import QSVT_inversion
from qrisp.jasp import terminal_sampling
from qrisp.operators import QubitOperator
from qrisp.block_encodings import BlockEncoding

def rand_binary_with_forced_one(n):
    b = np.random.randint(0, 2, size=n)
    if not b.any():
        b[np.random.randint(n)] = 1
    return b

def test_hermitian_matrix_pauli_block_encoding():

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

    H = QubitOperator.from_matrix(A, reverse_endianness=True)
    BA = H.pauli_block_encoding()

    BA_inv = QSVT_inversion(BA, 0.01, np.linalg.cond(A))

    def prep_b():
        operand = QuantumVariable(2)
        prepare(operand, b)
        return operand
    @terminal_sampling()
    def main():
        operand = BA_inv.apply_rus(prep_b)()
        return operand

    res_dict = main()

    for k, v in res_dict.items():
        res_dict[k] = v**0.5

    q = np.array([res_dict.get(key, 0) for key in range(n)])
    c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)

    assert np.linalg.norm(q-c) < 5e-1

def test_tridiagonal_shifted_custom_block_encoding():

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

    def U(case, operand):
            with conjugate(prepare)(case, np.sqrt(coeffs/alpha)):
                qswitch(operand, case, unitaries)
    
    BE = BlockEncoding(U, [QuantumFloat(2)], alpha)

    BE_inv = QSVT_inversion(BE, 0.01, np.linalg.cond(A))
    
    # Prepare initial system state |0>
    def prep_b():
        operand = QuantumFloat(2)
        prepare(operand, b)
        return operand
    
    @terminal_sampling()
    def main():
        operand = BE_inv.apply_rus(prep_b)()
        return operand

    res_dict = main()

    for k, v in res_dict.items():
        res_dict[k] = v**0.5

    q = np.array([res_dict.get(key, 0) for key in range(n)])
    c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)

    assert np.linalg.norm(q-c) < 1e-2
