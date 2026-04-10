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

import numpy as np
import pytest
from qrisp import *
from qrisp.block_encodings import BlockEncoding
from qrisp.gqsp import pseudo_inversion


def threshold_pseudoinverse(A, threshold):
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    S_inv = np.zeros_like(S)
    valid_mask = S >= threshold
    S_inv[valid_mask] = 1.0 / S[valid_mask]
    return (Vh.conj().T * S_inv) @ U.conj().T


def test_pseudo_inversion():

    # Define non-Hermitian matrix A
    #[[0.  1.4 0.  1.1]
    # [1.1 0.  1.4 0. ]
    # [0.  1.1 0.  1.4]
    # [1.4 0.  1.1 0. ]]
    N = 4
    A = 1.4 * np.eye(N, k=1) + 1.1 * np.eye(N, k=-1)
    A[0, N-1] = 1.1
    A[N-1, 0] = 1.4

    #_, S, _ = np.linalg.svd(A)
    #print("Singular values of A: ", S)
    # Singular values of A:  [2.5 2.5 0.3 0.3]

    b = np.array([0, 1, 1, 1])

    # Define efficient block-encoding.
    def f0(x): x+=1
    def f1(x): x-=1
    BE = BlockEncoding.from_lcu(np.array([1.4, 1.1]), [f0, f1])
    # alpha = 1.4 + 1.1 = 2.5

    # Choose threshold theta > 0.3 / 2.5 = 0.12 
    # to cut off smallest singular values.
    BE_inv = pseudo_inversion(BE, eps=0.01, theta=0.4, delta=0.1)

    # Prepares operand variable in state |b>
    def prep_b():
        operand = QuantumFloat(2)
        prepare(operand, b)
        return operand

    @terminal_sampling
    def main():
        operand = prep_b()
        ancillas = BE_inv.apply(operand)
        return operand, *ancillas

    res_dict = main()

    # Post-selection on ancillas being in |0> state
    filtered_dict = {k[0]: p for k, p in res_dict.items() \
                    if all(x == 0 for x in k[1:])}
    success_prob = sum(filtered_dict.values())
    filtered_dict = {k: p / success_prob for k, p in filtered_dict.items()}

    amps = np.sqrt([filtered_dict.get(i, 0) for i in range(len(b))])

    # Compare to target values
    c = (threshold_pseudoinverse(A, 0.4 * 2.5) @ b) 
    c = c / np.linalg.norm(c)
    assert np.allclose(amps, c, atol=1e-2)
