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
from qrisp.gqsp import GQSVT


@pytest.mark.parametrize("poly, parity", [
    (np.array([1., 1.]), "odd"),
    (np.array([1., 2., 1.]), "even"),
    (np.array([0.,1.,0.,1.]), "odd"),
    (np.array([2.,1.,0.,1.,2.,3.]), "odd"),
])
def test_gqsvt(poly, parity):

    # Define non-Hermitian matrix A
    # [[3. 1. 0. 0.]
    # [0. 3. 1. 0.]
    # [0. 0. 3. 1.]
    # [1. 0. 0. 3.]]
    N = 4
    A = np.eye(N, k=1) + 3 * np.eye(N)
    A[N-1,0] = 1

    b = np.array([1,0,0,0])

    # Define BlockEncoding for A
    def U0(qv): pass
    def U1(qv): qv-=1
    BE = BlockEncoding.from_lcu(np.array([3,1]), [U0,U1])

    # Apply polynomial via GQSVT
    BE_poly = GQSVT(BE, poly, parity=parity)

    # Prepare initial system state |b>
    def operand_prep():
        qv = QuantumFloat(2)
        prepare(qv, b)
        return qv

    @terminal_sampling
    def main():
        operand = BE_poly.apply_rus(operand_prep)()
        return operand

    res_dict = main()
    amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])

    # Compare to target values
    # Compute the SVD
    U, S, Vh = np.linalg.svd(A)

    # Apply polynomial z + z^3 to singular values
    start = 0 if parity == "even" else 1
    S_poly = sum(c * S ** (i * 2 + start) for i, c in enumerate(list(poly)[start::2]))

    # Reconstruct transformed matrix
    if parity == "even":
        A_poly = Vh.conj().T @ np.diag(S_poly) @ Vh
    else:
        A_poly = (U @ np.diag(S_poly) @ Vh).conj().T

    res = A_poly @ b / np.linalg.norm(A_poly @ b)

    assert np.linalg.norm(np.abs(res) - amps) < 1e-2
    