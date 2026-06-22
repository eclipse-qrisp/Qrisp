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
from qrisp.algorithms.gqsp import inversion
from qrisp.algorithms.gqsp.qsvt import QSVT


def evaluate_parity_polynomial(poly, S, parity):
    """Evaluate the fixed parity polynomial on the singular values."""
    start = 0 if parity == "even" else 1
    coeffs = poly[start::2][::-1]

    S_poly = np.polyval(coeffs, S**2)
    if parity == "odd":
        S_poly *= S

    return S_poly


@pytest.mark.parametrize(
    "poly, parity",
    [
        (np.array([1.0, 1.0]), "odd"),
        (np.array([1.0, 2.0, 1.0]), "even"),
        (np.array([0.0, 1.0, 0.0, 1.0]), "odd"),
        (np.array([2.0, 1.0, 0.0, 1.0, 2.0, 3.0]), "odd"),
    ],
)
def test_qsvt(poly, parity):
    """Test QSVT on a small 4x4 matrix with a simple polynomial transformation."""

    # Define non-Hermitian matrix A
    # [[3. 1. 0. 0.]
    # [0. 3. 1. 0.]
    # [0. 0. 3. 1.]
    # [1. 0. 0. 3.]]
    N = 4
    A = np.eye(N, k=1) + 3 * np.eye(N)
    A[N - 1, 0] = 1

    b = np.array([1, 0, 0, 0])

    # Define BlockEncoding for A
    def U0(qv):
        pass

    def U1(qv):
        qv -= 1

    BE = BlockEncoding.from_lcu(np.array([3, 1]), [U0, U1])

    # Apply polynomial via QSVT
    BE_poly = QSVT(BE, poly, parity=parity)

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

    S_poly = evaluate_parity_polynomial(poly, S, parity)

    # Reconstruct transformed matrix
    if parity == "even":
        A_poly = Vh.conj().T @ np.diag(S_poly) @ Vh
    else:
        A_poly = U @ np.diag(S_poly) @ Vh

    res = A_poly @ b / np.linalg.norm(A_poly @ b)
    assert np.linalg.norm(np.abs(res) - amps) < 1e-2


def test_qsvt_inversion():
    """Test QSVT-based inversion on a small 4x4 matrix."""

    # Define non-Hermitian matrix A
    # [[3. 1. 0. 0.]
    # [0. 3. 1. 0.]
    # [0. 0. 3. 1.]
    # [1. 0. 0. 3.]]
    N = 4
    A = np.eye(N, k=1) + 3 * np.eye(N)
    A[N - 1, 0] = 1

    b = np.array([1, 0, 0, 0])

    # Define BlockEncoding for A
    def U0(qv):
        pass

    def U1(qv):
        qv -= 1

    BE = BlockEncoding.from_lcu(np.array([3, 1]), [U0, U1])

    # Apply inversion via QSVT
    BE_inv = inversion(BE, 0.01, np.linalg.cond(A), method="QSVT")

    # Prepare initial system state |b>
    def operand_prep():
        qv = QuantumFloat(2)
        prepare(qv, b)
        return qv

    @terminal_sampling
    def main():
        operand = BE_inv.apply_rus(operand_prep)()
        return operand

    res_dict = main()
    amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])

    # Compare to target values
    A_inv = np.linalg.inv(A)
    res = A_inv @ b / np.linalg.norm(A_inv @ b)
    assert np.linalg.norm(np.abs(res) - amps) < 1e-2
