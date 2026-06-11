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

from qrisp import QuantumBool, QuantumFloat, control, h, prepare, swap, terminal_sampling
from qrisp.block_encodings import BlockEncoding


def create_matrix(coeffs, N):
    """Creates a banded matrix with specified coefficients on superdiagonal, subdiagonal, and their wrap-around elements."""

    A = np.zeros((N, N), dtype=complex)
    
    A += np.eye(N, k=1) * coeffs[0]  # Superdiagonal
    A += np.eye(N, k=-1) * coeffs[1]  # Subdiagonal
    A[0, N-1] = coeffs[1]  # Wrap-around for subdiagonal
    A[N-1, 0] = coeffs[0]  # Wrap-around for superdiagonal

    A += np.eye(N, k=2) * coeffs[2]  # Super-superdiagonal
    A += np.eye(N, k=-2) * coeffs[3]  # Sub-subdiagonal
    A[0, N-2] = coeffs[3]  # Wrap-around for sub-subdiagonal
    A[1, N-1] = coeffs[3]  # Wrap-around for sub-subdiagonal
    A[N-2, 0] = coeffs[2]  # Wrap-around for super-superdiagonal
    A[N-1, 1] = coeffs[2]  # Wrap-around for super-superdiagonal
    return A


def create_block_encoding(coeffs):
    """Creates a BlockEncoding of a banded matrix with specified coefficients on superdiagonal, subdiagonal,
    and their wrap-around elements using the LCU constructor."""

    def f0(qv): qv -= 1
    def f1(qv): qv += 1
    def f2(qv): qv -= 2
    def f3(qv): qv += 2
    return BlockEncoding.from_lcu(coeffs, [f0, f1, f2, f3])


@pytest.mark.parametrize("coeffs", [
    np.array([1.4, 1.1, 0.5, 0.3]),
    np.array([0.8, -0.6, 0.2, 0.1j]),
    np.array([1.0+0.5j, 0.9j, 0.4, 0.2]),
    np.array([-0.6j, 0.9j, 0.4, 0.7+0.2j]),
    np.array([0.3j, 0.9j, -0.4, 0.2]),
])
def test_block_encoding_from_lcu(coeffs):
    """Test the construction of a BlockEncoding from a linear combination of unitaries (LCU) with various types of coefficients."""

    A = create_matrix(coeffs, N=8)
    b = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    c = A @ b / np.linalg.norm(A @ b)

    BA = create_block_encoding(coeffs)

    def operand_prep():
        operand = QuantumFloat(3)
        prepare(operand, b)
        return operand
    
    # Use swap test to check if the block encoding correctly implements the desired operation on the operand state.
    @terminal_sampling
    def main():
        operand = BA.apply_rus(operand_prep)()

        psi = QuantumFloat(3)
        prepare(psi, c)

        qb = QuantumBool()
        h(qb)
        with control(qb):
            swap(operand, psi)
        h(qb)

        return qb

    res_dict = main()
    assert np.isclose(res_dict.get(False, 0), 1.0, atol=1e-6)


def test_block_encoding_from_lcu_single_unitary_raises_value_error():
    """Test that a ValueError is raised when a single unitary is provided with a complex coefficient."""
    coeffs = np.array([1.0 + 0.5j])
    def f0(qv): qv -= 1
    with pytest.raises(ValueError):
        BlockEncoding.from_lcu(coeffs, [f0])
