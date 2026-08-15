"""********************************************************************************
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

import pytest
import numpy as np

from qrisp import multi_measurement, prepare, terminal_sampling, QuantumFloat
from qrisp.block_encodings import BlockEncoding
from qrisp.gqsp import GQET, QET, QSVT, GQSVT
from qrisp.operators import X, Y, Z


def post_selection(res_dict, N):
    filtered_dict = {k[0]: p for k, p in res_dict.items() if all(x == 0 for x in k[1:])}
    success_prob = sum(filtered_dict.values())
    filtered_dict = {k: p / success_prob for k, p in filtered_dict.items()}
    amps = np.sqrt([filtered_dict.get(k, 0) for k in range(N)])
    return amps


# Issue #681
@pytest.mark.parametrize("alg", [QET, GQET, QSVT, GQSVT])
@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_non_hermitian_block_encoding(alg, mode):
    """The test constructs a non-Hermitian block-encoding of a Hermitian matrix A and applies a polynomial transformation to it using the specified algorithm (QET, GQET, QSVT, and GQSVT).
    The resulting amplitudes are compared against the expected target amplitudes calculated using NumPy.
    Note: We use an odd fixed parity polynomial applied to a Hermitian matrix which ensures compatibility and consistency with all four transformations.
    Note: GQET relies on qubitization for a non-Hermitian block-encoding unitary (Issue #681).
    The test verifies that GQET produces the same results as the other three algorithms when applied to a non-Hermitian block-encoding of a Hermitian matrix.
    """

    N = 8
    I = np.eye(N)
    A = 2 * I + np.eye(N, k=1) + np.eye(N, k=-1)
    A[0, N - 1] = 1
    A[N - 1, 0] = 1
    # [[2. 1. 0. 0. 0. 0. 0. 1.]
    # [1. 2. 1. 0. 0. 0. 0. 0.]
    # [0. 1. 2. 1. 0. 0. 0. 0.]
    # [0. 0. 1. 2. 1. 0. 0. 0.]
    # [0. 0. 0. 1. 2. 1. 0. 0.]
    # [0. 0. 0. 0. 1. 2. 1. 0.]
    # [0. 0. 0. 0. 0. 1. 2. 1.]
    # [1. 0. 0. 0. 0. 0. 1. 2.]

    b = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target_amps = np.abs(A @ A @ A @ b / np.linalg.norm(A @ A @ A @ b))

    def operand_prep():
        operand = QuantumFloat(3)
        return operand

    # Define LCU block-encoding of the Hermitian matrix A resulting in a non-Hermitian block-encoding unitary (V != V^dagger)
    def I(qv):
        pass

    def V(qv):
        qv += 1

    def V_dg(qv):
        qv -= 1

    unitaries = [I, V, V_dg]
    coeffs = np.array([2.0, 1.0, 1.0])

    BE = BlockEncoding.from_lcu(coeffs, unitaries, is_hermitian=False)
    BE_poly = alg(BE, np.array([0.0, 0.0, 0.0, 1.0]), kind="Polynomial")

    # Calculate target amplitudes for comparison
    def main():
        operand = operand_prep()
        ancillas = BE_poly.apply(operand)
        return operand, *ancillas

    if mode == "static":
        operand, *ancillas = main()
        res_dict = multi_measurement([operand] + ancillas)
        amps = post_selection(res_dict, N)
        assert np.allclose(amps, target_amps, atol=1e-4)
    elif mode == "dynamic":
        amps_jasp = post_selection(terminal_sampling(main)(), N)
        assert np.allclose(amps_jasp, target_amps, atol=1e-4)


# Issue #680
@pytest.mark.parametrize("alg", [QET, GQET, QSVT, GQSVT])
@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_nested_polynomial_application(alg, mode):
    """The test constructs a Hermitian block-encoding of an Ising Hamiltonian and applies nested polynomial transformations to it using the specified algorithm (QET, GQET, QSVT, and GQSVT).
    The resulting amplitudes are compared against the expected target amplitudes calculated using NumPy.
    Note: We use an odd fixed parity polynomial applied to a Hermitian matrix which ensures compatibility and consistency with all four transformations.
    Note: GQET relies on qubitization for a non-Hermitian block-encoding unitary in the second application (Issue #681).
    """

    if alg == QET and mode == "static":
        pytest.skip("(Issue #680).")

    def create_ising_hamiltonian(L, J, B):
        H = sum(-J * Z(i) * Z(i + 1) for i in range(L - 1)) + sum(B * X(i) for i in range(L))
        return H

    L = 4
    H = create_ising_hamiltonian(L, 0.25, 0.5)
    alpha = np.sum(np.abs(H.coeffs()))
    H = (1.0 / alpha) * H

    H2 = 0.9 * H + 0.8 * H * H * H
    H3 = 0.9 * H2 + 0.8 * H2 * H2 * H2

    BE = BlockEncoding.from_operator(H)
    BE3 = BlockEncoding.from_operator(H3)

    def operand_prep():
        operand = QuantumFloat(L)
        prepare(operand, np.array([0.0, 0.9, 0.0, 0.8, 0.1, 0.4, 0.2, 0.3, 0.0, 0.9, 0.0, 0.8, 0.1, 0.4, 0.2, 0.3]))
        return operand

    # Calculate target amplitudes for comparison
    def main():
        operand = operand_prep()
        ancillas = BE3.apply(operand)
        return operand, *ancillas

    if mode == "static":
        operand, *ancillas = main()
        res_dict = multi_measurement([operand] + ancillas)
        target_amps = post_selection(res_dict, 2**L)
    elif mode == "dynamic":
        target_amps = post_selection(terminal_sampling(main)(), 2**L)

    BE_poly = alg(BE, np.array([0.0, 0.9, 0.0, 0.8]), kind="Polynomial")
    BE_poly_poly = alg(BE_poly, np.array([0.0, 0.9, 0.0, 0.8]), kind="Polynomial")

    # Nested polynomial application
    def main():
        operand = operand_prep()
        ancillas = BE_poly_poly.apply(operand)
        return operand, *ancillas

    if mode == "static":
        operand, *ancillas = main()
        res_dict = multi_measurement([operand] + ancillas)
        amps = post_selection(res_dict, 2**L)
        assert np.allclose(amps, target_amps, atol=1e-3)
    elif mode == "dynamic":
        amps_jasp = post_selection(terminal_sampling(main)(), 2**L)
        assert np.allclose(amps_jasp, target_amps, atol=1e-4)
