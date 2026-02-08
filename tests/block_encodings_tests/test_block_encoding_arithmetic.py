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
import pytest
from qrisp import *
from qrisp.block_encodings import BlockEncoding
from qrisp.operators import X, Y, Z


def compare_results(res_dict_1, res_dict_2, n):
    for k in range(2 ** n):
        val_1 = res_dict_1.get(k, 0)
        val_2 = res_dict_2.get(k, 0)
        assert np.isclose(val_1, val_2), f"Mismatch at state |{k}>: {val_1} vs {val_2}"


@pytest.mark.parametrize("H1, H2", [
    (X(0)*X(1) + 0.2*Y(0)*Y(1), Z(0)*Z(1) + X(2)),
    (0.5*X(1) + 0.7*Y(1) + 0.3*X(4), Z(0) + Z(1) + X(2)),
    (X(0)*X(1), Z(0) + 0.9*Z(1) + X(3)),
])
def test_block_encoding_addition(H1, H2):

    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)

    H3 = H1 + H2
    BE3 = H3.pauli_block_encoding()
    BE_addition = BE1 + BE2

    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_be3 = main(BE3)
    res_be_add = main(BE_addition)
    compare_results(res_be3, res_be_add, n)


@pytest.mark.parametrize("H1, H2", [
    (X(0)*X(1) + 0.2*Y(0)*Y(1), Z(0)*Z(1) + X(2)),
    (0.5*X(1) + 0.7*Y(1) + 0.3*X(4), Z(0) + Z(1) + X(2)),
    (X(0)*X(1), Z(0) + 0.9*Z(1) + X(3)),
])
def test_block_encoding_subtraction(H1, H2):

    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)

    H3 = H1 - H2
    BE3 = BlockEncoding.from_operator(H3)
    BE_subtraction = BE1 - BE2

    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()
    
    res_be3 = main(BE3)
    res_be_sub = main(BE_subtraction)
    compare_results(res_be3, res_be_sub, n)


# The product of two Hermitian operators A and B is Hermitian if and only if they commute, i.e., AB = BA.
# Thus, to ensure that the multiplication test is valid, we should choose pairs of operators that commute.
@pytest.mark.parametrize("H1, H2", [
    (X(0)*X(1) + 0.2*Y(0)*Y(1), Z(0)*Z(1) + X(2)),
    (0.5*X(1) + 0.7*Y(1) + 0.3*X(4), X(0) + X(4)),
    (X(0)*X(1), Z(0)*Z(1) + Y(3)),
])
def test_block_encoding_multiplication(H1, H2):

    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)

    H3 = H1 * H2
    BE3 = BlockEncoding.from_operator(H3)
    BE_multiplication = BE1 @ BE2

    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()
    
    res_be3 = main(BE3)
    res_be_mul = main(BE_multiplication)
    compare_results(res_be3, res_be_mul, n)


@pytest.mark.parametrize("H1, H2, scalar", [
    (X(0)*X(1) + 0.2*Y(0)*Y(1), Z(0)*Z(1) + X(2), -2),
    (0.5*X(1) + 0.7*Y(1), Z(0) + X(2), 0.5),
    (X(0), Z(0), 1),
])
def test_block_encoding_scalar_multiplication(H1, H2, scalar):
    H_target = scalar * H1 + H2
    BE_target = BlockEncoding.from_operator(H_target)

    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)

    BE_left = scalar * BE1 + BE2
    BE_right = BE1 * scalar + BE2
    
    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_target = main(BE_target)
    res_left = main(BE_left)
    res_right = main(BE_right)
    compare_results(res_target, res_left, n)
    compare_results(res_target, res_right, n)


@pytest.mark.parametrize("H1, H2", [
    (X(0)*X(1) - 0.2*Y(0)*Y(1), 0.2*Y(0)*Y(1) - X(0)*X(1)),
    (0.5*X(1) - 0.7*Y(1) + 0.3*X(4), 0.7*Y(1) - 0.5*X(1) - 0.3*X(4)),
    (Z(0)*Z(1) - Y(3), Y(3) - Z(0)*Z(1)),
])
def test_block_encoding_negation(H1, H2):

    BE1 = BlockEncoding.from_operator(H1)
    BE_neg = -BE1

    BE2 = BlockEncoding.from_operator(H2)

    n = H1.find_minimal_qubit_amount()

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()
    
    res_be2 = main(BE2)
    res_be_neg = main(BE_neg)
    compare_results(res_be2, res_be_neg, n)


@pytest.mark.parametrize("H1, H2", [
    (X(0)*X(1) + 0.2*Y(0)*Y(1), Z(0)*Z(1) + X(2)),
])
def test_block_encoding_kron(H1, H2):

    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)

    BE_kron = BE1.kron(BE2)

    n1 = H1.find_minimal_qubit_amount()
    n2 = H2.find_minimal_qubit_amount()

    def operand_prep():
        qv1 = QuantumFloat(n1)
        qv2 = QuantumFloat(n2)
        return qv1, qv2

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(operand_prep)()

    result_be_kron = main(BE_kron)

    @terminal_sampling
    def main(BE1, BE2):
        qv1 = BE1.apply_rus(lambda : QuantumFloat(n1))()
        qv2 = BE2.apply_rus(lambda : QuantumFloat(n2))()
        return qv1, qv2

    result_be1_be2 = main(BE1, BE2)

    for k in range(2 ** n1):
        for l in range(2 ** n2):
            val_be_kron = result_be_kron.get((k, l), 0)
            val_be1_be2 = result_be1_be2.get((k, l), 0)
            assert np.isclose(val_be_kron, val_be1_be2), f"Mismatch at state |{k}>: {val_be_kron} vs {val_be1_be2}"