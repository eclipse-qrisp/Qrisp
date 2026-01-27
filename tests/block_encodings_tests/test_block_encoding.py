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
import jax.numpy as jnp
import pytest
from qrisp import *
from qrisp.operators import X, Y, Z


@pytest.mark.parametrize("H1, H2", [
    (X(0)*X(1) + 0.2*Y(0)*Y(1), Z(0)*Z(1) + X(2)),
    (0.5*X(1) + 0.7*Y(1) + 0.3*X(4), Z(0) + Z(1) + X(2)),
    (X(0)*X(1), Z(0) + 0.9*Z(1) + X(3)),
])
def test_block_encoding_addition(H1, H2):

    BE1 = H1.pauli_block_encoding()
    BE2 = H2.pauli_block_encoding()

    H3 = H1 + H2
    BE3 = H3.pauli_block_encoding()
    BE_addition = BE1 + BE2

    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_be3 = main(BE3)
    res_be_add = main(BE_addition)

    for k in range(2 ** n):
        val_be3 = res_be3.get(k, 0)
        val_be_add = res_be_add.get(k, 0)
        assert np.isclose(val_be3, val_be_add), f"Mismatch at state |{k}>: {val_be3} vs {val_be_add}"


@pytest.mark.parametrize("H1, H2", [
    (X(0)*X(1) + 0.2*Y(0)*Y(1), Z(0)*Z(1) + X(2)),
    (0.5*X(1) + 0.7*Y(1) + 0.3*X(4), Z(0) + Z(1) + X(2)),
    (X(0)*X(1), Z(0) + 0.9*Z(1) + X(3)),
])
def test_block_encoding_subtraction(H1, H2):

    BE1 = H1.pauli_block_encoding()
    BE2 = H2.pauli_block_encoding()

    H3 = H1 - H2
    BE3 = H3.pauli_block_encoding()
    BE_subtraction = BE1 - BE2

    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()
    
    res_be3 = main(BE3)
    res_be_sub = main(BE_subtraction)

    for k in range(2 ** n):
        val_be3 = res_be3.get(k, 0)
        val_be_sub = res_be_sub.get(k, 0)
        assert np.isclose(val_be3, val_be_sub), f"Mismatch at state |{k}>: {val_be3} vs {val_be_sub}"


# The product of two Hermitian operators A and B is Hermitian if and only if they commute, i.e., AB = BA.
# Thus, to ensure that the multiplication test is valid, we should choose pairs of operators that commute.
@pytest.mark.parametrize("H1, H2", [
    (X(0)*X(1) + 0.2*Y(0)*Y(1), Z(0)*Z(1) + X(2)),
    (0.5*X(1) + 0.7*Y(1) + 0.3*X(4), X(0) + X(4)),
    (X(0)*X(1), Z(0)*Z(1) + Y(3)),
])
def test_block_encoding_multiplication(H1, H2):

    BE1 = H1.pauli_block_encoding()
    BE2 = H2.pauli_block_encoding()

    H3 = H1 * H2
    BE3 = H3.pauli_block_encoding()
    BE_multiplication = BE1 * BE2

    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()
    
    res_be3 = main(BE3)
    res_be_mul = main(BE_multiplication)

    for k in range(2 ** n):
        val_be3 = res_be3.get(k, 0)
        val_be_mul = res_be_mul.get(k, 0)
        assert np.isclose(val_be3, val_be_mul), f"Mismatch at state |{k}>: {val_be3} vs {val_be_mul}"


@pytest.mark.parametrize("H1, H2, scalar", [
    (X(0)*X(1) + 0.2*Y(0)*Y(1), Z(0)*Z(1) + X(2), -2),
    (0.5*X(1) + 0.7*Y(1), Z(0) + X(2), 0.5),
    (X(0), Z(0), 1),
])
def test_block_encoding_scalar_multiplication(H1, H2, scalar):
    H_target = scalar * H1 + H2
    BE_target = H_target.pauli_block_encoding()

    BE1 = H1.pauli_block_encoding()
    BE2 = H2.pauli_block_encoding()

    BE_left = scalar * BE1 + BE2
    BE_right = BE1 * scalar + BE2
    
    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_target = main(BE_target)
    res_left = main(BE_left)
    res_right = main(BE_right)

    for k in range(2 ** n):
        val_target = res_target.get(k, 0)
        val_left = res_left.get(k, 0)
        val_right = res_right.get(k, 0)
        
        assert np.isclose(val_target, val_left), f"Left-mul mismatch at |{k}>"
        assert np.isclose(val_target, val_right), f"Right-mul mismatch at |{k}>"


@pytest.mark.parametrize("H1, H2", [
    (X(0)*X(1) - 0.2*Y(0)*Y(1), 0.2*Y(0)*Y(1) - X(0)*X(1)),
    (0.5*X(1) - 0.7*Y(1) + 0.3*X(4), 0.7*Y(1) - 0.5*X(1) - 0.3*X(4)),
    (Z(0)*Z(1) - Y(3), Y(3) - Z(0)*Z(1)),
])
def test_block_encoding_negation(H1, H2):

    BE1 = H1.pauli_block_encoding()
    BE_neg = -BE1

    BE2 = H2.pauli_block_encoding()

    n = H1.find_minimal_qubit_amount()

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()
    
    res_be_neg = main(BE_neg)
    res_be2 = main(BE2)

    for k in range(2 ** n):
        val_be_neg = res_be_neg.get(k, 0)
        val_be2 = res_be2.get(k, 0)
        assert np.isclose(val_be_neg, val_be2), f"Mismatch at state |{k}>: {val_be_neg} vs {val_be2}"