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
from qrisp.block_encodings.commutators import nested_commutators, unary_prep, unary_walk_prep, _get_chebyshev_commutator_coeffs
from qrisp.operators import X, Y, Z


"""Tests for the state preparation subroutines used in the nested commutator block encoding construction."""

def apply_unary_prep(d, coeffs):
    n = int(np.ceil(np.log2(d + 1)))
    anc = QuantumVariable(2 * n)
    qm = QuantumVariable(d + 1)
    qn = QuantumVariable(d + 1)

    unary_prep(anc, qm, qn, d, coeffs=coeffs)
    return qm, qn


def apply_unary_walk_prep(d, coeffs):
    steps = QuantumVariable(d)
    coin1 = QuantumVariable(d)
    coin2 = QuantumVariable(d)
    m_line = QuantumVariable(2 * d + 1)
    n_line = QuantumVariable(2 * d + 1)
    qm = QuantumVariable(d + 1)
    qn = QuantumVariable(d + 1)

    unary_walk_prep(steps, coin1, coin2, m_line, n_line, qm, qn, d, coeffs=coeffs)
    return qm, qn


def coeff_matrix(coeffs):
    """
    Constructs the coefficient matrix for the weighted sum of nested commutators given the coefficients for each order of the commutator.
    """
    d = len(coeffs)
    C = np.zeros((d + 1, d + 1))
    for k in range(1, d + 1):
        Ck_matrix = _get_chebyshev_commutator_coeffs(k)
        rows, cols = Ck_matrix.shape
        C[:rows, :cols] += coeffs[k - 1] * Ck_matrix
    return C


def measurement_to_coeff_matrix(res_dict, d):
    """
    Converts the measurement results from the state preparation into a coefficient matrix for the weighted sum of nested commutators.
    """
    C = np.zeros((d + 1, d + 1))
    for key, value in res_dict.items():
        i = key[0].count('1') - 1
        j = key[1].count('1') - 1
        C[i, j] = value
    return C


@pytest.mark.parametrize("coeffs", [
    np.array([0, 0, 0, 1]),
    np.array([0.1, 0.2, 0.3, 0.4]),
    np.array([0.4, 0.3, 0.2, 0.1]),
    np.array([0.25, 0.25, 0.25, 0.25]),
])
def test_state_prep_for_nested_commutators(coeffs):

    d = len(coeffs)
    C = np.abs(coeff_matrix(coeffs))
    C = C / np.sum(C)

    qm1, qn1 = apply_unary_prep(d, coeffs)
    res_dict = multi_measurement([qm1, qn1])
    C_meas = measurement_to_coeff_matrix(res_dict, d)
    assert np.allclose(C, C_meas, atol=1e-2)

    qm2, qn2 = apply_unary_walk_prep(d, coeffs)
    res_dict = multi_measurement([qm2, qn2])
    C_meas = measurement_to_coeff_matrix(res_dict, d)
    assert np.allclose(C, C_meas, atol=1e-2)


"""Tests for the nested commutator block encoding construction."""

@pytest.mark.parametrize("A, B, coeffs", [
    (0.5*X(0)*Z(1) + 0.5*Y(0)*Y(1), 0.5*Z(0)*Z(1) + 0.5*X(0)*Y(1), np.array([1., 0., 1.])),
    (0.5*X(0)*Z(1) + 0.5*Y(0)*Y(1), 0.5*Z(0)*Z(1) + 0.5*X(0)*Y(1), np.array([0., 0., 1.])),
])
def test_nested_commutators(A, B, coeffs):

    ad1 = (A*B - B*A)
    ad2 = A*ad1 - ad1*A
    ad3 = A*ad2 - ad2*A

    # Weighted sum of odd nested commutators
    ad_sum = sum(c * ad for c, ad in zip(coeffs, [ad1, ad2, ad3]))
    B_ad_sum = BlockEncoding.from_operator(1.j * ad_sum)

    # BlockEncodings for A and B
    B_A = BlockEncoding.from_operator(A)
    B_B = BlockEncoding.from_operator(B)

    # BlockEncoding of sum of odd nested commutators
    B_C = B_A.nested_commutators(B_B, coeffs, method="default")
    B_C_walk = B_A.nested_commutators(B_B, coeffs, method="walk")

    b = np.array([1., 1., 0., 1.])
    # Prepare variable in state |b>
    def prep_b():
        qv = QuantumFloat(2)
        prepare(qv, b)
        return qv
    
    @terminal_sampling
    def main(BE):
        return BE.apply_rus(prep_b)()

    res_dict_ad_sum = main(B_ad_sum)
    amps_ad_sum = np.sqrt([res_dict_ad_sum.get(i, 0) for i in range(len(b))])

    res_dict_C = main(B_C)
    amps_C = np.sqrt([res_dict_C.get(i, 0) for i in range(len(b))])

    res_dict_C_walk = main(B_C_walk)
    amps_C_walk = np.sqrt([res_dict_C_walk.get(i, 0) for i in range(len(b))])

    assert np.allclose(amps_ad_sum, amps_C, atol=1e-2)
    assert np.allclose(amps_ad_sum, amps_C_walk, atol=1e-2)
