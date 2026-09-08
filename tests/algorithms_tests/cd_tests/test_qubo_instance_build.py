# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

import numpy as np

from qrisp.algorithms.cold.problems.QUBO import create_COLD_instance, create_LCD_instance
from qrisp.algorithms.cold.problems.qubo_problems import Q6
from qrisp.operators.qubit import Y, Z
from qrisp.operators.qubit.qubit_operator import QubitOperator


def test_qubit_operator_sum_matches_naive_sum():
    """QubitOperator.sum matches plain sum(), including pruning of exact-zero coefficients."""
    # Half the coefficients are exactly zero, matching a sparse QUBO's coupling matrix.
    coeffs = [0.3, 0.0, -0.7, 0.0, 1.1, 0.0, -0.2, 0.5]
    N = len(coeffs)

    naive = sum(coeffs[i] * Z(i) for i in range(N))
    fast = QubitOperator.sum(coeffs[i] * Z(i) for i in range(N))

    assert fast.terms_dict.keys() == naive.terms_dict.keys()
    for term, coeff in naive.terms_dict.items():
        assert abs(fast.terms_dict[term] - coeff) < 1e-9


def test_create_cold_instance_H_prob_matches_naive_build():
    """create_COLD_instance's H_prob matches a naive-sum() reference build."""
    Q = Q6
    N = Q.shape[0]
    h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
    J = 0.5 * Q

    H_prob_naive = sum([sum([J[i][j] * Z(i) * Z(j) for j in range(i, N)]) for i in range(N)]) + sum(
        [h[i] * Z(i) for i in range(N)]
    )

    _, _, H_prob, _, _, _, _ = create_COLD_instance(Q, uniform_AGP_coeffs=False)

    assert H_prob.terms_dict.keys() == H_prob_naive.terms_dict.keys()
    for term, coeff in H_prob_naive.terms_dict.items():
        assert abs(H_prob.terms_dict[term] - coeff) < 1e-9


def test_create_lcd_instance_H_prob_and_nc_agp_match_naive_build():
    """create_LCD_instance(agp_type="nc")'s H_prob and nested-commutator A_lam match a naive-sum() reference build."""
    Q = Q6
    N = Q.shape[0]
    h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
    J = 0.5 * Q

    H_prob_naive = sum([sum([J[i][j] * Z(i) * Z(j) for j in range(i, N)]) for i in range(N)]) + sum(
        [h[i] * Z(i) for i in range(N)]
    )
    A_lam_naive = [
        -2 * (h[i] * Y(i) + sum([J[i][j] * (Y(i) * Z(j) + Z(i) * Y(j)) for j in range(i)])) for i in range(N)
    ]

    _, _, H_prob, A_lam, _, _ = create_LCD_instance(Q, agp_type="nc", uniform_AGP_coeffs=False)

    assert H_prob.terms_dict.keys() == H_prob_naive.terms_dict.keys()
    for term, coeff in H_prob_naive.terms_dict.items():
        assert abs(H_prob.terms_dict[term] - coeff) < 1e-9

    assert len(A_lam) == len(A_lam_naive)
    for op, op_naive in zip(A_lam, A_lam_naive):
        assert op.terms_dict.keys() == op_naive.terms_dict.keys()
        for term, coeff in op_naive.terms_dict.items():
            assert abs(op.terms_dict[term] - coeff) < 1e-9
