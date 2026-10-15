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

"""Tests GQSP polynomial transformations applied to a small unitary."""

import numpy as np
import pytest
from qrisp.gqsp import GQSP
from scipy.linalg import expm

from qrisp import *
from qrisp.operators import X, Z


def expvalm(poly, k, A):
    res = np.zeros(A.shape, dtype=complex)
    for ind, coeff in enumerate(poly):
        res += coeff * expm(1.0j * (ind - k) * A)
    return res


@pytest.mark.parametrize(
    "poly, k",
    [
        (np.array([0.5, 0.0, 0.5]), 1),  # cos
        (np.array([1.0, 1.0]), 0),
    ],
)
def test_gqsp(poly, k):
    """Test GQSP on a small 4x4 unitary with a simple polynomial transformation."""
    # All terms in Hamiltonian commute -> e^{iH} is implemented exactly by trotterization
    H = Z(0) * Z(1) + X(0) * X(1)

    # e^{iH}
    def U(operand):
        H.trotterization(forward_evolution=False)(operand)

    def operand_prep():
        operand = QuantumVariable(2)
        return operand

    @RUS
    def inner():
        operand = operand_prep()
        qbl = QuantumBool()
        GQSP(qbl, operand, unitary=U, p=poly, k=k)
        success_bool = measure(qbl) == 0
        reset(qbl)
        qbl.delete()
        return success_bool, operand

    @terminal_sampling
    def main():
        qv = inner()
        return qv

    res_dict = main()
    res = np.array([res_dict.get(key, 0) for key in range(4)])  # Measurement probabilities

    # Compare to target values
    H_arr = H.to_array()
    res_numpy = expvalm(poly, k, H_arr) @ np.array([1, 0, 0, 0])
    res_numpy = np.abs(res_numpy / np.linalg.norm(res_numpy)) ** 2
    assert np.linalg.norm(res - res_numpy) < 1e-2


def test_qpsp_complex_coeffs():
    """Test GQSP with complex coefficients."""

    # p(z) = (1 + i z)/2
    coeffs = np.array([0.5, 0.5j])

    # diag(1, i)
    def U(qv):
        p(np.pi / 2, qv[0])

    qv = QuantumFloat(1)
    h(qv)
    anc = QuantumBool()
    GQSP(anc, qv, unitary=U, p=np.array(coeffs))
    res = multi_measurement([qv, anc])
    post_selected_res = {k[0]: v for k, v in res.items() if k[1] is False}

    # I + i U = I + i * diag(1, i) = diag(1 + i, 1 - 1) = diag(1 + i, 0)
    assert post_selected_res.get(1, 0) == 0
