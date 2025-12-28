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
from qrisp.gqsp import GQSP
from qrisp.operators import X,Y,Z
from scipy.linalg import expm


def expvalm(poly, k, A):
    res = np.zeros(A.shape, dtype=complex)
    for ind, coeff in enumerate(poly):
        res += coeff * expm(1.j * (ind - k) * A)
    return res


@pytest.mark.parametrize("poly, k", [
    (np.array([0.5, 0., 0.5]), 1), # cos
    (np.array([1., 1.]), 0),
])
def test_qsp(poly, k):

    # All terms in Hamiltonian commute -> e^{iH} is implemented exactly by trotterization
    H = Z(0)*Z(1) + X(0)*X(1)

    # e^{iH}
    def U(operand):
        H.trotterization(forward_evolution=False)(operand)

    def operand_prep():
        operand = QuantumVariable(2)
        return operand
    
    @RUS
    def inner():
        operand = operand_prep()
        qbl = GQSP(operand, U, poly, k=k)
        success_bool = measure(qbl) == 0
        return success_bool, operand

    @terminal_sampling
    def main(): 
        qv = inner()
        return qv
    
    res_dict = main()
    res = np.array([res_dict.get(key, 0) for key in range(4)]) # Measurement probabilities

    # Compare to classical values
    H_arr = H.to_array()
    res_numpy = expvalm(poly, k, H_arr) @ np.array([1,0,0,0])
    res_numpy = np.abs(res_numpy / np.linalg.norm(res_numpy)) ** 2
    assert np.linalg.norm(res - res_numpy) < 1e-2

