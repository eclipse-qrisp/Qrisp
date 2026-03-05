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
from qrisp.operators import X, Y, Z


def compare_results(res_dict_1, res_dict_2, n):
    for k in range(2 ** n):
        val_1 = res_dict_1.get(k, 0)
        val_2 = res_dict_2.get(k, 0)
        assert np.isclose(val_1, val_2), f"Mismatch at state |{k}>: {val_1} vs {val_2}"


@pytest.mark.parametrize("H1, H2, rescaled", [
    (X(0)*X(1) + 0.2*Y(0)*Y(1), Z(0)*Z(1) + X(2), True),
    (0.5*X(1) + 0.7*Y(1) + 0.3*X(4), Z(0) + Z(1) + X(2), False),
])
def test_block_encoding_chebyshev(H1, H2, rescaled):

    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)

    alpha = BE2.alpha

    if rescaled:
        H_T1 = H1 + H2 # H1 + T_1(H2)
        H_T2 = H1 + (2 * H2**2 - 1) # H1 + T_2(H2)
        H_T3 = H1 + (4 * H2**3 - 3 * H2) # H1 + T_3(H2)
    else:
        H_T1 = H1 + H2 # H1 + T_1(H2 / alpha)
        H_T2 = H1 + (2 / alpha**2 * H2**2 - 1) # H1 + T_2(H2 / alpha)
        H_T3 = H1 + (4 / alpha**3 * H2**3 - 3 / alpha * H2) # H1 + T_3(H2 / alpha)   
    
    BE_T1 = BlockEncoding.from_operator(H_T1)
    BE_T2 = BlockEncoding.from_operator(H_T2)
    BE_T3 = BlockEncoding.from_operator(H_T3)

    BE_cheb_T1 = BE2.chebyshev(1, rescale=rescaled)
    BE_cheb_T2 = BE2.chebyshev(2, rescale=rescaled)
    BE_cheb_T3 = BE2.chebyshev(3, rescale=rescaled)

    BE_add_T1 = BE1 + BE_cheb_T1
    BE_add_T2 = BE1 + BE_cheb_T2
    BE_add_T3 = BE1 + BE_cheb_T3

    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_be_t1 = main(BE_T1)
    res_be_add_t1 = main(BE_add_T1)
    compare_results(res_be_t1, res_be_add_t1, n)

    res_be_t2 = main(BE_T2)
    res_be_add_t2 = main(BE_add_T2)
    compare_results(res_be_t2, res_be_add_t2, n)

    res_be_t3 = main(BE_T3)
    res_be_add_t3 = main(BE_add_T3)
    compare_results(res_be_t3, res_be_add_t3, n)