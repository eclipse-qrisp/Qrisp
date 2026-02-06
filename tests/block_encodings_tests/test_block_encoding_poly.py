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
from qrisp.operators import X, Y, Z


# This test confirms that BlockEncodings via GQET (used for BlockEncoding.poly) have the correct scaling factor alpha.
@pytest.mark.parametrize("H1, H2, poly", [
    (X(0)*X(1) + 0.2*Y(0)*Y(1), Z(0)*Z(1) + X(2), np.array([1,1,1])),
    (0.5*X(1) + 0.7*Y(1) + 0.3*X(4), Z(0) + Z(1) + X(2), np.array([0,1,0,1])),
    (X(0)*X(1), Z(0) + 0.9*Z(1) + X(3), np.array([1,1,0,1])),
])
def test_block_encoding_poly_scaling(H1, H2, poly):

    BE1 = H1.pauli_block_encoding()
    BE2 = H2.pauli_block_encoding()

    # Apply polynomial to QubitOperator H1 and add H2
    H3 = sum(poly[k] * H1 ** k for k in range(len(poly))) + H2
    BE3 = H3.pauli_block_encoding()

    # Apply polynomial to BlockEncoding BE1 and add BE2
    BE_poly = BE1.poly(poly) + BE2

    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_be3 = main(BE3)
    res_be_poly = main(BE_poly)

    for k in range(2 ** n):
        val_be3 = res_be3.get(k, 0)
        val_be_poly = res_be_poly.get(k, 0)
        assert np.isclose(val_be3, val_be_poly), f"Mismatch at state |{k}>: {val_be3} vs {val_be_poly}"