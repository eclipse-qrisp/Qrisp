"""
********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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

from qrisp.operators import X, Y, Z, A, C, P0, P1
from numpy.linalg import norm
from numpy import isclose


def test_to_pauli_converter():

    operator_list = [lambda x: 1, X, Y, Z, A, C, P0, P1]

    for O0 in operator_list:
        for O1 in operator_list:
            for O2 in operator_list:
                H = O0(0) * O1(1) * O2(2)
                if isinstance(H, int):
                    continue

                assert norm(H.to_array() - H.to_pauli().to_array()) < 1e-5


def test_pauli_coeff_dict_extraction():
    H0 = X(0) + 0.5 * Y(1) + 0.2 * Z(0) * Z(2)
    H1 = X(0) * Z(1)
    H2 = 1 + 2 * X(0) + 3 * X(0) * Y(1) * A(2) + C(4) * P1(0)

    res0 = H0._to_pauli_dict()
    res1 = H1._to_pauli_dict()
    res2 = H2._to_pauli_dict()

    expected0 = {
        ((0, "X"),): 1,
        ((1, "Y"),): 0.5,
        ((0, "Z"), (2, "Z")): 0.2,
    }

    expected1 = {
        ((0, "X"), (1, "Z")): 1,
    }

    # A(2) = (X(2) + iY(2)) / 2
    # C(4) = (X(4) - iY(4)) / 2
    # P1(0) = (I - Z(0)) / 2
    expected2 = {
        (): 1,
        ((0, "X"),): 2,
        # 3 * X(0) * Y(1) * A(2)
        ((0, "X"), (1, "Y"), (2, "X")): 1.5,
        ((0, "X"), (1, "Y"), (2, "Y")): 1.5j,
        # C(4) * P1(0)
        ((4, "X"),): 0.25,
        ((4, "Y"),): -0.25j,
        ((0, "Z"), (4, "X")): -0.25,
        ((0, "Z"), (4, "Y")): 0.25j,
    }

    def assert_dict_allclose(res, expected):
        assert set(res.keys()) == set(expected.keys())

        for key in expected:
            assert isclose(res[key], expected[key]), f"For key {key}: got {res[key]}, expected {expected[key]}"

    assert_dict_allclose(res0, expected0)
    assert_dict_allclose(res1, expected1)
    assert_dict_allclose(res2, expected2)
