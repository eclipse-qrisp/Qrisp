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

from qrisp import QuantumVariable, x
from qrisp.operators.bosonic import a_b as a, c_b as c
from qrisp.operators.bosonic import prepare_bosonic_fock_state
from numpy import isclose


def test_one_hot_expval():
    N = c(0) * a(0)
    for n in range(5):
        assert isclose(
            N.expectation_value(prepare_bosonic_fock_state, truncation=5, binary_encoding="one_hot")(n, 5, "one_hot"), n
        )
        assert isclose(
            a(0).expectation_value(prepare_bosonic_fock_state, truncation=5, binary_encoding="one_hot")(
                n, 5, "one_hot"
            ),
            0,
            atol=0.1,
        )


def test_gray_code_expval():
    N = c(0) * a(0)
    for n in range(5):
        assert isclose(
            N.expectation_value(prepare_bosonic_fock_state, truncation=8, binary_encoding="gray_code")(
                n, 8, "gray_code"
            ),
            n,
        )
        assert isclose(
            a(0).expectation_value(prepare_bosonic_fock_state, truncation=8, binary_encoding="gray_code")(
                n, 8, "gray_code"
            ),
            0,
            atol=0.1,
        )


def test_standard_binary_expval():
    N = c(0) * a(0)
    for n in range(5):
        assert isclose(
            N.expectation_value(prepare_bosonic_fock_state, truncation=8, binary_encoding="standard_binary")(
                n, 8, "standard_binary"
            ),
            n,
        )
        assert isclose(
            a(0).expectation_value(prepare_bosonic_fock_state, truncation=8, binary_encoding="standard_binary")(
                n, 8, "standard_binary"
            ),
            0,
            atol=0.1,
        )
