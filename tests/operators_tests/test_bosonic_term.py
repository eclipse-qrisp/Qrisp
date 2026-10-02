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

from qrisp.operators.bosonic import a_b as a, c_b as c

import pytest

def test_arithmetic():
    O_0 = a(0) * c(1)
    O_1 = -1 * c(1) * a(0)

    assert not O_0 == O_1

    O_0 = a(0) * c(1) + a(1) * c(2)
    O_1 = a(0) * c(1)

    assert not O_0 == O_1

    O_0 = a(0) * c(1) - a(1) * c(2)
    O_1 = a(0) * c(1)

    assert not O_0 == O_1

    O_0 = a(0) * c(0) + a(1) * c(1)
    O_1 = a(2) * c(2) + a(3) * c(3)

    assert not O_0 == O_1

    O_0 = 1 * a(0) * a(0)
    O_1 = 2 * c(0) * c(0)

    assert not O_0 == O_1

    O_0 = c(0) * a(0)
    O_1 = -1 * c(0) * a(0)

    assert -O_1 == O_0

    O_0 = a(0) * c(1) * a(2)
    O_1 = c(2) * a(1) * c(0)

    assert O_0 == O_1

    O_0 = a(0)
    O_1 = 1e-12 * a(1)

    assert O_0 == O_0 + O_1
    assert O_0 == O_1 + O_0
    assert O_0 == O_0 - O_1
    assert -O_0 == O_1 - O_0
    assert 1. + O_0 == 1. - O_1 + O_0
    assert 1.e-12 - O_0 == 1.e-12 - O_0

    O_0 = a(0) + c(0)
    O_1 = a(0) * a(0) + c(0) * a(0) + a(0) * c(0) + c(0) * c(0)

    assert O_0**2 == O_1

    O = c(0) * a(0) + 1.
    O = c(0) * a(0) - 1.
    O = 1. - c(0) * a(0)
    O = 1. + c(0) * a(0)

    O = c(0) * a(0)

    O += c(1)
    assert O == c(0) * a(0) + c(1)
    O += 1.
    assert O == c(0) * a(0) + c(1) + 1.
    O += 1.e-12 * c(0)
    assert O == c(0) * a(0) + c(1) + 1.

    O -= 1.

def test_hermitize():
    O_0 = a(0) * c(1)
    O_1 = c(1) * a(0)

    assert O_0.hermitize() == O_1.hermitize()

def test_reduce():
    O = 3 * a(0) * c(1) + c(1) * a(0)
    O = O.reduce()

    assert str(O) == "4*a0*c1"

    O = a(0) * a(1) - c(1) * c(0)
    O = O.reduce(assume_hermitian=True)

    assert str(O) == "0"

def test_error_combining_operator():
    with pytest.raises(TypeError, match="Cannot add BosonicOperator"):
        O = c(0) + [1,2]
    with pytest.raises(TypeError, match="Cannot subtract BosonicOperator"):
        O = c(0) - [1,2]
    with pytest.raises(TypeError, match="Cannot subtract BosonicOperator"):
        O = [1,2] -  c(0)
    with pytest.raises(TypeError, match="Cannot multipliy BosonicOperator"):
        O = c(0) * [1,2]
    with pytest.raises(TypeError, match="Operators can be exponentiated only with positive integers"):
        O = c(0) ** (-1)
    with pytest.raises(TypeError, match="Cannot add BosonicOperator"):
        O = c(0)
        O += [1,2]
    with pytest.raises(TypeError, match="Cannot subtract BosonicOperator"):
        O = c(0)
        O -= [1,2]

def test_len():
    O = c(0) * a(0) + c(1) * a(1)

    assert len(O) == 2

def test_coeffs():
    O = 5 * c(0) * a(0) + 7 * c(1) * a(1)

    assert all(O.coeffs() == [5, 7])

def test_latex():
    assert (c(0)*a(0))._repr_latex_() == "$c_{0} a_{0}$"

def test_hash():
    O = c(0) * a(0)

    h = hash(O)
