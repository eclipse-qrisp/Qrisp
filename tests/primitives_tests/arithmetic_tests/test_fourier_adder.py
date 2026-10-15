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

"""Tests the static and Jasp-traceable Fourier adder implementations."""

import pytest

from qrisp import (
    QFT,
    QuantumFloat,
    conjugate,
    fourier_adder,
    inpl_adder_test,
    jasp_fourier_adder,
    measure,
)
from qrisp.alg_primitives.arithmetic.adders.fourier_adder import QuasiRZZ
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger
from qrisp.jasp import make_jaspr


def test_fourier_adder():

    inpl_adder_test(fourier_adder)


def test_fourier_adder_without_qft():
    """``perform_QFT=False`` assumes ``b`` is already in the Fourier basis."""

    a = QuantumFloat(3)
    b = QuantumFloat(3)
    a[:] = 5
    b[:] = 2

    with conjugate(QFT)(b, exec_swap=False):
        fourier_adder(a, b, perform_QFT=False)

    assert b.get_measurement() == {7: 1.0}


def test_fourier_adder_precision_error():
    """Adding a wider register onto a narrower one is rejected."""

    a = QuantumFloat(5)
    b = QuantumFloat(3)

    with pytest.raises(Exception, match="higher precision"):
        fourier_adder(a, b)


def test_quasi_rzz_inverse():
    """``QuasiRZZ.inverse`` negates the rotation angle."""

    inverse = QuasiRZZ(0.5).inverse()

    assert isinstance(inverse, QuasiRZZ)
    assert inverse.params[0] == -0.5


def test_jasp_fourier_adder():

    def call_fourier_adder():

        a = QuantumFloat(5)
        b = QuantumFloat(5)

        a[:] = 5
        b[:] = 4
        jasp_fourier_adder(a, b)

        return measure(b)

    jaspr = make_jaspr(call_fourier_adder)()
    assert jaspr() == 9


def test_jasp_fourier_adder_classical():
    """Classical addend (plain integer)."""

    def call_fourier_adder():

        b = QuantumFloat(5)
        b[:] = 4
        jasp_fourier_adder(5, b)
        return measure(b)

    assert make_jaspr(call_fourier_adder)()() == 9


def test_jasp_fourier_adder_bigint():
    """Classical addend of type ``BigInteger``."""

    def call_fourier_adder():

        b = QuantumFloat(5)
        b[:] = 4
        jasp_fourier_adder(BigInteger.create(5, 1), b)
        return measure(b)

    assert make_jaspr(call_fourier_adder)()() == 9


def test_jasp_fourier_adder_lists():
    """List-of-qubit inputs in eager (non-traced) mode."""

    a = QuantumFloat(4)
    b = QuantumFloat(4)
    a[:] = 3
    b[:] = 2

    jasp_fourier_adder(list(a), list(b))

    assert b.get_measurement() == {5: 1.0}
