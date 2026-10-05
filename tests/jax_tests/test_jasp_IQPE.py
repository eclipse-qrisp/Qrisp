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

"""Tests IQPE (iterative quantum phase estimation) under Jasp tracing."""

import numpy as np
import pytest

from qrisp import IQPE, QuantumVariable, h, p, rx, x
from qrisp.jasp import jaspify, make_jaspr


def test_IQPE_integration():
    """IQPE estimates the phase of two rotations on their eigenstates.

    Each qubit is in the state |->, on which rx(2 pi x) has the eigenvalue
    exp(i pi x), so the phase is (x + y) / 2 for the angles 2 pi x and 2 pi y.
    """
    x_angle, y_angle = 1 / 2**3, 1 / 2**2

    def f():
        def U(qv):
            rx(x_angle * 2 * np.pi, qv[0])
            rx(y_angle * 2 * np.pi, qv[1])

        qv = QuantumVariable(2)

        x(qv)
        h(qv)

        return IQPE(qv, U, precision=4)

    jaspr = make_jaspr(f)()
    assert jaspr() == (x_angle + y_angle) / 2


@pytest.mark.parametrize("iter_spec", [False, True], ids=["repeated U", "iter_spec"])
def test_IQPE_exact_phases(iter_spec):
    """Every phase y / 2**t with t bits is estimated exactly.

    Regression test: the powers of U were U**(2**t), ..., U**2 instead of
    U**(2**(t - 1)), ..., U, so IQPE returned twice the phase modulo 1 and lost
    its most significant bit, for example 0.5 instead of 0.75.
    """
    precision = 3

    @jaspify
    def estimate(phase):
        def U(qv, **kwargs):
            # With iter_spec, IQPE passes the number of iterations as the keyword iter
            p(2 * np.pi * phase * kwargs.get("iter", 1), qv[0])

        qv = QuantumVariable(1)
        x(qv)
        return IQPE(qv, U, precision=precision, iter_spec=iter_spec)

    for y in range(2**precision):
        assert estimate(y / 2**precision) == y / 2**precision, y
