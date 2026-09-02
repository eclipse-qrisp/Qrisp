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


def test_jasp_QuantumFloat():
    """Test QuantumFloat's decoder under Jasp tracing, and that signed/exponent are static/dynamic respectively."""
    # Test decoder for QuantumFloat (Issue #271)
    from qrisp import QuantumFloat, h
    from qrisp.jasp import make_jaspr, qache, terminal_sampling

    @terminal_sampling
    def main():
        a = QuantumFloat(3, -2, signed=True)

        h(a)

        return a

    res = main()
    assert res == {
        -2.0: 0.0625,
        -1.75: 0.0625,
        -1.5: 0.0625,
        -1.25: 0.0625,
        -1.0: 0.0625,
        -0.75: 0.0625,
        -0.5: 0.0625,
        -0.25: 0.0625,
        0.0: 0.0625,
        0.25: 0.0625,
        0.5: 0.0625,
        0.75: 0.0625,
        1.0: 0.0625,
        1.25: 0.0625,
        1.5: 0.0625,
        1.75: 0.0625,
    }

    # Test that the signed attribute behaves statically and the exponent
    # attribute dynamically.

    @qache
    def inner(qf):
        assert isinstance(qf.signed, bool)
        assert not isinstance(qf.exponent, int)
        return qf.signed

    @make_jaspr
    def check_signed_static_exponent_dynamic():
        a = QuantumFloat(3, -1, signed=True)
        inner(a)
        inner(a)
        b = QuantumFloat(3, -1, signed=False)
        inner(b)
        inner(b)

        assert a.signed
        assert not b.signed


def test_jasp_QuantumFloat_comparisons():
    """Test QuantumFloat's comparison operators under Jasp tracing."""
    from qrisp import QuantumFloat
    from qrisp.jasp import terminal_sampling

    @terminal_sampling
    def compare(a_value, b_value, op):
        a = QuantumFloat(4)
        b = QuantumFloat(4)
        a[:] = a_value
        b[:] = b_value
        if op == "lt":
            return a < b
        if op == "gt":
            return a > b
        if op == "le":
            return a <= b
        if op == "ge":
            return a >= b
        if op == "eq":
            return a == b
        return a != b

    assert compare(2, 5, "lt") == {True: 1.0}
    assert compare(5, 2, "lt") == {False: 1.0}
    assert compare(5, 2, "gt") == {True: 1.0}
    assert compare(3, 3, "le") == {True: 1.0}
    assert compare(3, 3, "ge") == {True: 1.0}
    assert compare(4, 4, "eq") == {True: 1.0}
    assert compare(4, 5, "eq") == {False: 1.0}
    assert compare(4, 5, "ne") == {True: 1.0}


def test_jasp_QuantumFloat_arithmetic():
    """Test QuantumFloat's arithmetic operators under Jasp tracing."""
    from qrisp import QuantumFloat
    from qrisp.jasp import terminal_sampling

    @terminal_sampling
    def add():
        a = QuantumFloat(4)
        b = QuantumFloat(4)
        a[:] = 3
        b[:] = 2
        return a + b

    @terminal_sampling
    def sub():
        a = QuantumFloat(4, signed=True)
        b = QuantumFloat(4)
        a[:] = 5
        b[:] = 2
        return a - b

    @terminal_sampling
    def mul():
        a = QuantumFloat(4)
        b = QuantumFloat(4)
        a[:] = 3
        b[:] = 2
        return a * b

    @terminal_sampling
    def iadd():
        a = QuantumFloat(4)
        b = QuantumFloat(4)
        a[:] = 3
        b[:] = 2
        a += b
        return a

    @terminal_sampling
    def isub():
        a = QuantumFloat(4, signed=True)
        b = QuantumFloat(4)
        a[:] = 5
        b[:] = 2
        a -= b
        return a

    assert add() == {5.0: 1.0}
    assert sub() == {3.0: 1.0}
    assert mul() == {6.0: 1.0}
    assert iadd() == {5.0: 1.0}
    assert isub() == {3.0: 1.0}
