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

from qrisp import *


def test_injection_and():
    """Inject AND gate onto a pre-flipped target."""
    def AND(a, b):
        res = QuantumBool()
        mcx([a, b], res)
        return res

    a = QuantumBool()
    b = QuantumBool()
    target = QuantumBool()

    target.flip()
    (target << AND)(a, b)

    res = target.get_measurement()
    assert res == {1: 1.0}, f"Expected {{1: 1.0}}, got {res}"


def test_injection_uncomputation():
    """Manual uncomputation using injection + invert."""
    a = QuantumFloat(4)
    b = QuantumFloat(4)

    a[:] = 3
    b[:] = 5

    c = a * b
    res_before = c.get_measurement()
    assert res_before == {15: 1.0}, f"Expected {{15: 1.0}}, got {res_before}"

    with invert():
        (c << (lambda x, y: x * y))(a, b)

    res_after = c.get_measurement()
    assert res_after == {0: 1.0}, f"Expected {{0: 1.0}} after uncomputation, got {res_after}"

    c.delete()


def test_injection_state_prep():
    """Inject a state-prep function onto an existing variable."""
    def init_state():
        qv = QuantumFloat(4)
        x(qv[0])
        x(qv[1])
        return qv

    target = QuantumFloat(4)
    (target << init_state)()

    res = target.get_measurement()
    assert res == {3: 1.0}, f"Expected {{3: 1.0}}, got {res}"


def test_injection_superposition_uncompute():
    """Uncompute addition result that depends on a superposition."""
    a = QuantumFloat(3)
    b = QuantumFloat(3)

    h(a[0])
    b[:] = 2

    c = a + b

    with invert():
        (c << (lambda x, y: x + y))(a, b)

    res = c.get_measurement()
    assert res == {0: 1.0}, f"Expected {{0: 1.0}} after uncomputation, got {res}"

    c.delete()


def test_injection_chained():
    """Two sequential injections onto the same target."""
    def set_bit0():
        qv = QuantumFloat(3)
        x(qv[0])
        return qv

    def set_bit1():
        qv = QuantumFloat(3)
        x(qv[1])
        return qv

    target = QuantumFloat(3)

    (target << set_bit0)()
    (target << set_bit1)()

    res = target.get_measurement()
    assert res == {3: 1.0}, f"Expected {{3: 1.0}}, got {res}"
