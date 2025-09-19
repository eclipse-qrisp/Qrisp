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

from qrisp.core import Qubit
from qrisp.qtypes import QuantumFloat
from qrisp.environments import control, invert
from qrisp.core.gate_application_functions import x, cx, mcx
from qrisp.jasp import jrange
from typing import Tuple


def q_max(a: QuantumFloat, b: QuantumFloat) -> QuantumFloat:
    """Returns a QuantumFloat containing the maximum of two QuantumFloats a and b
    a: QuantumFloat
    b: QuantumFloat
    return: QuantumFloat
    """

    res = QuantumFloat(a.size)
    c = a >= b

    with control(c):
        for i in jrange(a.size):
            cx(a[i], res[i])
    with control(c, 0):
        for i in jrange(b.size):
            cx(b[i], res[i])

    compare_func = lambda x, y: x >= y

    with invert():
        (c << compare_func)(a, b)

    c.delete()

    return res


def q_min(a: QuantumFloat, b: QuantumFloat) -> QuantumFloat:
    """Returns a QuantumFloat containing the minimum of two QuantumFloats a and b
    a: QuantumFloat
    b: QuantumFloat
    return: QuantumFloat
    """
    res = QuantumFloat(a.size)
    c = a <= b

    with control(c):
        for i in jrange(a.size):
            cx(a[i], res[i])
    with control(c, 0):
        for i in jrange(b.size):
            cx(b[i], res[i])

    compare_func = lambda x, y: x <= y

    with invert():
        (c << compare_func)(a, b)

    c.delete()

    return res


def q_floor(a: QuantumFloat) -> QuantumFloat:
    """Returns a QuantumFloat containing the floor of a QuantumFloat a
    a: QuantumFloat
    return: QuantumFloat
    """

    b = a.duplicate()
    for i in jrange(-a.exponent, a.size):
        cx(a[i], b[i])

    return b


def q_ceil(a: QuantumFloat) -> QuantumFloat:
    """Returns a QuantumFloat containing the ceiling of a QuantumFloat a
    a: QuantumFloat
    return: QuantumFloat
    """
    b = q_floor(a)

    with control(a.exponent < 0):
        flag = QuantumFloat(1)

        # Logical OR on the fractional part
        logical_OR(a[:-a.exponent], flag)

        with control(flag[0]):
            b += 1

        logical_OR(a[:-a.exponent], flag)
        flag.delete()
    return b


def logical_OR(operands: list[Qubit], flag: Qubit):
    # handle the case fo an empty list
    # Implement the case of a Quantum Variable with more the 2 qubits
    x(operands)
    x(flag)
    mcx(operands, flag[0])
    x(operands)

def q_round(a: QuantumFloat) -> QuantumFloat:
    """Returns a QuantumFloat containing the rounded value of a QuantumFloat a
    a: QuantumFloat
    return: QuantumFloat
    """
    b = q_floor(a)

    with control(a.exponent < 0):
        with control(a[-a.exponent - 1]):
            b += 1

    return b


def q_fractional(a: QuantumFloat) -> QuantumFloat:
    """Returns a QuantumFloat containing the fractional part of a QuantumFloat a
    a: QuantumFloat
    return: QuantumFloat
    """
    b = a.duplicate()
    for i in jrange(a.size):
        cx(a[i], b[i])

    if a.signed:
        flag = QuantumFloat(1)
        cx(a[-1], flag)
        with control(flag):
            c = q_ceil(a)
            b -= c
            injected_qceil = c << q_ceil
            with invert():
                injected_qceil(a)
            c.delete()
            
        with control(flag, 0):
            c = q_floor(a)
            b -= c
            injected_qceil = c << q_floor
            with invert():
                injected_qceil(a)
            c.delete()
        cx(a[-1], flag)
        flag.delete()
    else:
        c = q_floor(a)
        b -= c
        injected_qceil = c << q_floor
        with invert():
            injected_qceil(a)
        c.delete()
    return b


def q_modf(a: QuantumFloat) -> Tuple[QuantumFloat, QuantumFloat]:
    """Returns a tuple of two QuantumFloats containing the floor and fractional part of a QuantumFloat a
    a: QuantumFloat
    return: (QuantumFloat, QuantumFloat)
    """
    return (q_floor(a), q_fractional(a))
