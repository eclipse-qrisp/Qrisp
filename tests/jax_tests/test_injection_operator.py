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
from qrisp.jasp import *
from jax import make_jaxpr

def test_injection_operator():
    
    @jaspify
    def main(i):
        
        a = QuantumFloat(i)
        b = QuantumFloat(i)
        
        h(a)
        h(b)
        
        s = a*b
        
        with invert():
            (s << (lambda a, b : a*b))(a,b)
        
        return measure(s)
    
    for i in range(2, 6):
        assert main(i) == 0
        
    def AND(a, b):
        res = QuantumBool()
        mcx([a, b], res)
        return res

    @jaspify
    def main():
        a = QuantumBool()
        b = QuantumBool()

        tar = QuantumBool()

        (tar << AND)(a,b)

        res = measure(tar)
        return res

    main()

    def init_psi():
        qv = QuantumFloat(5)
        x(qv[0])
        return qv

    @jaspify
    def main():
        qv = QuantumFloat(5)

        (qv << init_psi)()
        return measure(qv)

    assert main() == 1


def test_injection_and():
    """Inject AND gate onto a pre-flipped target."""
    def AND(a, b):
        res = QuantumBool()
        mcx([a, b], res)
        return res

    @jaspify
    def main():
        a = QuantumBool()
        b = QuantumBool()
        target = QuantumBool()
        target.flip()
        (target << AND)(a, b)
        return measure(target)

    assert main() == 1


def test_injection_uncomputation():
    """Uncomputation via injection + invert."""
    @jaspify
    def main(n):
        a = QuantumFloat(n)
        b = QuantumFloat(n)
        a[:] = 3
        b[:] = 5

        c = a * b

        with invert():
            (c << (lambda x, y: x * y))(a, b)

        return measure(c)

    for n in range(4, 7):
        assert main(n) == 0, f"Uncomputation failed for n={n}"


def test_injection_state_prep():
    """Inject a state-prep function onto an existing variable."""
    def init_state():
        qv = QuantumFloat(4)
        x(qv[0])
        x(qv[1])
        return qv

    @jaspify
    def main():
        target = QuantumFloat(4)
        (target << init_state)()
        return measure(target)

    assert main() == 3


def test_injection_superposition_uncompute():
    """Uncompute addition from superposition."""
    @jaspify
    def main():
        a = QuantumFloat(3)
        b = QuantumFloat(3)

        h(a[0])
        b[:] = 2

        c = a + b

        with invert():
            (c << (lambda x, y: x + y))(a, b)

        return measure(c)

    assert main() == 0


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

    @jaspify
    def main():
        target = QuantumFloat(3)
        (target << set_bit0)()
        (target << set_bit1)()
        return measure(target)

    assert main() == 3
