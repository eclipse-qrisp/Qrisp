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

from qrisp import *
from qrisp.jasp import *


def test_modular_adder():

    def call_mod_adder():

        a = QuantumFloat(5)
        b = QuantumFloat(5)

        a[:] = 5
        b[:] = 4
        jasp_mod_adder(a, b, 8, inpl_adder=jasp_fourier_adder)

        return measure(b)

    jaspr = make_jaspr(call_mod_adder)()
    assert jaspr() == 1


def test_fourier_adder():

    def call_fourier_adder():

        a = QuantumFloat(5)
        b = QuantumFloat(5)

        a[:] = 5
        b[:] = 4
        jasp_fourier_adder(a, b)

        return measure(b)

    jaspr = make_jaspr(call_fourier_adder)()
    assert jaspr() == 9


def test_out_of_place_arithmetic():

    @terminal_sampling
    def main(i):
        a = QuantumFloat(i)
        b = QuantumFloat(i)
        h(a)
        h(b)
        c = a + b
        return a, b, c

    for i in range(1, 5):
        sampling_dic = main(i)

        for a, b, c in sampling_dic.keys():
            assert (a + b) % 2**i == c

    @terminal_sampling
    def main(i):
        a = QuantumFloat(i)
        b = QuantumFloat(i)
        h(a)
        h(b)
        c = a - b
        return a, b, c

    for i in range(1, 5):
        sampling_dic = main(i)

        for a, b, c in sampling_dic.keys():
            assert (a - b) % 2**i == c


def test_pow():

    for i in range(1, 4):
        for j in range(3):

            @terminal_sampling
            def main(i):
                a = QuantumFloat(i)
                h(a)
                b = a**j
                return a, b

            sampling_dic = main(i)
            for a, b in sampling_dic.keys():
                assert a**j == b


def test_non_trivial_exponent_addition():

    @boolean_simulation
    def main(k, e):
        a = QuantumFloat(10, e)
        a += k
        b = QuantumFloat(10, e)
        b -= k
        return measure(a), measure(b)

    for e in range(-4, 2):
        for k in range(0, 2**5, 4):
            assert main(k, e) == (k % (2 ** (10 + e)), (-k) % (2 ** (10 + e)))

    @boolean_simulation
    def main(k, l, e0, e1):
        a = QuantumFloat(10, e0)
        a[:] = k
        b = QuantumFloat(10, e1)
        b[:] = l
        a += b
        return measure(a)

    for e0 in range(-4, 2):
        for e1 in range(-4, 2):
            for k in range(0, 2**5, 4):
                for l in range(0, 2**5, 4):
                    assert main(k, l, e0, e1) == (k + l) % (2 ** (10 + e0))
