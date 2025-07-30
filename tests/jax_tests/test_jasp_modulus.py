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

def test_jasp_create_measure_modulus():
    from qrisp import QuantumModulus, jaspify, measure

    @jaspify
    def test():
        a = QuantumModulus(13)
        a[:] = 7
        return measure(a)
    
    assert test() == 7

    @jaspify
    def test_m():
        a = QuantumModulus(13)
        a.m = 4
        a[:] = 7
        return measure(a)
    
    assert test_m() == 7

def test_jasp_qc_inplace_multiply_modulus():
    from qrisp import QuantumModulus, jaspify, measure

    @jaspify
    def test():
        a = QuantumModulus(13)
        a[:] = 7
        a *= 3
        return measure(a)
    
    assert test() == (7*3)%13

def test_jasp_qc_multiply_modulus():
    from qrisp import QuantumModulus, jaspify, measure

    @jaspify
    def test_l():
        a = QuantumModulus(13)
        a[:] = 7
        b = a * 3
        return measure(a), measure(b)
    
    assert test_l() == (7, (7*3)%13)

    @jaspify
    def test_r():
        a = QuantumModulus(13)
        a[:] = 7
        b = 3 * a
        return measure(a), measure(b)
    
    assert test_r() == (7, (7*3)%13)

def test_jasp_qq_multiply_modulus():
    from qrisp import QuantumModulus, jaspify, measure

    @jaspify
    def test():
        a = QuantumModulus(13)
        a.m = 4
        a[:] = 7
        b = QuantumModulus(13)
        b.m = 4
        b[:] = 12
        c = a * b
        return measure(a), measure(b), measure(c)
    
    assert test() == (7, 12, (7*12)%13)
