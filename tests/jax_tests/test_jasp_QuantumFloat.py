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

def test_jasp_QuantumFloat():

    # Test decoder for QuantumFloat (Issue #271) 
    from qrisp import QuantumFloat, h
    from qrisp.jasp import terminal_sampling, qache, make_jaspr

    @terminal_sampling
    def main():
        a = QuantumFloat(3, -2, signed = True) 

        h(a)

        return a
    
    res = main()
    assert res == {-2.0: 0.0625, -1.75: 0.0625, -1.5: 0.0625, -1.25: 0.0625, -1.0: 0.0625, -0.75: 0.0625, -0.5: 0.0625, -0.25: 0.0625, 0.0: 0.0625, 0.25: 0.0625, 0.5: 0.0625, 0.75: 0.0625, 1.0: 0.0625, 1.25: 0.0625, 1.5: 0.0625, 1.75: 0.0625}
    
    # Test that the signed attribute behaves statically and the exponent 
    # attribute dynamically.
    
    @qache
    def inner(qf):
        assert isinstance(qf.signed, bool)
        assert not isinstance(qf.exponent, int)
        return qf.signed

    @make_jaspr
    def main():
        a = QuantumFloat(3, -1, signed = True)
        inner(a)
        inner(a)
        b = QuantumFloat(3, -1, signed = False)
        inner(b)
        inner(b)

        assert a.signed == True
        assert b.signed == False


