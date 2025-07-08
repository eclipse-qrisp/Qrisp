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

def test_jasp_QuantumBool():

    from qrisp import QuantumBool, measure
    from qrisp.jasp import jaspify

    @jaspify
    def main(i,j):
        a = QuantumBool()
        a[:] = i
        b = QuantumBool()
        b[:] = j
    
        return measure(a), measure(b)

    res = main(True, False)
    assert res[0] == True
    assert res[1] == False
