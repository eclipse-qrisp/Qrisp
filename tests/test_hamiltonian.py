"""
\********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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
********************************************************************************/
"""

def test_hamiltonian():

    from qrisp import QuantumVariable, QuantumArray, h
    from qrisp.operators import X,Y,Z
            
    qv = QuantumVariable(2)
    h(qv)
    H = Z(0)*Z(1)
    res = H.get_measurement(qv)
    assert res == 0.0

    qtype = QuantumVariable(2)
    q_array = QuantumArray(qtype, shape=(2))
    h(q_array)
    H = Z(0)*Z(1) + X(2)*X(3)
    res = H.get_measurement(q_array)
    assert res == 1.0