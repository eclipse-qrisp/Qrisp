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

from qrisp.operators import X, Y, Z, A, C, P0, P1
from numpy.linalg import norm

def test_to_pauli_converter():

    operator_list = [lambda x : 1, X, Y, Z, A, C, P0, P1]

    for O0 in operator_list: 
        for O1 in operator_list:
            for O2 in operator_list:
                H = O0(0)*O1(1)*O2(2)
                if isinstance(H, int):
                    continue
                
                assert norm(H.to_array() - H.to_pauli().to_array()) < 1E-5

