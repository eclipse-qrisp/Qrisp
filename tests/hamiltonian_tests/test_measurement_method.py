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
from qrisp import *

def test_measurement_method():

    def testing_helper(qv):
        operator_list = [lambda x : 1, X, Y, Z, A, C, P0, P1]
        for O0 in operator_list: 
            for O1 in operator_list:
                for O2 in operator_list:
                    for O3 in operator_list:
                        H = O0(0)*O1(1)*O2(2)*O3(3)
                        if isinstance(H, int):
                            continue
                        
                        print(H)
                        assert abs(H.get_measurement(qv, precision = 0.001) - H.to_pauli().get_measurement(qv, precision = 0.001)) < 1E-2

    qv = QuantumVariable(4)
    
    testing_helper(qv)
    
    h(qv[0])
    
    testing_helper(qv)
    
    cx(qv[0], qv[1])
    
    testing_helper(qv)
    
    cx(qv[0], qv[2])
    
    testing_helper(qv)
    
    h(qv[0])
    
    
    
    
    