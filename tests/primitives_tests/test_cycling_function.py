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

from qrisp import cyclic_shift, QuantumVariable, QuantumArray, QuantumFloat, OutcomeArray
import numpy as np
def test_cycling_function():
    
    n = 3
    
    
    for l in range(5):
        for k in range(3):
            
            qa = QuantumArray(QuantumVariable(1), 2**n)
            qa[0][:] = "1"
            
            for i in range(l):
                cyclic_shift(qa, k)
    
            assert qa[(k*l)%(len(qa))].get_measurement() == {"1": 1}
            
            qa = QuantumArray(QuantumVariable(1), 2**n+k)
            qa[0][:] = "1"
            
            for i in range(l):
                cyclic_shift(qa)
    
            assert qa[(l)%(len(qa))].get_measurement() == {"1": 1}
    
    qa = QuantumArray(QuantumFloat(3), 8)

    qa[:] = np.arange(len(qa))

    shift_amount = QuantumFloat(2, signed = True, qs = qa.qs)

    shift_amount[:] = {0: 1, -4 : 1, 1: 1}
    cyclic_shift(qa, shift_amount)
    assert qa.get_measurement() == {OutcomeArray([7, 0, 1, 2, 3, 4, 5, 6]): 0.3333333333333333, OutcomeArray([4, 5, 6, 7, 0, 1, 2, 3]): 0.3333333333333333, OutcomeArray([0, 1, 2, 3, 4, 5, 6, 7]): 0.3333333333333333}
