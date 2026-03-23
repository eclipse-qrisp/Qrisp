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

from qrisp import conjugate, QuantumFloat, p, QFT, control, PTControlledOperation
import numpy as np

def test_conjugation_env():
    
    def fourier_adder(qf, n):
    
        with conjugate(QFT)(qf):
    
            for i in range(qf.size):
                p(n*np.pi*2**(i-qf.size+1), qf[i])
                
    
    qf = QuantumFloat(5)
    fourier_adder(qf, 3)
    
    assert qf.get_measurement() == {3: 1.0}
    
    fourier_adder(qf, 2)
    
    assert qf.get_measurement() == {5: 1.0}
    
    ctrl = QuantumFloat(1)
    qf = QuantumFloat(5)

    with control(ctrl):
        fourier_adder(qf, 3)
        
    transpiled_qc = qf.qs.transpile(1)
    
    for instr in transpiled_qc.data:
        
        if "conjugator" in instr.op.name:
            assert not isinstance(instr.op, PTControlledOperation)
        
        
        
        