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

import numpy as np

def test_depth_computation():
    
    from qrisp import QuantumCircuit
    qc = QuantumCircuit(4)
    qc.cx(0,1)
    qc.x(1)
    qc.cx(1,2)
    qc.y(2)
    qc.cx(2,3)
    qc.cx(1,0)
    
    qc.cnot_depth() == 3
    
    qc = QuantumCircuit(2)
    qc.t(0)
    qc.cx(0,1)
    qc.rx(2*np.pi*3/2**4, 1)
    
    
    assert qc.t_depth(epsilon = 2**-5) == 16