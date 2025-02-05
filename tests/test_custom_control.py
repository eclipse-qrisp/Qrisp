"""
\********************************************************************************
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
********************************************************************************/
"""

import time
import random
import numpy as np

from qrisp import QuantumVariable, QuantumBool, custom_control, h, x, y, cp, control, invert, QuantumFloat, cx, qcla

def test_custom_control():
    
    qv_0 = QuantumVariable(2)
    qv_1 = QuantumVariable(2)

            
    qv_0 = QuantumVariable(1)
    qv_1 = QuantumVariable(1)
    qf = QuantumFloat(1)
            
    @custom_control
    def test_function(qv, ctrl = None):
        if ctrl is None:
            y(qv)
        else:
            temp = QuantumVariable(1)
            cx(qv, temp)
            cp(np.pi/4, ctrl, qv)
            x(qv)
            temp.uncompute()
        return None


    with control(qv_0):
        test_function(qv_1)
        with invert():
            x(qv_1)
        test_function(qv_1)
        with invert():
            test_function(qv_1)
            x(qv_1)
            with qf == 0:
                test_function(qv_1)
                
        test_function(qv_1)
    # print(qv_1.qs)
    assert qv_1.get_measurement() == {"1" : 1.0}
    
    # Test whether qubit management and compilation are still working without performance loss
    for n in [3,4,7,8]:
        for a in range(2**n-1):
            
            for i in range(6):
                b = QuantumFloat(n)
                h(b)
                
                qbl = QuantumBool()
                
                # Semi classical qcla is custom controlled
                with control(qbl):
                    qcla(a, b)
                
                from qrisp import t_depth_indicator
                gate_speed = lambda x : t_depth_indicator(x, epsilon = 2**-10)
                qc = b.qs.compile(gate_speed = gate_speed, compile_mcm = True)
                
                env_num_qubits = qc.num_qubits()
                env_t_depth = qc.t_depth()
                
                b = QuantumFloat(n)
                h(b)
                qbl = QuantumBool()
                
                qcla(a, b, ctrl = qbl[0])
                
                qc = b.qs.compile(gate_speed = gate_speed, compile_mcm = True)
                if env_num_qubits <= qc.num_qubits() +2 and env_t_depth <= qc.t_depth() + 2:
                    break
                
            else:
                assert False
        