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

import pytest
import time

from qrisp import *
from qrisp.jasp import *


def test_khattar_dyn():
    
    @terminal_sampling
    def main(i, j):
        qf = QuantumFloat(i)
        h(qf)
        qbl = QuantumBool()
        mcx(qf.reg, qbl[0], method = "khattar", ctrl_state = j)
        return qf, qbl
    
    for i in range(1,8):
        for j in range(2**i):
            res_dict = main(i, j)
            
            for k in res_dict.keys():
                if k[0] == j:
                    assert k[1], f"Assertion failed: Expected True for i={i}, j={j}, but got False."
                else:
                    assert not k[1], f"Assertion failed: Expected False for i={i}, j={j}, but got True."
    
    
    # Test dynamic mcp
    @terminal_sampling
    def main(phi, i):
        
        qv = QuantumFloat(i)
    
        x(qv[:qv.size-1])
        
        with conjugate(h)(qv[qv.size-1]):
            mcp(phi, qv, method="khattar")
        
        return qv
      
    assert main(np.pi, 5) == {31: 1.0}

    @terminal_sampling
    def main(phi, i, j):
        
        qv = QuantumFloat(i)
    
        with conjugate(h)(qv[qv.size-1]):
            mcp(phi, qv, method="khattar", ctrl_state = j)
        
        return qv
    
    for i in range(2,8):
        for j in range(2**i):
            if j==0 or j==2**(i-1):        
                assert main(np.pi, i, j) == {2**(i-1): 1.0}
            else: 
                assert main(np.pi, i, j) == {0: 1.0}   
    
    @terminal_sampling
    def main(phi, i, j):
        
        qv = QuantumFloat(i)
    
        with conjugate(h)(qv[qv.size-1]):
            mcp(phi, [qv[i] for i in range(5)], method="khattar", ctrl_state = j)
        
        return qv
            
    for j in range(2**5):
        if j==0 or j==2**(4):        
            assert main(np.pi, 5, j) == {2**(4): 1.0}
        else: 
            assert main(np.pi, 5, j) == {0: 1.0}
    
    # Test no additional phase on output
    @terminal_sampling
    def main(i):
        
        qf = QuantumFloat(i)        
        qbl = QuantumBool()
        with conjugate(h)(qf[qf.size-1]):
            h(qbl[0])
            mcx(qf.reg, qbl[0], method = "balauca")
        return qf
    for i in range (1,8):                
        assert(main(i))=={0.0: 1.0}    

def test_khattar_stat():
    
    def test_mcx_inner(n,ctrl_state):
        ctrl = QuantumFloat(n)

        target = QuantumBool()

        h(ctrl)

        mcx(ctrl, target,  method = "khattar", ctrl_state = ctrl_state)
        meas_res = multi_measurement([ctrl, target])
        
        return meas_res
    
    for i in range(1,8):
        if i == 2:
            continue    
        for j in range(2**i):
            res_dict = test_mcx_inner(i, j)
            print(res_dict)
            for k in res_dict.keys():
                if k[0] == j:
                    assert k[1], f"Assertion failed: Expected True for i={i}, j={j}, but got False."
                else:
                    assert not k[1], f"Assertion failed: Expected False for i={i}, j={j}, but got True."


