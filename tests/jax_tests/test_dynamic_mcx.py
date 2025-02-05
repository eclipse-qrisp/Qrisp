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

from qrisp import *
from qrisp.jasp import *

def test_dynamic_mcx():
    
    @terminal_sampling
    def main(i, j):
        qf = QuantumFloat(i)
        h(qf)
        qbl = QuantumBool()
        mcx(qf.reg, qbl[0], method = "balauca", ctrl_state = j)
        return qf, qbl
    
    for i in range(1, 5):
        for j in range(2**i):
            res_dict = main(i, j)
            
            for k in res_dict.keys():
                if k[0] == j:
                    assert k[1]
                else:
                    assert not k[1]

    # Test static list behavior
    @terminal_sampling
    def main(j):
        qf = QuantumFloat(5)
        h(qf)
        qbl = QuantumBool()
        qb_list = [qf[i] for i in range(5)]
        mcx(qf.reg, qbl[0], method = "balauca", ctrl_state = j)
        return qf, qbl

    for j in range(2**5):
        res_dict = main(j)
        
        for k in res_dict.keys():
            if k[0] == j:
                assert k[1]
            else:
                assert not k[1]
                
    # Test dynamic mcp
        
    @jaspify
    def main(phi, i):
        
        qv = QuantumFloat(i)
    
        x(qv[:qv.size-1])
        
        with conjugate(h)(qv[qv.size-1]):
            mcp(phi, qv)
        
        return measure(qv)
            
    assert main(np.pi, 5) == 31

    @jaspify
    def main(phi, i, j):
        
        qv = QuantumFloat(i)
    
        with conjugate(h)(qv[qv.size-1]):
            mcp(phi, qv, ctrl_state = j)
        
        return measure(qv)
            
    assert main(np.pi, 5, 0) == 16        
    
    @jaspify
    def main(phi, i, j):
        
        qv = QuantumFloat(i)
    
        with conjugate(h)(qv[qv.size-1]):
            mcp(phi, [qv[i] for i in range(5)], ctrl_state = j)
        
        return measure(qv)
            
    assert main(np.pi, 5, 0) == 16        