"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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

from qrisp import QuantumFloat, h, t, conjugate, measure
from qrisp.jasp import jaspify, sample, jrange

def test_sampling():
    
    def inner_f(i):
        qf = QuantumFloat(4)
        
        with conjugate(h)(qf):
            for k in jrange(i):
                t(qf[0])
                
        return qf

    @jaspify
    def main():
        res = sample(inner_f, 100)(2)
        return res
    
    assert set(int(i) for i in main()) == {0,1}
    
    def inner_f(i):
        qf = QuantumFloat(4)
        qf_2 = QuantumFloat(4)
        qf_3 = QuantumFloat(4)
        with conjugate(h)(qf):
            for k in jrange(i):
                t(qf[0])
                
        return qf, qf_2, qf_3

    
    @jaspify
    def main():
        
        res = sample(inner_f, 10)(2)
        
        return res
    
    assert main().shape == (10, 3)

