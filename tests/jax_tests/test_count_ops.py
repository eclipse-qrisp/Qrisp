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


def test_count_ops():

    # Test general pipeline
    @count_ops
    def main(i):
        
        a = QuantumFloat(i)
        b = QuantumFloat(i)
        
        c = a*b
        
        return measure(c)
    
    print(main(5))
    print(main(5000))
    
    # Test pipeline correctness
    
    def main(i):
        
        qf = QuantumFloat(i)
        for i in jrange(qf.size):
            x(qf[i])
        
        with invert():
            y(qf)
        
        return qf
    
    for i in range(1, 10):
        assert main(i).qs.compile().count_ops() == count_ops(main)(i)
        

    # Test kernilization error message    
    def state_prep():
        qf = QuantumFloat(3)
        h(qf)
        return qf
    
    @count_ops
    def main():
        return expectation_value(state_prep, 10)()
    
    try:
        main()
    except Exception as e:
        if "kernel" in str(e):
            return
        else:
            assert False
        
        
        
            
        