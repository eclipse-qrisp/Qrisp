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

from qrisp import *
from qrisp.jasp import *


def test_qubit_array_fusion():
    
    @jaspify
    def main():
        qf =QuantumFloat(3)
        x(qf)
        qf.extend(1)
        a = measure(qf)
        return a

    assert main() == 7
    
    @jaspify
    def main():
        qf =QuantumFloat(3)
        x(qf)
        qf.extend(1, position = 0)
        
        a = measure(qf)
        return a

    assert main() == 14
    
    @jaspify
    def main():
        qf =QuantumFloat(3)
        x(qf)
        qf.extend(1, position = qf.size//2)
        a = measure(qf)
        return a
    
    assert main() == 13
    
    def main():
        
        qv_a = QuantumFloat(3).ensure_reg()
        qv_b = QuantumFloat(3).ensure_reg()
        
        tmp = qv_a + qv_b
        x(tmp)
        
        return measure(tmp)

    assert jaspify(main)() == 63
    assert boolean_simulation(main)() == 63
    
    
    
    
    try:
        import catalyst
    except:
        return
    
    assert qjit(main)() == 63