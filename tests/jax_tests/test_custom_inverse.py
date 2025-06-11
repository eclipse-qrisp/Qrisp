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

def test_custom_inverse():
    
    @custom_inversion
    def c_inv_function(qbl, inv = False):
        if inv:
            pass
        else:
            qbl.flip()

    def recursion(qbl, recursion_level):
        
        if recursion_level == 0:
            c_inv_function(qbl)
        else:
            with QuantumEnvironment():
                with invert():
                    recursion(qbl, recursion_level = recursion_level - 1)

    @jaspify
    def main():
        qbl = QuantumBool()
        recursion(qbl, 6)
        return measure(qbl)

    assert main() == True

    @jaspify
    def main():
        qbl = QuantumBool()
        recursion(qbl, 5)
        return measure(qbl)

    assert main() == False
    
    @custom_control
    @custom_inversion
    def c_inv_control_function(qf, inv = False, ctrl = None):
        
        if inv and ctrl is None:
            qf[:] = 1
        if not inv and ctrl is not None:
            qf[:] = 2
        if inv and ctrl is not None:
            qf[:] = 3
        if not inv and ctrl is None:
            qf[:] = 4

    @jaspify
    def main(j):
        
        qbl = QuantumBool()
        qf = QuantumFloat(4)
        
        for i in range(8):
            
            if i&1:
                env_0 = invert()
            else:
                env_0 = QuantumEnvironment()
                
            if i&2:
                env_1 = control(qbl)
            else:
                env_1 = QuantumEnvironment()
                
            if i&4:
                env_0, env_1 = env_1, env_0
            
            with control(j == i):
                with env_0:
                    with env_1:
                        c_inv_control_function(qf)
        
        return measure(qf)

    for i in range(8):
        assert main(i)%4 == i%4
        
    @custom_inversion
    @custom_control
    def c_inv_control_function(qf, inv = False, ctrl = None):
        
        if inv and ctrl is None:
            qf[:] = 1
        if not inv and ctrl is not None:
            qf[:] = 2
        if inv and ctrl is not None:
            qf[:] = 3
        if not inv and ctrl is None:
            qf[:] = 4

    @jaspify
    def main(j):
        
        qbl = QuantumBool()
        qf = QuantumFloat(4)
        
        for i in range(8):
            
            if i&1:
                env_0 = invert()
            else:
                env_0 = QuantumEnvironment()
                
            if i&2:
                env_1 = control(qbl)
            else:
                env_1 = QuantumEnvironment()
                
            if i&4:
                env_0, env_1 = env_1, env_0
            
            with control(j == i):
                with env_0:
                    with env_1:
                        c_inv_control_function(qf)
        
        return measure(qf)

    for i in range(8):
        assert main(i)%4 == i%4
        
    @custom_inversion
    def inner_f(qv, inv = False):
        if inv:
            measure(qv[0])
            x(qv[0])
        else:
            measure(qv[0])
            z(qv[0])

    def main():
        
        qv = QuantumBool()
        qbl = QuantumBool()
        qbl.flip()
        
        with control(qbl):
            with QuantumEnvironment():
                with conjugate(h)(qv):
                    with invert():
                        with conjugate(h)(qv):
                            inner_f(qv)

        return measure(qv)
    jsp = make_jaspr(main)()
    assert jsp() == True
    
    



