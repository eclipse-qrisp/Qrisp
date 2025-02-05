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

from qrisp import *
from qrisp.jasp import *

def test_conjugation_env():
    
    # Test base conjugation
    def conj_test_0():
        
        qv = QuantumFloat(3)
        
        with conjugate(h)(qv[0]):
            with conjugate(h)(qv[1]):
                with conjugate(h)(qv[2]):
                    z(qv[0])
                    z(qv[1])
                    z(qv[2])
            
        return measure(qv)


    jaspr = make_jaspr(conj_test_0)()

    assert jaspr() == 7
    
    # Test controlled conjugation    
    def conj_test_1():
        
        qv = QuantumFloat(2)
        a = QuantumVariable(3)
        
        def conjugator(qv):
            h(qv[0])
            y(qv[1])
        with control([a[0], a[1], a[2]]):
            with conjugate(conjugator)(qv):
                z(qv[0])
            

    jaspr = make_jaspr(conj_test_1)()
    assert jaspr.to_qc().cnot_count() == 29


