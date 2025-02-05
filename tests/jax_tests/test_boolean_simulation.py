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

from qrisp import QuantumFloat, measure
from qrisp.jasp import boolean_simulation, jrange

def test_boolean_simulation():
    
    @boolean_simulation
    def main(i, j):
        
        a = QuantumFloat(10)
        
        b = QuantumFloat(10)
        
        a[:] = i
        b[:] = j
        
        c = QuantumFloat(30)
        
        for i in jrange(150): 
            c += a*b
        
        return measure(c)
    
    for i in range(5):
        for j in range(5):
            assert main(i, j) == 150*i*j