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


def test_modular_adder():

    def call_mod_adder():
        
        a = QuantumFloat(5)
        b = QuantumFloat(5)
        
        a[:] = 5
        b[:] = 4
        jasp_mod_adder(a, b, 8, inpl_adder = jasp_fourier_adder)
        
        return measure(b)

    jaspr = make_jaspr(call_mod_adder)()
    assert jaspr() == 1

def test_fourier_adder():

    def call_fourier_adder():
        
        a = QuantumFloat(5)
        b = QuantumFloat(5)
        
        a[:] = 5
        b[:] = 4
        jasp_fourier_adder(a, b)
        
        return measure(b)

    jaspr = make_jaspr(call_fourier_adder)()
    assert jaspr() == 9
    
    