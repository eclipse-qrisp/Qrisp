"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

import pytest
import time

from qrisp import *

def test_fermionic_term():
    from qrisp.operators.fermionic import a, c
    
    O_0 = a(0)*c(1)
    O_1 = c(1)*a(0)

    assert (O_0 == O_1) == False
    
    O_0 = a(0)*c(1)
    O_1 = -c(1)*a(0)

    assert (O_0.hermitize() == O_1.hermitize()) == True
    
    O_0 = a(0)*c(1)
    O_1 = -1*c(1)*a(0)

    assert (O_0.hermitize() == O_1.hermitize()) == True
    
    O_0 = a(0)*c(1)*a(2)
    O_1 = c(2)*a(1)*c(0)

    assert (O_0 == O_1) == True
    
    O = 3*a(0)*c(1) + c(1)*a(0)
    O = O.reduce()
    assert str(O) == "2*a0*c1"
    
    O = a(0)*a(1) - c(1)*c(0)
    O = O.reduce(assume_hermitian=True)
    assert str(O) == "0"
    
