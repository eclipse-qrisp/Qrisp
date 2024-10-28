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

import pytest
import time

from qrisp import *

def test_fermionic_term():
    from qrisp.operators.fermionic import a, c
    
    H_0 = a(0)*c(1)
    H_1 = c(1)*a(0)

    assert (H_0 == H_1) == False
    
    H_0 = a(0)*c(1)
    H_1 = -c(1)*a(0)

    assert (H_0 == H_1) == True
    
    H_0 = a(0)*c(1)
    H_1 = -1*c(1)*a(0)

    assert (H_0 == H_1) == True
    
    H_0 = a(0)*c(1)*a(2)
    H_1 = c(2)*a(1)*c(0)

    assert (H_0 == H_1) == True
    
    H = 3*a(0)*c(1) + c(0)*a(1)
    assert str(H) == "a0*c1 + a1*c0"
    
    H = a(0)*a(1) + a(1)*a(0)
    assert str(H) == "0"
    
