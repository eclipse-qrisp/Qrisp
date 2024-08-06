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

def test_redirection_operator():
    
    qbl = QuantumBool()

    qf = QuantumFloat(4)
    qbl.flip()
    cx(qbl, qf[1])
    qbl <<= (qf < 4)

    assert qbl.most_likely() == False
    
    a = QuantumFloat(4)
    a[:] = 3
    b = QuantumFloat(4)
    b[:] = 4
    
    c = QuantumFloat(6)
    c[:] = 7
    
    with invert():
        c <<= a + b
    
    assert c.most_likely() == 0
    
    a = QuantumFloat(4)
    a[:] = 3
    b = QuantumFloat(4)
    b[:] = 4
    
    c = QuantumFloat(6)
    c[:] = 7
    
    qbl = QuantumBool()
    with control(qbl):
        with invert():
            c <<= a + b
        
    assert c.most_likely() == 7
