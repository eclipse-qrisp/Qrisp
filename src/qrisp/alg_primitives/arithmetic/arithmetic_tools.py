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

from qrisp.qtypes import QuantumFloat
from qrisp.environments import control, invert
from qrisp.core.gate_application_functions import cx
from qrisp.jasp import jrange 

def qmax(a: QuantumFloat, b: QuantumFloat) -> QuantumFloat:

    res = QuantumFloat(a.size)
    c = (a >= b)
    
    with control(c):
        for i in jrange(a.size):
            cx(a[i], res[i])
    with control(c, 0):
        for i in jrange(b.size):
            cx(b[i], res[i])
    
    compare_func = lambda x , y: x >= y
    
    with invert():
        (c << compare_func)(a,b)
    
    c.delete()
            
    return res


def qmin(a: QuantumFloat, b: QuantumFloat) -> QuantumFloat:

    res = QuantumFloat(a.size)
    c = (a <= b)

    with control(c):
        for i in jrange(a.size):
            cx(a[i], res[i])
    with control(c, 0):
        for i in jrange(b.size):
            cx(b[i], res[i])
    
    compare_func = lambda x , y: x <= y
    
    with invert():
        (c << compare_func)(a,b)
    
    c.delete()
            
    return res