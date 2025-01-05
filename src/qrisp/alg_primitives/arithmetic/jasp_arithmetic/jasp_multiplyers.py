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

import numpy as np

from qrisp.alg_primitives.arithmetic import gidney_adder
from qrisp.jasp import qache, jrange
from qrisp.qtypes import QuantumFloat, QuantumBool
from qrisp.environments import control
from qrisp.core import cx, x

@qache(static_argnames = "inpl_adder")
def jasp_controlling_multiplyer(a, b, inpl_adder = gidney_adder):
    
    s = QuantumFloat(a.size + b.size)
    
    for i in jrange(b.size):
        with control(b[i]):
            inpl_adder(a, s[i:i+a.size+1])
    
    return s

@qache(static_argnames = "inpl_adder")
def jasp_squaring(a, inpl_adder = gidney_adder):
    
    s = QuantumFloat(2*a.size)
    temp = QuantumBool()
    
    for i in jrange(a.size):
        
        cx(a[i], temp[0])
        
        with control(temp[0]):
            inpl_adder(a, s[i:i+a.size+1])
            
        cx(a[i], temp[0])
        
    temp.delete()
    
    return s

# @qache(static_argnames = "inpl_adder")
def jasp_multiplyer(factor_1, factor_2, inpl_adder = gidney_adder):
    
    n = factor_1.size-1
    s = QuantumFloat(factor_1.size + factor_2.size, 
                     exponent = factor_1.exponent + factor_2.exponent)
    
    for i in jrange(factor_2.size):
        cx(factor_2[i], s[i+n])

    
    x(s)
    
    with control(factor_1[0], ctrl_state = 0):
        inpl_adder(factor_2, s)
    
    for j in jrange(s.size):
        cx(factor_1[1], s[j])
        
    for i in jrange(1, factor_1.size-1):
        
        inpl_adder(factor_2[:s.size-i], s[i-1:])    
        
        cx(factor_1[i], factor_1[i+1])
        for j in jrange(s.size):
            cx(factor_1[i+1], s[j])
        cx(factor_1[i], factor_1[i+1])

    inpl_adder(factor_2[:s.size-factor_1.size+1], s[factor_1.size-2:])
    
    for i in jrange(s.size):
        cx(factor_1[factor_1.size-1], s[i])

    x(s)

    return s