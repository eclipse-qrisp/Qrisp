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


from qrisp.core import QuantumVariable
from qrisp.qtypes import QuantumBool
from qrisp.environments import invert, adaptive_condition

def uint_qq_less_than(a, b, inpl_adder):
    
    res = QuantumBool()
    if a.size < b.size:
        temp_var = QuantumVariable(b.size-a.size)
        a = list(a) + list(temp_var)
    
    with invert():
        inpl_adder(b, list(a) + [res[0]])
    inpl_adder(b, a)
    
    try:
        temp_var.delete()
    except UnboundLocalError:
        pass
    
    return res

def uint_cq_less_than(a, b, inpl_adder):
    
    res = QuantumBool()
    
    if a > 2**b.size:
        res.flip()
        return res
    
    elif a < 0:
        return res
    
    inpl_adder(-a-1, list(b) + [res[0]])
    inpl_adder(a+1, b)
    
    return res.flip()

def uint_qc_less_than(a, b, inpl_adder):
    
    res = QuantumBool()
    
    if b > 2**a.size:
        res.flip()
        return res
    elif b < 0:
        return res
    
    inpl_adder(-b, list(a) + [res[0]])
    inpl_adder(b, a)
    
    return res

def uint_less_than(a,b, inpl_adder):
    
    if isinstance(a, QuantumVariable) and isinstance(b, QuantumVariable):
        return uint_qq_less_than(a, b, inpl_adder)
    elif isinstance(a, QuantumVariable):
        return uint_qc_less_than(a, b, inpl_adder)
    elif isinstance(b, QuantumVariable):
        return uint_cq_less_than(a, b, inpl_adder)
    else:
        return a < b

@adaptive_condition
def uint_lt(a, b, inpl_adder):
    return uint_less_than(a, b, inpl_adder)

@adaptive_condition
def uint_gt(a, b, inpl_adder):
    return uint_less_than(b, a, inpl_adder)

@adaptive_condition
def uint_le(a, b, inpl_adder):
    return uint_less_than(b, a, inpl_adder).flip()

@adaptive_condition
def uint_ge(a, b, inpl_adder):
    return uint_less_than(a, b, inpl_adder).flip()
