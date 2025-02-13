"""
\********************************************************************************
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
********************************************************************************/
"""

import jax.numpy as jnp

from qrisp.core import QuantumVariable, cx
from qrisp.qtypes import QuantumBool
from qrisp.environments import invert, adaptive_condition, conjugate, control


def uint_qq_less_than(a, b, inv_adder):
    comparison_anc = QuantumBool()
    comparison_res = QuantumBool()
    
    temp_var = QuantumVariable(jnp.maximum(0, b.size-a.size))
    a = a.ensure_reg() + temp_var.ensure_reg()
    
    with conjugate(inv_adder, allocation_management = False)(b, a + comparison_anc.ensure_reg()):
        cx(comparison_anc, comparison_res)
    
    comparison_anc.delete()
    
    temp_var.delete()
    
    return comparison_res

def uint_cq_less_than(a, b, inv_adder):
    
    comparison_res = QuantumBool()
    
    
    with control(a > 2**b.size):
        comparison_res.flip()
    
    with control(a >= 0):    
        comparison_anc = QuantumBool()
        
        with conjugate(inv_adder, allocation_management = False)(a+1, b.reg + comparison_anc.reg):
            cx(comparison_anc, comparison_res)
        
        comparison_anc.delete()
    
    return comparison_res.flip()


def uint_qc_less_than(a, b, inv_adder):
    
    comparison_res = QuantumBool()
    
    
    with control(b > 2**a.size):
        comparison_res.flip()
    with control(b >= 0):
        comparison_anc = QuantumBool()
        
        with conjugate(inv_adder, allocation_management = False)(b, a.reg + comparison_anc.reg):
            cx(comparison_anc, comparison_res)
        
        comparison_anc.delete()
    
    return comparison_res

def uint_less_than(a,b, inpl_adder):
    
    def inv_adder(a, b):
        with invert():
            inpl_adder(a, b)
    
    if isinstance(a, QuantumVariable) and isinstance(b, QuantumVariable):
        return uint_qq_less_than(a, b, inv_adder)
    elif isinstance(a, QuantumVariable):
        return uint_qc_less_than(a, b, inv_adder)
    elif isinstance(b, QuantumVariable):
        return uint_cq_less_than(a, b, inv_adder)
    else:
        return a < b

def uint_lt(a, b, inpl_adder):
    return uint_less_than(a, b, inpl_adder)

def uint_gt(a, b, inpl_adder):
    return uint_less_than(b, a, inpl_adder)

def uint_le(a, b, inpl_adder):
    return uint_less_than(b, a, inpl_adder).flip()

def uint_ge(a, b, inpl_adder):
    return uint_less_than(a, b, inpl_adder).flip()
