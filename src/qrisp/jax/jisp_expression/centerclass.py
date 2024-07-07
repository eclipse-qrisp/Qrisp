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

from jax import jit, make_jaxpr
from jax.core import Jaxpr

from qrisp.jax.jisp_expression import invert_jispr
from qrisp.jax import AbstractQuantumCircuit

class Jispr(Jaxpr):
    
    __slots__ = "permeability", "isqfree", "hashvalue"
    
    def __init__(self, *args, permeability = {}, isqfree = None, **kwargs):
        
        self.permeability = permeability
        self.isqfree = isqfree
        
        if len(args) == 1:
            kwargs["jaxpr"] = args[0]
        
        if "jaxpr" in kwargs:
            jaxpr = kwargs["jaxpr"]

            self.hashvalue = hash(jaxpr)
        
            Jaxpr.__init__(self,
                           constvars = jaxpr.constvars,
                           invars = jaxpr.invars,
                           outvars = jaxpr.outvars,
                           eqns = jaxpr.eqns,
                           effects = jaxpr.effects,
                           debug_info = jaxpr.debug_info
                           )
        else:
            self.hashvalue = id(self)
            
            Jaxpr.__init__(self, **kwargs)
            
        if not isinstance(self.invars[0].aval, AbstractQuantumCircuit):
            print(type(self.invars[0].aval))
            raise Exception
        
    def __hash__(self):
        
        return self.hashvalue
    
    def __eq__(self, other):
        if not isinstance(other, Jaxpr):
            return False
        return self.hashvalue == hash(other)
    
    def inverse(self):
        return invert_jispr(self)


def make_jispr(fun):
    from qrisp.jax import get_tracing_qs, AbstractQuantumCircuit
    def jispr_creator(*args, **kwargs):
        
        def ammended_function(qc, *args, **kwargs):
            
            qs = get_tracing_qs()
            temp = qs.abs_qc
            qs.abs_qc = qc
            
            res = fun(*args, **kwargs)
                
            res_qc = qs.abs_qc
            qs.abs_qc = temp
            
            return qc, res
        
        jaxpr = make_jaxpr(ammended_function)(AbstractQuantumCircuit(), *args, **kwargs).jaxpr
        
        return Jispr(jaxpr)
    
    return jispr_creator
        
        
        
            
            
            
            
            
            
    
