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

from jax.core import JaxprEqn, Literal, ClosedJaxpr
from jax import jit, make_jaxpr
from qrisp.jisp.interpreter_tools import eval_jaxpr, extract_invalues, insert_outvalues, reinterpret

def evaluate_pjit_eqn(pjit_primitive, *args, **kwargs):
    
    definition_jaxpr = kwargs["jaxpr"].jaxpr
    
    res = jit(eval_jaxpr(definition_jaxpr), inline = True)(*args)
    
    if len(definition_jaxpr.outvars) == 1:
        res = [res]

    return res
                
# Flattens/Inlines a pjit calls in a jaxpr
def flatten_pjit(jaxpr):
    
    if isinstance(jaxpr, ClosedJaxpr):
        jaxpr = jaxpr.jaxpr
    
    def eqn_evaluator(primitive, *args, **kwargs):
        if primitive.name == "pjit":
            return evaluate_pjit_eqn(primitive, *args, **kwargs)
        else:
            return primitive.bind(*args, **kwargs)
    
    return type(jaxpr)(reinterpret(jaxpr, eqn_evaluator))
    