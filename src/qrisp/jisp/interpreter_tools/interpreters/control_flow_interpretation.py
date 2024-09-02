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
from qrisp.jisp.interpreter_tools import eval_jaxpr, extract_invalues, insert_outvalues

def evaluate_cond_eqn(primitive, *invalues, **params):
    
    if bool(invalues[0]):
        res = eval_jaxpr(params["branches"][1])(*invalues[1:])
    else:
        res = eval_jaxpr(params["branches"][0])(*invalues[1:])
    
    return res
    
    
def evaluate_while_loop(primitive, *args, **params):
    
    num_const_cond_args = params["body_nconsts"]
    num_const_body_args = params["body_nconsts"]
    invalues = list(args)
    
    def break_condition(invalues):
        non_const_values = invalues[num_const_cond_args:]
        return eval_jaxpr(params["cond_jaxpr"])(*non_const_values)
    
    while break_condition(invalues):
        outvalues = eval_jaxpr(params["body_jaxpr"])(*invalues)
        
        # Update the non-const invalues
        invalues[num_const_body_args:] = outvalues
    
    return outvalues