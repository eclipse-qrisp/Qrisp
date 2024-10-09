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
from qrisp.jasp.interpreter_tools import eval_jaxpr, extract_invalues, insert_outvalues, exec_eqn

def evaluate_cond_eqn(cond_eqn, context_dic, eqn_evaluator = exec_eqn):
    
    # Extract the invalues from the context dic
    invalues = extract_invalues(cond_eqn, context_dic)
    
    if bool(invalues[0]):
        res = eval_jaxpr(cond_eqn.params["branches"][1], eqn_evaluator = eqn_evaluator)(*invalues[1:])
    else:
        res = eval_jaxpr(cond_eqn.params["branches"][0], eqn_evaluator = eqn_evaluator)(*invalues[1:])
    
    if not isinstance(res, tuple):
        res = (res,)
        
    insert_outvalues(cond_eqn, context_dic, res)
    
    
def evaluate_while_loop(while_loop_eqn, context_dic, eqn_evaluator = exec_eqn):
    
    num_const_cond_args = while_loop_eqn.params["body_nconsts"]
    num_const_body_args = while_loop_eqn.params["body_nconsts"]
    
    def break_condition(invalues):
        non_const_values = invalues[num_const_cond_args:]
        return eval_jaxpr(while_loop_eqn.params["cond_jaxpr"], eqn_evaluator = eqn_evaluator)(*non_const_values)
    
    # Extract the invalues from the context dic
    invalues = extract_invalues(while_loop_eqn, context_dic)
    outvalues = invalues[num_const_body_args:]
    
    while break_condition(invalues):
        outvalues = eval_jaxpr(while_loop_eqn.params["body_jaxpr"], eqn_evaluator = eqn_evaluator)(*invalues)
        
        # Update the non-const invalues
        invalues[num_const_body_args:] = outvalues
    
    insert_outvalues(while_loop_eqn, context_dic, outvalues)