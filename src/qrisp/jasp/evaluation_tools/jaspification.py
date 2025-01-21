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

from jax.tree_util import tree_flatten, tree_unflatten

from qrisp.jasp.interpreter_tools import extract_invalues, insert_outvalues, eval_jaxpr
from qrisp.simulator import BufferedQuantumState

def jaspify(func):
    
    from qrisp.jasp import make_jaspr
    
    treedef_container = []
    def tracing_function(*args):
        res = func(*args)
        flattened_values, tree_def = tree_flatten(res)
        treedef_container.append(tree_def)
        return flattened_values
    
    def return_function(*args):
        jaspr = make_jaspr(tracing_function)(*args)
        jaspr_res = jaspr(*args)
        if isinstance(jaspr_res, tuple):
            jaspr_res = tree_unflatten(treedef_container[0], jaspr_res)
        return jaspr_res
    return return_function


def simulate_jaspr(jaspr, *args):
    
    if len(jaspr.outvars) == 1:
        return None
    
    args = [BufferedQuantumState()] + list(tree_flatten(args)[0])
            
    flattened_jaspr = jaspr
    
    def eqn_evaluator(eqn, context_dic):
        if eqn.primitive.name == "pjit":
            
            if eqn.params["name"] == "expectation_value_eval_function":
                from qrisp.jasp.program_control import sampling_evaluator
                sampling_evaluator("ev")(eqn, context_dic, eqn_evaluator = eqn_evaluator)
                return
            
            if eqn.params["name"] == "sampling_eval_function":
                from qrisp.jasp.program_control import sampling_evaluator
                sampling_evaluator("array")(eqn, context_dic, eqn_evaluator = eqn_evaluator)
                return
                
            invalues = extract_invalues(eqn, context_dic)
            outvalues = eval_jaxpr(eqn.params["jaxpr"], eqn_evaluator = eqn_evaluator)(*invalues)
            if not isinstance(outvalues, (list, tuple)):
                outvalues = [outvalues]
            insert_outvalues(eqn, context_dic, outvalues)
        elif eqn.primitive.name == "jasp.quantum_kernel":
            insert_outvalues(eqn, context_dic, BufferedQuantumState())
        else:
            return True
    
    res = eval_jaxpr(flattened_jaspr, eqn_evaluator = eqn_evaluator)(*(args + jaspr.consts))
    
    if len(jaspr.outvars) == 2:
        return res[1]
    else:
        return res[1:]