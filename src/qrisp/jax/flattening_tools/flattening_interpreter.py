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

from jax.core import JaxprEqn, Literal, ClosedJaxpr, Tracer
from jax import jit, make_jaxpr

def eval_jaxpr(jaxpr, 
               return_context_dic = False, 
               eqn_eval_dic = {}):
    """
    Evaluates a Jaxpr using the given context dic to replace variables

    Parameters
    ----------
    jaxpr : jax.core.Jaxpr
        The jaxpr to evaluate.

    Returns
    -------
    None.

    """
    
    if isinstance(jaxpr, ClosedJaxpr):
        jaxpr = jaxpr.jaxpr
    
    def jaxpr_evaluator(*args):
        
        temp_var_list = jaxpr.invars + jaxpr.constvars
        
        if len(temp_var_list) != len(args):
            raise Exception("Tried to evaluate jaxpr with insufficient arguments")
        
        context_dic = {temp_var_list[i] : args[i] for i in range(len(args))}
        
        eval_jaxpr_with_context_dic(jaxpr, context_dic, eqn_eval_dic)
        
        if return_context_dic:
            outvals = [context_dic]
        else:
            outvals = []
        
        for i in range(len(jaxpr.outvars)):
            outvals.append(context_dic[jaxpr.outvars[i]])
        
        if len(outvals) == 1:
            return outvals[0]
        else:
            return tuple(outvals)
    
    return jaxpr_evaluator

def reinterpret(jaxpr, eqn_eval_dic = {}):
    
    if isinstance(jaxpr, ClosedJaxpr):
        inter_jaxpr = jaxpr.jaxpr
    else:
        inter_jaxpr = jaxpr
        
    res = make_jaxpr(eval_jaxpr(inter_jaxpr,
                                eqn_eval_dic = eqn_eval_dic))(*[var.aval for var in jaxpr.invars + jaxpr.constvars]).jaxpr
    
    if isinstance(jaxpr, ClosedJaxpr):
        res = ClosedJaxpr(res, jaxpr.consts)    
    
    return res


def eval_jaxpr_with_context_dic(jaxpr, context_dic, eqn_eval_dic = {}):
    
    from qrisp import QuantumCircuit
    # for eqn in jaxpr.eqns:
        # for outvar in eqn.outvars:
            # context_dic[outvar] = eqn
    
    # Iterate through the equations
    for eqn in jaxpr.eqns:
        
        # Evaluate the primitive
        if eqn.primitive.name in eqn_eval_dic.keys():
            eqn_eval_dic[eqn.primitive.name](eqn, context_dic)
        else:
            exec_eqn(eqn, context_dic)
        
        # Mark any processed QuantumCircuit as "burned"
        # In priniciple the syntax gives these QuantumCircuits a meaning
        # However to avoid constant copying, we act in-place
        # Tracing high-level code should never produce a Jaxpr such that
        # this Exception is called
        for i in range(len(eqn.invars)):
            invar = eqn.invars[i]
            
            if isinstance(invar, Literal):
                continue
            
            val = context_dic[invar]
            
            if isinstance(val, QuantumCircuit):
                context_dic[invar] = "burned_qc"
                continue
            elif isinstance(val, str):
                if val == "burned_qc":
                    raise Exception("Tried to use a consumed QuantumCircuit")
    

def exec_eqn(eqn, context_dic):
    invalues = extract_invalues(eqn, context_dic)
    
    if eqn.primitive.name in ["while", "cond"]:
        for val in invalues:
            if isinstance(val, Tracer):
                break
        else:
            from qrisp.jax import evaluate_cond_eqn, evaluate_while_loop
            
            if eqn.primitive.name == "while":
                evaluate_while_loop(eqn, context_dic)
            else:
                evaluate_cond_eqn(eqn, context_dic)
            
            return
    
    res = eqn.primitive.bind(*invalues, **eqn.params)
    insert_outvalues(eqn, context_dic, res)

    
def extract_invalues(eqn, context_dic):
    invalues = []
    for i in range(len(eqn.invars)):
        invar = eqn.invars[i]
        if isinstance(invar, Literal):
            invalues.append(invar.val)
            continue
        invalues.append(context_dic[invar])
    return invalues

def extract_constvalues(eqn, context_dic):
    constvalues = []
    for i in range(len(eqn.constvars)):
        constvar = eqn.constvars[i]
        if isinstance(constvar, Literal):
            constvalues.append(constvar.val)
            continue
        constvalues.append(context_dic[constvar])
    return constvalues

def insert_outvalues(eqn, context_dic, outvalues):
    if eqn.primitive.multiple_results:
        for i in range(len(eqn.outvars)):
            context_dic[eqn.outvars[i]] = outvalues[i]
    else:
        context_dic[eqn.outvars[0]] = outvalues