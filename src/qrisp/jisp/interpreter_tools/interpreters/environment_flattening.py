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

from functools import lru_cache

from jax.core import JaxprEqn, Jaxpr, Literal, ClosedJaxpr
from jax import make_jaxpr

from qrisp.jisp.interpreter_tools import eval_jaxpr, extract_invalues, eval_jaxpr_with_context_dic, exec_eqn, reinterpret, ContextDict

@lru_cache(maxsize = int(1E5))
def flatten_environments(jaxpr):
    """
    This function takes in a jaxpr containing enter/exit commands of QuantumEnvironments
    and compiles these according to their semantics

    Parameters
    ----------
    jaxpr : jax.core.Jaxpr
        The jaxpr to flatten.

    Returns
    -------
    jaxpr
        The jaxpr without enter/exit commands.

    """

    # { lambda ; a:i32[]. let
    #     b:QuantumCircuit = qdef 
    #     c:QuantumCircuit d:QubitArray = create_qubits b a
    #     _:QuantumCircuit = q_env[
    #       jaxpr={ lambda d:QubitArray; e:QuantumCircuit. let
    #           f:Qubit = get_qubit d 0
    #           g:Qubit = get_qubit d 1
    #           h:QuantumCircuit = cx e f g
    #           i:Qubit = get_qubit d 0
    #           j:QuantumCircuit = h h i
    #         in (j,) }
    #       stage=collected
    #     ] c
    #   in (d,) }
    
    # We see that the QuantumEnvironment no longer is specified by enter/exit
    # statements but a q_env[jaxpr = {...}] call.
    res = flatten_collected_environments(jaxpr)
    
    return res
    

@lru_cache(maxsize = int(1E5))
def flatten_collected_environments(jispr):
    
    # It is now much easier to apply higher order transformations with this kind
    # of data structure.
    def eqn_evaluator(eqn, context_dic):
        if eqn.primitive.name == "jisp.q_env":
            flatten_environment_eqn(eqn, context_dic)
        elif eqn.primitive.name == "pjit":
            flatten_environments_in_pjit_eqn(eqn, context_dic)
        elif eqn.primitive.name == "while":
            flatten_environments_in_while_eqn(eqn, context_dic)
        elif eqn.primitive.name == "cond":
            flatten_environments_in_cond_eqn(eqn, context_dic)
        else:
            return True
    
    # The flatten_environment_eqn function below executes the collected QuantumEnvironments
    # according to their semantics
    from qrisp.jisp import Jispr
    # To perform the flattening, we evaluate with the usual tools
    reinterpreted_jaxpr = reinterpret(jispr, eqn_evaluator)
    try:
        return Jispr(reinterpreted_jaxpr)
    except:
        return reinterpreted_jaxpr
    
    

def flatten_environment_eqn(env_eqn, context_dic):
    """
    Specifies how a collected QuantumEnvironment equation is evaluated.

    Parameters
    ----------
    env_eqn : JaxprEqn
        An equation containing a collected QuantumEnvironment.
    context_dic : dict
        A dictionary translating from variables to values.

    Returns
    -------
    None.

    """
    
    return env_eqn.primitive.jcompile(env_eqn, context_dic)
    
    # Extract the invalues
    invalues = extract_invalues(env_eqn, context_dic)
    
    res = env_eqn.primitive.jcompile(env_eqn.params["jispr"], *invalues)
    # Insert the outvalues into the context dic
    
    if not isinstance(res, tuple):
        res = (res,)
    
    for i in range(len(env_eqn.outvars)):
        context_dic[env_eqn.outvars[i]] = res[i]
    
    return
    
    # Set an alias for the function body
    body_jispr = env_eqn.params["jispr"]
    
    from qrisp.environments import InversionEnvironment, ControlEnvironment
    from qrisp.jisp.environment_compilation import inv_transform

    # Perform the environment compilation logic
    if isinstance(env_eqn.primitive, InversionEnvironment):
        transformed_jaxpr = inv_transform(body_jispr)
    elif isinstance(env_eqn.primitive, ControlEnvironment):
        num_ctrl = len(env_eqn.invars) - len(env_eqn.params["jispr"].invars)
        transformed_jaxpr = body_jispr.control(num_ctrl)
    else:
        
        transformed_jaxpr = body_jispr
    
    # Create a new context_dic
    new_context_dic = ContextDict()
        
    # Fill the new context dic with the previously collected invalues
    for i in range(len(transformed_jaxpr.invars)):
        new_context_dic[transformed_jaxpr.invars[i]] = invalues.pop(0)
    
    # Fill the new context dic with the constvalues
    for i in range(len(transformed_jaxpr.constvars)):
        # The constvars of the jaxpr of the collected environment are given as 
        # the invars of the equation. See the corresponding line in collect_environments.
        new_context_dic[transformed_jaxpr.constvars[i]] = invalues.pop(0)
    
    
    def eqn_evaluator(eqn, context_dic):
        if eqn.primitive.name == "jisp.q_env":
            flatten_environment_eqn(eqn, context_dic)
        else:
            return True
    
    # Execute the transformed jaxpr for flattening
    eval_jaxpr_with_context_dic(transformed_jaxpr, new_context_dic, eqn_evaluator = eqn_evaluator)
    
    # Insert the outvalues into the context dic
    for i in range(len(env_eqn.outvars)):
        context_dic[env_eqn.outvars[i]] = new_context_dic[transformed_jaxpr.outvars[i]]
    
    
def flatten_environments_in_pjit_eqn(eqn, context_dic):
    """
    Flattens environments in a pjit primitive

    Parameters
    ----------
    eqn : jax.core.JaxprEqn
        A pjit equation, with collected environments.
    context_dic : dict
        The context dictionary.

    Returns
    -------
    None.

    """
    
    eqn = copy_jaxpr_eqn(eqn)
    
    eqn.params["jaxpr"] = ClosedJaxpr(flatten_collected_environments(eqn.params["jaxpr"].jaxpr),
                                      eqn.params["jaxpr"].consts)
    
    exec_eqn(eqn, context_dic)
    
def flatten_environments_in_while_eqn(eqn, context_dic):
    """
    Flattens environments in a pjit primitive

    Parameters
    ----------
    eqn : jax.core.JaxprEqn
        A pjit equation, with collected environments.
    context_dic : dict
        The context dictionary.

    Returns
    -------
    None.

    """
    
    eqn = copy_jaxpr_eqn(eqn)
    eqn.params["body_jaxpr"] = ClosedJaxpr(flatten_collected_environments(eqn.params["body_jaxpr"].jaxpr),
                                      eqn.params["body_jaxpr"].consts)
    
    
    exec_eqn(eqn, context_dic)

def flatten_environments_in_cond_eqn(eqn, context_dic):
    """
    Flattens environments in a pjit primitive

    Parameters
    ----------
    eqn : jax.core.JaxprEqn
        A pjit equation, with collected environments.
    context_dic : dict
        The context dictionary.

    Returns
    -------
    None.

    """
    
    eqn = copy_jaxpr_eqn(eqn)
    eqn.params["branches"] = (eqn.params["branches"][0], ClosedJaxpr(flatten_collected_environments(eqn.params["branches"][1].jaxpr),
                                      eqn.params["branches"][1].consts))
    
    exec_eqn(eqn, context_dic)

    
def copy_jaxpr_eqn(eqn):
    return JaxprEqn(primitive = eqn.primitive,
                    invars = list(eqn.invars),
                    outvars = list(eqn.outvars),
                    params = dict(eqn.params),
                    source_info = eqn.source_info,
                    effects = eqn.effects,)
    