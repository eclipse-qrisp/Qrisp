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

from jax.core import JaxprEqn, ClosedJaxpr

from qrisp.jasp.interpreter_tools import exec_eqn, reinterpret

    
def copy_jaxpr_eqn(eqn):
    return JaxprEqn(primitive = eqn.primitive,
                    invars = list(eqn.invars),
                    outvars = list(eqn.outvars),
                    params = dict(eqn.params),
                    source_info = eqn.source_info,
                    effects = eqn.effects,)

@lru_cache(maxsize = int(1E5))
def flatten_environments(jaspr):
    """
    This function takes in a jaspr with QuantumEnvironment primitives 
    (```q_env```) and compiles these according to their semantics.

    Parameters
    ----------
    jaspr : Jaspr
        The jaspr to flatten.

    Returns
    -------
    jaspr
        The jaxpr without q_env primitives.

    """
    
    # It is now much easier to apply higher order transformations with this kind
    # of data structure.
    def eqn_evaluator(eqn, context_dic):
        if eqn.primitive.name == "jasp.q_env":
            eqn.primitive.jcompile(eqn, context_dic)
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
    from qrisp.jasp import Jaspr
    # To perform the flattening, we evaluate with the usual tools
    reinterpreted_jaxpr = reinterpret(jaspr, eqn_evaluator)
    
    try:
        return Jaspr(reinterpreted_jaxpr)
    except:
        return reinterpreted_jaxpr
    
    
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
    
    jaxpr = eqn.params["jaxpr"].jaxpr
    
    from qrisp.jasp import Jaspr
    if isinstance(jaxpr, Jaspr):
        jaxpr = jaxpr.flatten_environments()
    
    eqn.params["jaxpr"] = ClosedJaxpr(jaxpr,
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
    eqn.params["body_jaxpr"] = ClosedJaxpr(flatten_environments(eqn.params["body_jaxpr"].jaxpr),
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
    eqn.params["branches"] = (eqn.params["branches"][0], ClosedJaxpr(flatten_environments(eqn.params["branches"][1].jaxpr),
                                      eqn.params["branches"][1].consts))
    
    exec_eqn(eqn, context_dic)

