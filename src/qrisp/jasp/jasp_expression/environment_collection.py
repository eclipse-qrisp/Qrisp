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

from functools import lru_cache
from jax.core import ClosedJaxpr, JaxprEqn, Literal

@lru_cache(maxsize = int(1E5))
def collect_environments(jaxpr):
    """
    This function turns Jaxpr that contain QuantumEnvironment primitive in enter/exit
    form into the collected form. Collected means that the QuantumEnvironments content
    is represented by a Jaspr.

    Parameters
    ----------
    jaxpr : jax.core.Jaxpr
        The Jaxpr with QuantumEnvironment in enter/exit form.

    Returns
    -------
    jax.core.Jaxpr
        A Jaxpr with QuantumEnvironments in collected form.

    """
    
    # We iterate through the list of equations, appending the equations to
    # the new list containing the processed equations.

    # Once we hit an exit primitive, we collect the Equations between the enter
    # and exit primitive.    
    eqn_list = list(jaxpr.eqns)
    new_eqn_list = []
    
    from qrisp.jasp import Jaspr
    
    if isinstance(jaxpr, Jaspr) and jaxpr.envs_flattened:
        return jaxpr
    
    while len(eqn_list) != 0:
        
        eqn = eqn_list.pop(0)
        
        if eqn.primitive.name == "pjit":
            
            new_params = dict(eqn.params)
            
            collected_jaspr = collect_environments(eqn.params["jaxpr"].jaxpr)
            
            new_params["jaxpr"] = ClosedJaxpr(collected_jaspr,
                                              eqn.params["jaxpr"].consts)
            
            eqn = JaxprEqn(params = new_params,
                                    primitive = eqn.primitive,
                                    invars = list(eqn.invars),
                                    outvars = list(eqn.outvars),
                                    effects = eqn.effects,
                                    source_info = eqn.source_info,)
            
        if eqn.primitive.name == "cond":
            
            new_params = dict(eqn.params)
            
            false_collected_jaspr = collect_environments(eqn.params["branches"][0].jaxpr)
            true_collected_jaspr = collect_environments(eqn.params["branches"][1].jaxpr)
            
            false_collected_jaspr = ClosedJaxpr(false_collected_jaspr, eqn.params["branches"][0].consts)
            true_collected_jaspr = ClosedJaxpr(true_collected_jaspr, eqn.params["branches"][1].consts)
            
            new_params["branches"] = (false_collected_jaspr, true_collected_jaspr)
            
            eqn = JaxprEqn(params = new_params,
                                    primitive = eqn.primitive,
                                    invars = list(eqn.invars),
                                    outvars = list(eqn.outvars),
                                    effects = eqn.effects,
                                    source_info = eqn.source_info,)
            
        if eqn.primitive.name == "while":
            
            new_params = dict(eqn.params)
            
            body_collected_jaspr = collect_environments(eqn.params["body_jaxpr"].jaxpr)
            
            new_params["body_jaxpr"] = ClosedJaxpr(body_collected_jaspr, eqn.params["body_jaxpr"].consts)
            
            eqn = JaxprEqn(params = new_params,
                                    primitive = eqn.primitive,
                                    invars = list(eqn.invars),
                                    outvars = list(eqn.outvars),
                                    effects = eqn.effects,
                                    source_info = eqn.source_info,)

        # If an exit primitive is found, start the collecting mechanism.
        if eqn.primitive.name == "jasp.q_env" and "exit" in eqn.params.values():
            
            # Find the position of the enter primitive.            
            for i in range(len(new_eqn_list))[::-1]:
                enter_eq = new_eqn_list[i]
                if enter_eq.primitive.name == "jasp.q_env" and "enter" in enter_eq.params.values():
                    break
            else:
                raise
            
            # Set an alias for the equations marked as the body
            environment_body_eqn_list = new_eqn_list[i+1:]
            
            # Remove the AbstractQuantumCircuit variable and prepend it.            
            invars = find_invars(environment_body_eqn_list)
            try:
                invars.remove(enter_eq.outvars[0])
            except ValueError:
                pass
            
            # Same for the outvars
            outvars = find_outvars(environment_body_eqn_list, eqn_list, [var for var in jaxpr.outvars if not isinstance(var, Literal)])
            
            # Create the Jaxpr
            environment_body_jaspr = Jaspr(constvars = [],
                                           invars =  invars + enter_eq.outvars,
                                           outvars =  outvars + eqn.invars[-1:],
                                           eqns = environment_body_eqn_list)
            
            # Create the Equation
            eqn = JaxprEqn(
                           params = {"type" : eqn.params["type"], "jaspr" : environment_body_jaspr},
                           primitive = eqn.primitive,
                           invars =  enter_eq.invars[:-1] + invars + enter_eq.invars[-1:],
                           outvars =  outvars + eqn.outvars[-1:],
                           effects = eqn.effects,
                           source_info = eqn.source_info,)
            
            # Remove the collected equations from the new_eqn_list
            new_eqn_list = new_eqn_list[:i]
        
        # Append the equation
        new_eqn_list.append(eqn)
    
    if isinstance(jaxpr, Jaspr):
        res = jaxpr.update_eqns(new_eqn_list)
        if jaxpr.ctrl_jaspr is not None:
            res.ctrl_jaspr = jaxpr.ctrl_jaspr
        return res
    else:
    # Return the transformed equation
        return type(jaxpr)(constvars = jaxpr.constvars, 
                     invars = jaxpr.invars,
                     outvars = jaxpr.outvars,
                     eqns = new_eqn_list)
        
    

def find_invars(eqn_list):
    """
    This function takes a list of equations and infers which variables would
    have been defined previously.

    Parameters
    ----------
    eqn_list : list[JaxprEqn]
        A list containing the equations to check for the invars.

    Returns
    -------
    list[jax.core.Var]
        The list of variables that would have to be defined previously.

    """
    
    # This dictionary keeps track of the variables that have been defined
    # by the given equations
    defined_vars = {}
    
    # This list keeps track of the variables that have not been defined
    invars = []
    
    # Iterate through the list of equations and find the variables that are
    # defined nowhere
    for eqn in eqn_list:
        for var in eqn.invars:
            if isinstance(var, Literal):
                continue
            if var not in defined_vars:
                invars.append(var)
        
        for var in eqn.outvars:
            defined_vars[var] = None
    
    return list(dict.fromkeys(invars))

def find_outvars(body_eqn_list, script_remainder_eqn_list, return_vars):
    """
    This function takes the equations of a function and some "follow-up" 
    instructions and infers which variables need to be returned by the function.

    Parameters
    ----------
    body_eqn_list : list[JaxprEqn]
        A list of equations describing a function.
    script_remainder_eqn_list : list[JaxprEqn]
        A list of equations describing the follow up requirements.

    Returns
    -------
    list[jax.core.Var]
        A list of variables that would have to be returned by the function.

    """
    
    # This list will contain all variables produced by the function
    outvars = []
    
    # Fill the list
    for eqn in body_eqn_list:
        outvars.extend(eqn.outvars)
    
    # Remove the duplicates
    outvars = list(set(outvars))
    
    # Find which variables are required for executing the follow-up
    required_remainder_vars = find_invars(script_remainder_eqn_list)
    
    # The result is the intersection between both sets of variables
    return list(set(outvars).intersection(required_remainder_vars + return_vars))
                