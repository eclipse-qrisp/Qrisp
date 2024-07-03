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

from jax.core import JaxprEqn, Jaxpr, Literal, ClosedJaxpr
from jax import make_jaxpr

from qrisp.jax.flattening_tools import eval_jaxpr, extract_invalues, eval_jaxpr_with_context_dic, exec_eqn


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
    
    # The idea is to "collect" the QuantumEnvironments first.
    # Collect means that the enter/exit statements are transformed into Jaxpr
    # which are subsequently called. Example:
        
    # from qrisp import *
    # from qrisp.jax import *
    # import jax

    # def outer_function(x):
    #     qv = QuantumVariable(x)
    #     with QuantumEnvironment():
    #         cx(qv[0], qv[1])
    #         h(qv[0])
    #     return qv

    # jaxpr = make_jaxpr(outer_function)(2).jaxpr
    
    # This piece of code results in the following jaxpr
    
    # { lambda ; a:i32[]. let
    #     b:QuantumCircuit = qdef 
    #     c:QuantumCircuit d:QubitArray = create_qubits b a
    #     e:QuantumCircuit = q_env[stage=enter type=quantumenvironment] c
    #     f:Qubit = get_qubit d 0
    #     g:Qubit = get_qubit d 1
    #     h:QuantumCircuit = cx e f g
    #     i:Qubit = get_qubit d 0
    #     j:QuantumCircuit = h h i
    #     _:QuantumCircuit = q_env[stage=exit type=quantumenvironment] j
    #   in (d,) }
    
    # We now apply the collecting mechanism:
    
    jaxpr = collect_environments(jaxpr)
    
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
    
    return flatten_collected_environments(jaxpr)
    

def flatten_collected_environments(jaxpr):
    
    # It is now much easier to apply higher order transformations with this kind
    # of data structure.
    eqn_evaluator_function_dic = {"q_env" : flatten_environment_eqn,
                                  "pjit" : flatten_environments_in_pjit_eqn}
    
    # The flatten_environment_eqn function below executes the collected QuantumEnvironments
    # according to their semantics
    
    # To perform the flattening, we evaluate with the usual tools
    return make_jaxpr(eval_jaxpr(jaxpr, 
                                 eqn_evaluator_function_dic = eqn_evaluator_function_dic))(*[var.aval for var in jaxpr.invars + jaxpr.constvars]).jaxpr
    

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
    
    # Set an alias for the function body
    body_jaxpr = env_eqn.params["jaxpr"]
    
    from qrisp.environments import InversionEnvironment
    from qrisp.jax.environment_compilation import inv_transform
    
    # Perform the environment compilation logic
    if isinstance(env_eqn.primitive, InversionEnvironment):
        transformed_jaxpr = inv_transform(body_jaxpr)
    else:
        transformed_jaxpr = body_jaxpr
    
    # Extract the invalues
    invalues = extract_invalues(env_eqn, context_dic)
    
    # Create a new context_dic
    new_context_dic = {}
    
    # Fill the new context dic with the previously collected invalues
    for i in range(len(transformed_jaxpr.invars)):
        new_context_dic[transformed_jaxpr.invars[i]] = invalues[i]
        
    # Fill the new context dic with the constvalues
    for i in range(len(transformed_jaxpr.constvars)):
        new_context_dic[transformed_jaxpr.constvars[i]] = context_dic[env_eqn.invars[i+len(transformed_jaxpr.invars)]]
    
    # Execute the transformed jaxpr for flattening
    eval_jaxpr_with_context_dic(transformed_jaxpr, new_context_dic, eqn_evaluator_function_dic = {"q_env" : flatten_environment_eqn})
    
    # Insert the outvalues into the context dic
    for i in range(len(env_eqn.outvars)):
        context_dic[env_eqn.outvars[i]] = new_context_dic[transformed_jaxpr.outvars[i]]
    
    
    
def collect_environments(jaxpr):
    """
    This function turns Jaxpr that contain QuantumEnvironment primitive in enter/exit
    form into the collected form. Collected means that the QuantumEnvironments content
    is represented by a Jaxpr.

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
    
    while len(eqn_list) != 0:
        
        eqn = eqn_list.pop(0)
        
        if eqn.primitive.name == "pjit":
            eqn.params["jaxpr"] = ClosedJaxpr(collect_environments(eqn.params["jaxpr"].jaxpr),
                                              eqn.params["jaxpr"].consts)
        
        # If an exit primitive is found, start the collecting mechanism.
        if eqn.primitive.name == "q_env" and eqn.params["stage"] == "exit":
            
            # Find the position of the enter primitive.            
            for i in range(len(new_eqn_list))[::-1]:
                enter_eq = new_eqn_list[i]
                if enter_eq.primitive.name == "q_env" and enter_eq.params["stage"] == "enter":
                    break
            else:
                raise
            
            # Set an alias for the equations marked as the body
            environment_body_eqn_list = new_eqn_list[i+1:]
            
            # To turn these equations into a Jaxpr, we need to figure out which
            # variables are constvars of this Jaxpr. For this we use the helper function
            # defined below.
            constvars = find_invars(environment_body_eqn_list)
            
            # Same for the outvars
            outvars = find_outvars(environment_body_eqn_list, [eqn] + eqn_list)
            
            # We only want to denote the non QuantumCircuit arguments as constvars
            # The QuantumCircuit should be an actual invar
            constvars.remove(enter_eq.outvars[0])
            
            # Create the Jaxpr
            environment_body_jaxpr = Jaxpr(constvars = constvars,
                                           invars = enter_eq.outvars,
                                           outvars = outvars,
                                           eqns = environment_body_eqn_list)
            
            # Create the Equation
            eqn = JaxprEqn(
                           params = {"stage" : "collected", "jaxpr" : environment_body_jaxpr},
                           primitive = eqn.primitive,
                           invars = list(enter_eq.invars) + constvars,
                           outvars = list(eqn.outvars),
                           effects = eqn.effects,
                           source_info = eqn.source_info)
            
            # Remove the collected equations from the new_eqn_list
            new_eqn_list = new_eqn_list[:i]
        
        # Append the equation
        new_eqn_list.append(eqn)
    
    # Return the transformed equation
    return Jaxpr(constvars = jaxpr.constvars, 
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
    
    return list(set(invars))

def find_outvars(body_eqn_list, script_remainder_eqn_list):
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
    return list(set(outvars).intersection(required_remainder_vars))
                
    
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
    
    eqn.params["jaxpr"] = ClosedJaxpr(flatten_collected_environments(eqn.params["jaxpr"].jaxpr),
                                      eqn.params["jaxpr"].consts)
    exec_eqn(eqn, context_dic)
    