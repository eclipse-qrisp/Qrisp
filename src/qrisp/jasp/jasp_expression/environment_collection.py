"""
********************************************************************************
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
********************************************************************************
"""

from functools import lru_cache

import numpy as np
from jax.core import ClosedJaxpr, JaxprEqn, Literal


@lru_cache(maxsize=int(1e5))
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
    eqn_var_tracker = VarTracker(eqn_list)
    new_eqn_list = []
    new_eqn_var_tracker = VarTracker([])
    
    from qrisp.jasp import Jaspr

    if isinstance(jaxpr, Jaspr) and jaxpr.envs_flattened:
        return jaxpr
    
    for j in range(len(eqn_list)):
        eqn = eqn_list[j]

        if eqn.primitive.name == "pjit":

            new_params = dict(eqn.params)

            collected_jaspr = collect_environments(eqn.params["jaxpr"].jaxpr)

            new_params["jaxpr"] = ClosedJaxpr(
                collected_jaspr, eqn.params["jaxpr"].consts
            )

            eqn = JaxprEqn(
                params=new_params,
                primitive=eqn.primitive,
                invars=list(eqn.invars),
                outvars=list(eqn.outvars),
                effects=eqn.effects,
                source_info=eqn.source_info,
            )

        if eqn.primitive.name == "cond":

            new_params = dict(eqn.params)

            branch_list = []

            for i in range(len(eqn.params["branches"])):
                collected_branch_jaxpr = collect_environments(
                    eqn.params["branches"][i].jaxpr
                )
                collected_branch_jaxpr = ClosedJaxpr(
                    collected_branch_jaxpr, eqn.params["branches"][i].consts
                )
                branch_list.append(collected_branch_jaxpr)

            new_params["branches"] = tuple(branch_list)

            eqn = JaxprEqn(
                params=new_params,
                primitive=eqn.primitive,
                invars=list(eqn.invars),
                outvars=list(eqn.outvars),
                effects=eqn.effects,
                source_info=eqn.source_info,
            )

        if eqn.primitive.name == "while":

            new_params = dict(eqn.params)

            body_collected_jaspr = collect_environments(eqn.params["body_jaxpr"].jaxpr)

            new_params["body_jaxpr"] = ClosedJaxpr(
                body_collected_jaspr, eqn.params["body_jaxpr"].consts
            )

            eqn = JaxprEqn(
                params=new_params,
                primitive=eqn.primitive,
                invars=list(eqn.invars),
                outvars=list(eqn.outvars),
                effects=eqn.effects,
                source_info=eqn.source_info,
            )

        # If an exit primitive is found, start the collecting mechanism.
        if eqn.primitive.name == "jasp.q_env" and "exit" in eqn.params.values():

            # Find the position of the enter primitive.
            for i in range(len(new_eqn_list))[::-1]:
                enter_eq = new_eqn_list[i]
                if (
                    enter_eq.primitive.name == "jasp.q_env"
                    and "enter" in enter_eq.params.values()
                ):
                    break
            else:
                raise

            # Set an alias for the equations marked as the body
            environment_body_eqn_list = new_eqn_list[i + 1 :]
            environment_body_var_tracker = new_eqn_var_tracker.slice_start(i+1)
            
            invars = environment_body_var_tracker.find_invars()
            
            # Remove the AbstractQuantumCircuit variable and prepend it.
            # invars = find_invars(environment_body_eqn_list)
            
            try:
                invars.remove(enter_eq.outvars[0])
            except ValueError:
                pass

            # Same for the outvars
            outvars = fast_find_outvars(
                environment_body_eqn_list,
                eqn_var_tracker.slice_start(j+1),
                [var for var in jaxpr.outvars if not isinstance(var, Literal)],
            )

            # Create the Jaxpr
            environment_body_jaspr = Jaspr(
                constvars=[],
                invars=invars + enter_eq.outvars,
                outvars=outvars + eqn.invars[-1:],
                eqns=environment_body_eqn_list,
            )

            # Create the Equation
            eqn = JaxprEqn(
                params={"type": eqn.params["type"], "jaspr": environment_body_jaspr},
                primitive=eqn.primitive,
                invars=enter_eq.invars[:-1] + invars + enter_eq.invars[-1:],
                outvars=outvars + eqn.outvars[-1:],
                effects=eqn.effects,
                source_info=eqn.source_info,
            )

            # Remove the collected equations from the new_eqn_list
            new_eqn_list = new_eqn_list[:i]
            new_eqn_var_tracker = new_eqn_var_tracker.slice_end(i)
            

        # Append the equation
        new_eqn_list.append(eqn)
        new_eqn_var_tracker.append(eqn)
        
    if isinstance(jaxpr, Jaspr):
        res = jaxpr.update_eqns(new_eqn_list)
        if jaxpr.ctrl_jaspr is not None:
            res.ctrl_jaspr = jaxpr.ctrl_jaspr
        if jaxpr.inv_jaspr is not None:
            res.inv_jaspr = jaxpr.inv_jaspr
        return res
    else:
        # Return the transformed equation
        return type(jaxpr)(
            constvars=jaxpr.constvars,
            invars=jaxpr.invars,
            outvars=jaxpr.outvars,
            eqns=new_eqn_list,
        )



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


def find_invar_kernel(invar_indices, outvar_indices):
    
    max_invar = np.max(invar_indices)
    max_outvar = np.max(outvar_indices)
    max_var = max(max_invar, max_outvar)
    
    
    if max_var == -1:
        return np.zeros(0, dtype = np.int64)
    
    invar_array = np.zeros(max_var+1, dtype = np.int8)
    invar_array[invar_indices] = 1
    invar_array[outvar_indices] = 0
    res = np.nonzero(invar_array)[0]
    
    return res            

from numba import njit

jitted_find_invar_kernel = njit(find_invar_kernel)


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


def fast_find_outvars(body_eqn_list, script_remainder_var_tracker, return_vars):
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
    required_remainder_vars = script_remainder_var_tracker.find_invars()

    # The result is the intersection between both sets of variables
    return list(set(outvars).intersection(required_remainder_vars + return_vars))


class VarTracker:
    
    def __init__(self, eqn_list):
        
        var_to_int_dic = {}
        int_to_var_dic = {}
        
        eqn_invar_list = []
        eqn_outvar_list = []
        
        outvar_eqn_index_tracker = [0]
        invar_eqn_index_tracker = [0]
        
        for eqn in eqn_list:
            invar_integers = []
            for var in eqn.invars:
                try:
                    invar_integers.append(var_to_int_dic[var])
                except KeyError:
                    var_to_int_dic[var] = len(var_to_int_dic)
                    int_to_var_dic[var_to_int_dic[var]] = var
                    invar_integers.append(var_to_int_dic[var])
                except TypeError:
                    continue
            outvar_integers = []
            for var in eqn.outvars:
                try:
                    outvar_integers.append(var_to_int_dic[var])
                except KeyError:
                    var_to_int_dic[var] = len(var_to_int_dic)
                    int_to_var_dic[var_to_int_dic[var]] = var
                    outvar_integers.append(var_to_int_dic[var])
                except TypeError:
                    continue
            
            
            eqn_invar_list.extend(invar_integers)
            invar_eqn_index_tracker.append(len(eqn_invar_list))
            
            eqn_outvar_list.extend(outvar_integers)
            outvar_eqn_index_tracker.append(len(eqn_outvar_list))
            
        
        self.var_to_int_dic = var_to_int_dic
        self.int_to_var_dic = int_to_var_dic
        
        self.eqn_invar_list = eqn_invar_list
        self.eqn_outvar_list = eqn_outvar_list
        
        self.invar_eqn_index_tracker = invar_eqn_index_tracker
        self.outvar_eqn_index_tracker = outvar_eqn_index_tracker
        

    def append(self, eqn):
        
        invar_integers = []
        for var in eqn.invars:
            try:
                invar_integers.append(self.var_to_int_dic[var])
            except KeyError:
                self.var_to_int_dic[var] = len(self.var_to_int_dic)
                self.int_to_var_dic[self.var_to_int_dic[var]] = var
                invar_integers.append(self.var_to_int_dic[var])
            except TypeError:
                continue
        outvar_integers = []
        for var in eqn.outvars:
            try:
                outvar_integers.append(self.var_to_int_dic[var])
            except KeyError:
                self.var_to_int_dic[var] = len(self.var_to_int_dic)
                self.int_to_var_dic[self.var_to_int_dic[var]] = var
                outvar_integers.append(self.var_to_int_dic[var])
            except TypeError:
                continue
    
        
        self.eqn_invar_list.extend(invar_integers)
        self.invar_eqn_index_tracker.append(len(self.eqn_invar_list))
        
        self.eqn_outvar_list.extend(outvar_integers)
        self.outvar_eqn_index_tracker.append(len(self.eqn_outvar_list))
        
        
    def slice_start(self, starting_point):
        
        res = VarTracker([])
        
        invar_starting_point = self.invar_eqn_index_tracker[starting_point]
        res.eqn_invar_list = self.eqn_invar_list[invar_starting_point:]
        res.invar_eqn_index_tracker = [i - invar_starting_point for i in self.invar_eqn_index_tracker[starting_point:]]
        
        outvar_starting_point = self.outvar_eqn_index_tracker[starting_point]
        res.eqn_outvar_list = self.eqn_outvar_list[outvar_starting_point:]
        res.outvar_eqn_index_tracker = [i - outvar_starting_point for i in self.outvar_eqn_index_tracker[starting_point:]]
        
        res.int_to_var_dic = self.int_to_var_dic
        res.var_to_int_dic = self.var_to_int_dic
        
        return res
    
    def slice_end(self, end_point):
        
        res = VarTracker([])
        
        invar_end_point = self.invar_eqn_index_tracker[end_point]
        res.eqn_invar_list = self.eqn_invar_list[:invar_end_point]
        res.invar_eqn_index_tracker = self.invar_eqn_index_tracker[:end_point+1]
        
        outvar_end_point = self.outvar_eqn_index_tracker[end_point]
        res.eqn_outvar_list = self.eqn_outvar_list[:outvar_end_point]
        res.outvar_eqn_index_tracker = self.outvar_eqn_index_tracker[:end_point+1]
        
        res.int_to_var_dic = self.int_to_var_dic
        res.var_to_int_dic = self.var_to_int_dic
        
        return res
    
    def find_invars(self):
        
        if len(self.eqn_invar_list) < 20 or len(self.eqn_outvar_list) < 20:
            invar_index_list = find_invar_kernel([-1] + self.eqn_invar_list, [-1] +self.eqn_outvar_list)
        else:
            invar_index_list = jitted_find_invar_kernel(np.array([-1] + self.eqn_invar_list, dtype = np.int32), 
                                                        np.array([-1] +self.eqn_outvar_list, dtype = np.int32))
            
        
        res = []
        
        
        
        for i in range(len(invar_index_list)):
            res.append(self.int_to_var_dic[invar_index_list[i]])
        
        sorting_dic = {self.eqn_invar_list[i] : i for i in range(len(self.eqn_invar_list))[::-1]}
        
        res.sort(key = lambda x : sorting_dic[self.var_to_int_dic[x]])
        
        return res
        
        
        
        
        
        
