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
from numba import njit
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
    new_eqn_list = []
    
    # An important part of collecting the quantum environments is determining
    # the input output variables. Doing this analysis can be prohibitvely costly
    # if implemented naively. For this reason the VarTracker class implements,
    # which tracks the I/O variables in a specialized data structured that
    # enables an efficient solution to this problem.
    eqn_var_tracker = VarTracker(eqn_list)
    new_eqn_var_tracker = VarTracker(new_eqn_list)
    
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
            
            # Compute the sliced version of the var tracker
            environment_body_var_tracker = new_eqn_var_tracker.slice_start(i+1)
            
            # Compute the invars
            invars = environment_body_var_tracker.find_invars()
            
            # Remove the AbstractQuantumCircuit variable and prepend it.
            try:
                invars.remove(enter_eq.outvars[0])
            except ValueError:
                pass


            remaining_script_var_tracker = eqn_var_tracker.slice_start(j+1)
            
            # Same for the outvars
            outvars = find_outvars(
                environment_body_eqn_list,
                remaining_script_var_tracker,
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


def find_outvars(body_eqn_list, script_remainder_var_tracker, return_vars):
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
    """
    This class is motivated by the task of identifying the inputs and outputs
    of the collected environments. For large Jaxpr, this can be prohibitively
    expensive, which is why this class tracks a specialized data structure, which
    allows an efficient solution of this problem.
    
    To describe how this works in detail, we start by noting that it is beneficial
    to assign every variable (input and outputs) a simply integer.
    
    The translation between this assignment is achieved through the var_to_int_dic
    and the int_to_var_dic.
    
    Given a list of equations, what this class tracks is now a list of integers 
    for both inputs and outputs that describe all the inputs and outputs.
    
    i.e. if we are given the equations
    
    %1 = prim_0 %2 %3 %4
    %5 %6 = prim_1 %1
    
    These lists would look like this:
        
    inputs = [%2, %3, %4, %1]
    outputs = [%1, %5, %6]
    
    We can now compute efficiently which variables are the invars of this list
    of equations by initializing an array with 6 entries, first setting all
    the invar positions to 1, i.e. [1, 1, 1, 1, 0, 0]
    and then setting all the outvar position to 0, i.e. [0, 1, 1, 1, 0, 0].

    This is implemented in the find_invars method.
    
    
    An important feature that is required by the environment collection function
    is to slice the list of equations - afterall the environment body will
    be a slice of the equation list in most cases.
    
    Because of this, the class tracks another list of integers, demarking
    which interval of integers belongs to which equation. This list always
    starts with 0.
    
    For the above example we would have
    
    invar_index_tracker = [0, 3, 4]
    outvar_index_tracker = [0, 1, 3]
    
    Each entry of these lists therefore denotes where the corresponding equation
    starts.
    
    This list can be used to efficiently implement the slicing features.
    """
    
    def __init__(self, eqn_list):
        
        
        # Initialize the translation dics
        var_to_int_dic = {}
        int_to_var_dic = {}
        
        # Initialize the variable lists
        eqn_invar_list = []
        eqn_outvar_list = []
        
        # Initialize the index trackers
        outvar_eqn_index_tracker = [0]
        invar_eqn_index_tracker = [0]
        
        # Fill the variable lists
        for eqn in eqn_list:
            invar_integers = []
            for var in eqn.invars:
                try:
                    invar_integers.append(var_to_int_dic[var])
                # This represents the case that the variable has not been
                # converted yet
                except KeyError:
                    var_to_int_dic[var] = len(var_to_int_dic)
                    int_to_var_dic[var_to_int_dic[var]] = var
                    invar_integers.append(var_to_int_dic[var])
                # This represents the case that the variable is a Literal
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
            
        # Store the attributes
        self.var_to_int_dic = var_to_int_dic
        self.int_to_var_dic = int_to_var_dic
        
        self.eqn_invar_list = eqn_invar_list
        self.eqn_outvar_list = eqn_outvar_list
        
        self.invar_eqn_index_tracker = invar_eqn_index_tracker
        self.outvar_eqn_index_tracker = outvar_eqn_index_tracker
        

    def append(self, eqn):
        """
        Adds the equation eqn to the list of tracked equations.

        Parameters
        ----------
        eqn : jax.core.Equation

        """
        
        # Perform similar logic as in __init__        
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
        """
        If self is represented by VarTracker(eqn_list), the result of this
        function is VarTracker(eqn_list[starting_point:])

        Parameters
        ----------
        starting_point : int
        """
        
        res = VarTracker([])
        
        # Identify where the invar list has to be sliced from
        invar_starting_point = self.invar_eqn_index_tracker[starting_point]
        # Slice the invar list
        res.eqn_invar_list = self.eqn_invar_list[invar_starting_point:]
        # Slice the index tracker and ensure it starts from 0
        res.invar_eqn_index_tracker = [i - invar_starting_point for i in self.invar_eqn_index_tracker[starting_point:]]
        
        # Same for the outvars
        outvar_starting_point = self.outvar_eqn_index_tracker[starting_point]
        res.eqn_outvar_list = self.eqn_outvar_list[outvar_starting_point:]
        res.outvar_eqn_index_tracker = [i - outvar_starting_point for i in self.outvar_eqn_index_tracker[starting_point:]]
        
        res.int_to_var_dic = self.int_to_var_dic
        res.var_to_int_dic = self.var_to_int_dic
        
        return res
    
    def slice_end(self, end_point):
        """
        If self is represented by VarTracker(eqn_list), the result of this
        function is VarTracker(eqn_list[:end_point])

        Parameters
        ----------
        end_point : int
        """
        
        res = VarTracker([])
        
        # Perform similar slicing logic as in slice_start
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
        """
        Computes the undefined invars of the currently tracked equation list,
        i.e. all the variables that are used as invars but not defined by one
        of the equations.

        Returns
        -------
        res : list[Eqn]
        """
        
        # If viable, call the jitted version.
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

jitted_find_invar_kernel = njit(find_invar_kernel)        
        
        
        
