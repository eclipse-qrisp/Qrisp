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

from qrisp import *
import numpy as np
from scipy.optimize import minimize
from sympy import Symbol
import itertools
from numba import njit, prange

@njit(cache = True)
def maxcut_obj(x, edge_list):
    cut = 0
    for i, j in edge_list:
        # the edge is cut
        if ((x >> i) ^ (x >>j)) & 1:
        # if x[i] != x[j]:                          
            cut -= 1
    return cut

@njit(parallel = True, cache = True)
def maxcut_energy(outcome_array, count_array, edge_list):
    
    res_array = np.zeros(len(outcome_array))    
    for i in prange(len(outcome_array)):
        res_array[i] = maxcut_obj(outcome_array[i], edge_list)*count_array[i]
        
    return np.sum(res_array)


def create_maxcut_cl_cost_function(G):
    """
    Creates the classical cost function for maxcut for the specific graph G that we are attempting to cut.

    Parameters
    ----------
    G : nx.Graph
        Graph to color.

    Returns
    -------
    cl_cost_function : function
        Classical cost function, which in the end returns the ratio between 
        the energy calculated using the maxcut_obj objective funcion and the 
        amount of counts used in the experiment.

    """    
    def cl_cost_function(counts):
        
        edge_list = np.array(list(G.edges()), dtype = np.uint32)
        
        counts_keys = list(counts.keys())
        
        int_list = []
        if not isinstance(counts_keys[0], str):
            
            for c_array in counts_keys:
                integer = int("".join([c for c in c_array]), 2)
                int_list.append(integer)
        else:
            for c_str in counts_keys:
                integer = int(c_str, 2)
                int_list.append(integer)
            
        counts_array = np.array(list(counts.values()))
        outcome_array = np.array(int_list, dtype = np.uint32)
        
        return maxcut_energy(outcome_array, counts_array, edge_list)
    
    return cl_cost_function


def create_maxcut_cost_operator(G):
    """
    Creates the maxcut operator as a sequence of unitary gates. In the QAOA overview section this is also called the phase separator U_P.

    Parameters
    ----------
    G : nx.Graph
        Graph to color.

    Returns
    -------
    cost_operator function.

    """
    def maxcut_cost_operator(qv, gamma):
        
        if len(G) != len(qv):
            raise Exception(f"Tried to call MaxCut cost Operator for graph of size {len(G)} on argument of invalid size {len(qv)}")
        
        for pair in list(G.edges()):
            rzz(2*gamma, qv[pair[0]], qv[pair[1]])
            # cx(qv[pair[0]], qv[pair[1]])
            # rz(2 * gamma, qv[pair[1]])
            # cx(qv[pair[0]], qv[pair[1]])
            # barrier(qv)
        
    return maxcut_cost_operator


def maxcut_problem(G):
    """
    Creates a QAOA problem instance taking the phase separator, appropriate mixer, and
    appropriate classical cost function into account.

    Parameters
    ----------
    G : nx.Graph
        Graph to color.

    Returns
    -------
    QAOAProblem : function
        QAOA problem instance for maxcut with which the QAOA algorithm is ran for.

    """        
    from qrisp.qaoa import QAOAProblem, RX_mixer
    
    return QAOAProblem(create_maxcut_cost_operator(G), RX_mixer, create_maxcut_cl_cost_function(G))