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

from qrisp import *
import numpy as np
import jax.numpy as jnp

from numba import njit, prange
from jax import jit, vmap

def maxcut_obj(x, G):
    return maxcut_obj_jitted(int(x[::-1], 2), list(G.edges()))


@njit(cache = True)
def maxcut_obj_jitted(x, edge_list):
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
        res_array[i] = maxcut_obj_jitted(outcome_array[i], edge_list)*count_array[i]
        
    return np.sum(res_array)


def create_maxcut_cl_cost_function(G):
    """
    Creates the classical cost function for an instance of the maximum cut problem for a given graph ``G``.

    Parameters
    ----------
    G : nx.Graph
        The Graph for the problem instance.

    Returns
    -------
    cl_cost_function : function
        The classical cost function for the problem instance, which takes a dictionary of measurement results as input.

    """    
    def cl_cost_function(counts):
        
        edge_list = np.array(list(G.edges()), dtype = np.uint32)
        
        counts_keys = list(counts.keys())
        
        int_list = []
        if not isinstance(counts_keys[0], str):
            
            for c_array in counts_keys:
                integer = int("".join([c for c in c_array])[::-1], 2)
                int_list.append(integer)
        else:
            for c_str in counts_keys:
                integer = int(c_str[::-1], 2)
                int_list.append(integer)
            
        counts_array = np.array(list(counts.values()))
        outcome_array = np.array(int_list, dtype = np.uint32)
        
        return maxcut_energy(outcome_array, counts_array, edge_list)
    
    return cl_cost_function


@jit
def extract_boolean_digit(integer, digit):
    return (integer >> digit) & 1


def create_cut_computer(G):
    edge_list = jnp.array(G.edges()) 

    @jit
    def cut_computer(x):
        x_uint = jnp.uint32(x)
        bools = extract_boolean_digit(x_uint, edge_list[:, 0]) != extract_boolean_digit(x_uint, edge_list[:, 1])
        cut = jnp.sum(bools)  # Count the number of edges crossing the cut
        return -cut

    return cut_computer


def create_maxcut_sample_array_post_processor(G):
    cut_computer = create_cut_computer(G)

    def post_processor(sample_array):
        cut_values = vmap(cut_computer)(sample_array)  # Use vmap for automatic vectorization
        average_cut = jnp.mean(cut_values)  
        return average_cut

    return post_processor


def create_maxcut_cost_operator(G):
    r"""
    Creates the cost operator for an instance of the maximum cut problem for a given graph ``G``.

    Parameters
    ----------
    G : nx.Graph
        The Graph for the problem instance.

    Returns
    -------
    cost_operator : function
        A function receiving a :ref:`QuantumVariable` and a real parameter $\gamma$.
        This function performs the application of the cost operator.

    """
    def maxcut_cost_operator(qv, gamma):
        
        if not check_for_tracing_mode():
            if len(G) != len(qv):
                raise Exception(f"Tried to call MaxCut cost Operator for graph of size {len(G)} on argument of invalid size {len(qv)}")
        
        for pair in list(G.edges()):
            rzz(2 * gamma, qv[pair[0]], qv[pair[1]])
            # cx(qv[pair[0]], qv[pair[1]])
            # rz(2 * gamma, qv[pair[1]])
            # cx(qv[pair[0]], qv[pair[1]])
            # barrier(qv)
        
    return maxcut_cost_operator


def maxcut_problem(G):
    """
    Creates a QAOA problem instance with appropriate phase separator, mixer, and
    classical cost function.

    Parameters
    ----------
    G : nx.Graph
        The graph for the problem instance.

    Returns
    -------
    :ref:`QAOAProblem`
        A QAOA problem instance for MaxCut for a given graph ``G``.

    """        
    from qrisp.qaoa import QAOAProblem, RX_mixer
    
    return QAOAProblem(create_maxcut_cost_operator(G), RX_mixer, create_maxcut_cl_cost_function(G))