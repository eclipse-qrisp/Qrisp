"""
\********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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

from qrisp import rz, rzz, x
import numpy as np
import copy
from qrisp.algorithms.qiro.qiroproblems.qiro_utils import find_max
from qrisp.algorithms.qiro.qiro_mixers import qiro_controlled_RX_mixer_gen
from qrisp import QuantumBool, mcx
from qrisp.algorithms.qaoa import controlled_RX_mixer_gen


def create_max_indep_replacement_routine(res, problem_updated):
    """
    Creates a replacement routine for the problem structure, i.e., defines the replacement rules. 
    See the `original paper <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.020327>`_ for a description of the update rules.

    Parameters
    ----------
    res : dict
        Result dictionary of QAOA optimization procedure.
    problem_updated : List
        Updates that happened during the QIRO routine. Consits of the updated problem, a list of Qubits which were found to be positively correlated, i.e. part of the problem solution, 
        and a list Qubits which were found to be negatively correlated, i.e. they contradict solution qubits in accordance with the update rules.  

    Returns
    -------
    new_graph : nx.Graph
        Updated graph for the problem instance.
    solutions : list
        Updated set of solutions to the problem.
    sign : int
        The sign of the correlation.
    exclusions : list
        Updated set of exclusions for the problem.
        
    """
    graph = problem_updated[0]
    solutions = problem_updated[1]
    exclusions =  problem_updated[2]

    # for multi qubit correlations
    orig_edges = [list(item) for item in graph.edges()]

    # for single qubit correlations
    orig_nodes = list(graph.nodes())
    
    max_item, sign = find_max(orig_nodes, orig_edges , res, solutions)
    if max_item == None:
        return graph, solutions, 0 ,exclusions

    # create a copy of the graph
    new_graph = copy.copy(graph)

    # we remove nodes from the graph, as suggested by the replacement rules
    # if the item is an int, it is a single node, else it is an edge
    if isinstance(max_item, int):
        # if sign <0 then its mostly appeared as a "1" in the results--> part of solution set
        if sign < 0:
            # remove all adjacent nodes
            to_remove = list(graph.adj[max_item])
            new_graph.remove_nodes_from(to_remove)
            solutions.append(max_item)
            exclusions += to_remove
        # if sign >0 then its mostly appeared as a "0" in the results
        elif sign > 0:
            # remove the node
            new_graph.remove_node(max_item)
            exclusions.append(max_item)

    else:
        if sign > 0:
            # remove both nodes
            new_graph.remove_nodes_from(max_item)
            exclusions += list(max_item)

        elif sign < 0:
            # remove all nodes connected to both nodes 
            intersect = list(set( list(graph.adj[max_item[0]].keys()) ) & set( list(graph.adj[max_item[0]].keys()) ))
            new_graph.remove_nodes_from(intersect)
            exclusions += intersect 

    return new_graph, solutions, sign, exclusions


def create_max_indep_cost_operator_reduced(problem_updated):
    r"""
    Creates the ``cost_operator`` for the problem instance.
    This operator is adjusted to consider qubits that were found to be a part of the problem solution.

    Parameters
    ----------
    problem_updated : List
        Updates that happened during the QIRO routine. Consits of the updated problem, a list of Qubits which were found to be positively correlated, i.e. part of the problem solution, 
        and a list Qubits which were found to be negatively correlated, i.e. they contradict solution qubits in accordance with the update rules.  
    
    Returns
    -------
    cost_operator : function
        A function receiving a :ref:`QuantumVariable` and a real parameter $\gamma$. This function performs the application of the cost operator.

    """

    problem = problem_updated[0]
    solutions = problem_updated[1]

    def cost_operator(qv, gamma):
        for pair in list(problem.edges()):
            #cx(qv[pair[0]], qv[pair[1]])
            rzz(3*gamma, qv[pair[0]], qv[pair[1]])
            rz(-gamma, qv[pair[0]])
            rz(-gamma, qv[pair[1]])
        for i in problem.nodes():
            if not i in solutions:
                rz(gamma, qv[i])

    return cost_operator



def create_max_indep_controlled_mixer_reduced(problem_updated):
    r"""
    Creates the ``controlled_RX_mixer`` for a QIRO instance of the maximal independet set problem for a given graph ``G`` following `Hadfield et al. <https://arxiv.org/abs/1709.03489>`_

    The belonging ``predicate`` function indicates if a set can be swapped into the solution.

    Parameters
    ----------
    problem_updated : List
        Updates that happened during the QIRO routine. Consits of the updated problem, a list of Qubits which were found to be positively correlated, i.e. part of the problem solution, 
        and a list Qubits which were found to be negatively correlated, i.e. they contradict solution qubits in accordance with the update rules.  

    Returns
    -------
    controlled_RX_mixer : function
        A Python function receiving a :ref:`QuantumVariable` and real parameter $\beta$. 
        This function performs the application of the mixer associated to the graph ``G``.

    """

    problem = problem_updated[0]
    solutions = problem_updated[1]
    exclusions =  problem_updated[2]
    neighbors_dict = {node: list(problem.adj[node]) for node in problem.nodes()}

    def qiro_predicate(qv,i):
        qbl = QuantumBool()
        if len(neighbors_dict[i])==0:
            x(qbl)
        else:
            mcx([qv[j] for j in neighbors_dict[i]],qbl,ctrl_state='0'*len(neighbors_dict[i]))
        return qbl

    controlled_RX_mixer=qiro_controlled_RX_mixer_gen(qiro_predicate, solutions+exclusions)

    return controlled_RX_mixer




def qiro_max_indep_set_init_function(solutions =[], exclusions = []):
    r"""
    To be used for the controlled mixer approach of QIRO MIS. Only flips qubits which we found to be a part of the problem soultion.
    
    Parameters
    ----------
    solutions : List
        List of Qubits which were found to be positively correlated, i.e. part of the problem solution
    exclusions : List
        List Qubits which were found to be negatively correlated, i.e. they contradict solution qubits in accordance with the update rules.  
    
    """
    def init_function(qv):
        for i in range(len(qv)):
            if i in solutions:
                x(qv[i])

    return init_function
