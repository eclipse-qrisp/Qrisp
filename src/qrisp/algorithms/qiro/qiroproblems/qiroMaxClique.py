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
import networkx as nx
from qrisp.algorithms.qiro.qiroproblems.qiro_utils import * 


def create_max_clique_replacement_routine(res, problem_updated):
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

    orig_edges = [list(item) for item in graph.edges()]
    orig_nodes = list(graph.nodes())

    #get the max_edge and eval the sum and sign
    
    max_item, sign = find_max(orig_nodes, orig_edges , res, solutions)
    if max_item == None:
        return graph, solutions, 0 ,exclusions

    new_graph = copy.deepcopy(graph)

    # we just directly remove vertices from the graph 
    if isinstance(max_item, int):
        if sign < 0:
            border = list(graph.adj[max_item].keys())
            border.append(max_item)
            to_remove = [int(item) for item in graph.nodes() if item not in border]
            new_graph.remove_nodes_from( to_remove)
            solutions.append(max_item)
            exclusions += to_remove

        elif sign > 0:
            #remove item
            new_graph.remove_node(max_item)
            exclusions.append(max_item)

    else:
        if sign > 0:
            #keep the two items in solution and remove all that are not adjacent to both
            
            intersect = list(set(list(graph.adj[max_item[0]].keys())) & set(list(graph.adj[max_item[0]].keys())))
            intersect.append(max_item[0])
            intersect.append(max_item[1])
            to_remove = [int(item) for item in graph.nodes() if item not in intersect]
            new_graph.remove_nodes_from([item for item in graph.nodes() if item not in intersect])
            solutions.append(max_item[0])
            solutions.append(max_item[1])
            exclusions += to_remove

        elif sign < 0:
            #remove all that do not border on either! node
            union = list(graph.adj[max_item[0]].keys())
            union += list(graph.adj[max_item[1]].keys())
            union.append(max_item[0])
            union.append(max_item[1])
            to_remove = [int(item) for item in graph.nodes() if item not in union]
            #to_delete = [item for item in graph.nodes() if item not in union]
            new_graph.remove_nodes_from(to_remove)
            exclusions += to_remove

    return new_graph, solutions, sign, exclusions


def create_max_clique_cost_operator_reduced(problem_updated):
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

    G_compl = nx.complement(problem)
    def cost_operator(qv, gamma):
        for pair in list(G_compl.edges()):
            rzz(3*gamma, qv[pair[0]], qv[pair[1]])
            rz(-gamma, qv[pair[0]])
            rz(-gamma, qv[pair[1]])
        for i in problem.nodes():
            if not i in solutions:
                rz(gamma, qv[i])

    return cost_operator

