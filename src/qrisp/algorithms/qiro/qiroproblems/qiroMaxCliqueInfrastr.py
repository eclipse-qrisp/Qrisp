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


from qrisp import rz, rzz, x
import numpy as np
import copy
import networkx as nx
from qrisp.algorithms.qiro.qiroproblems.qiro_utils import * 




def create_maxClique_replacement_routine( res, Graph, solutions, exclusions):
    """
    Creates a replacement routine for the problem structure, i.e. defines the replacement rules. 
    See the original paper for description of the update rules

    Parameters
    ----------
    res : dict
        Result dictionary of initial QAOA optimization procedure.
    Graph : nx.Graph
        The Graph defining the problem instance.
    solutions : List
        Qubits which have been found to be positive correlated, i.e. part of the problem solution.
    exclusions : List
        Qubits which have been found to be negative correlated, i.e. not part of the problem solution, or contradict solution qubits in accordance to the update rules.  

    Returns
    -------
    newGraph : nx.Graph
        Updated graph of the problem instance.
    solutions : List
        Updated set of solutions to the problem.
    sign : Int
        The sign of the correlation.
    exclusions : List
        Updated set of exclusions to the problem.
        
    """

    orig_edges = [list(item) for item in Graph.edges()]
    orig_nodes = list(Graph.nodes())

    #get the max_edge and eval the sum and sign
    max_item, sign = find_max(orig_nodes, orig_edges , res, solutions)

    newGraph = copy.deepcopy(Graph)

    # we just directly remove vertices from the graph 
    if isinstance(max_item, int):
        if sign > 0:
            border = list(Graph.adj[max_item].keys())
            border.append(max_item)
            to_remove = [int(item) for item in Graph.nodes() if item not in border]
            newGraph.remove_nodes_from( [item for item in Graph.nodes() if item not in border])
            solutions.append(max_item)
            exclusions += to_remove

        elif sign < 0:
            #remove item
            newGraph.remove_node(max_item)
            exclusions.append(max_item)

    else:
        if sign > 0:
            #keep the two items in solution and remove all that are not adjacent to both
            
            intersect = list(set(list(Graph.adj[max_item[0]].keys())) & set(list(Graph.adj[max_item[0]].keys())))
            intersect.append(max_item[0])
            intersect.append(max_item[1])
            to_remove = [int(item) for item in Graph.nodes() if item not in intersect]
            newGraph.remove_nodes_from([item for item in Graph.nodes() if item not in intersect])
            solutions.append(max_item[0])
            solutions.append(max_item[1])
            exclusions += to_remove

        elif sign < 0:
            #remove all that do not border on either! node
            union = list(Graph.adj[max_item[0]].keys())
            union += list(Graph.adj[max_item[1]].keys())
            union.append(max_item[0])
            union.append(max_item[1])
            to_remove = [int(item) for item in Graph.nodes() if item not in union]
            #to_delete = [item for item in Graph.nodes() if item not in union]
            newGraph.remove_nodes_from([item for item in Graph.nodes() if item not in union])
            exclusions += to_remove

    return newGraph, solutions, sign, exclusions



def create_maxClique_cost_operator_reduced(Graph, solutions = []):
    """
    |  Based on PennyLane unconstrained mixer implementation.
    |  Initial state in :math:`(|0>+|1>)^{\otimes n}` . 
    |  This operator is then adjusted to consider qubits that have been found to be a part of the problem solution.

    Parameters
    ----------
    G : nx.Graph
        Graph of the problem instance
    solutions : List
        Qubits which have been found to be positive correlated, i.e. part of the problem solution.
    
    Returns
    -------
    partialCostMixer : function
        The Operator to be applied to the problem ``QuantumVariable``

    """
    G_compl = nx.complement(Graph)
    def partialcostMixer(qv, gamma):
        for pair in list(G_compl.edges()):
            rzz(3*gamma, qv[pair[0]], qv[pair[1]])
            rz(-gamma, qv[pair[0]])
            rz(-gamma, qv[pair[1]])
        for i in Graph.nodes():
            if not i in solutions:
                rz(gamma, qv[i])


    return partialcostMixer









"""
BM: 18 nodes, seed 99
0.5933101570400762
normal
0.08909071112167886


"""