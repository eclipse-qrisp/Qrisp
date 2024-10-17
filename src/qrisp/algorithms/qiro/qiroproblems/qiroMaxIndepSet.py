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
from qrisp.algorithms.qiro.qiroproblems.qiro_utils import * 


def create_maxIndep_replacement_routine(res, graph, solutions=[], exclusions=[]):
    """
    Creates a replacement routine for the problem structure, i.e., defines the replacement rules. 
    See the `original paper <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.020327>`_ for a description of the update rules.

    Parameters
    ----------
    res : dict
        Result dictionary of initial QAOA optimization procedure.
    graph : nx.Graph
        The graph defining the problem instance.
    solutions : list
        Qubits which were found to be positively correlated, i.e., part of the problem solution.
    exclusions : list
        Qubits which were found to be negatively correlated, i.e., not part of the problem solution, or contradict solution qubits in accordance with the update rules.  

    Returns
    -------
    newgraph : nx.Graph
        Updated graph for the problem instance.
    solutions : list
        Updated set of solutions to the problem.
    sign : int
        The sign of the correlation.
    exclusions : list
        Updated set of exclusions for the problem.
        
    """

    # for multi qubit correlations
    orig_edges = [list(item) for item in graph.edges()]

    # for single qubit correlations
    orig_nodes = list(graph.nodes())
    
    max_item = []
    max_item, sign = find_max(orig_nodes, orig_edges , res, solutions)

    # create a copy of the graph
    newgraph = copy.deepcopy(graph)

    # we remove nodes from the graph, as suggested by the replacement rules
    # if the item is an int, it is a single node, else it is an edge
    if isinstance(max_item, int):
        if sign > 0:
            # remove all adjacent nodes
            to_remove = graph.adj[max_item]
            newgraph.remove_nodes_from(to_remove)
            solutions.append(max_item)
            exclusions += to_remove

        elif sign < 0:
            # remove the node
            newgraph.remove_node(max_item)
            exclusions.append(max_item)

    else:
        if sign > 0:
            # remove both nodes
            newgraph.remove_nodes_from(max_item)
            exclusions += list(max_item)

        elif sign < 0:
            # remove all nodes connected to both nodes 
            intersect = list(set( list(graph.adj[max_item[0]].keys()) ) & set( list(graph.adj[max_item[0]].keys()) ))
            newgraph.remove_nodes_from(intersect)
            exclusions += intersect 

    return newgraph, solutions, sign, exclusions


def create_maxIndep_cost_operator_reduced(graph, solutions=[]):
    r"""
    Creates the ``cost_operator`` for the problem instance.
    This operator is adjusted to consider qubits that were found to be a part of the problem solution.

    Parameters
    ----------
    G : nx.Graph
        The graph for the problem instance.
    solutions : list
        Qubits which were found to be positively correlated, i.e., part of the problem solution.
    
    Returns
    -------
    cost_operator : function
        A function receiving a :ref:`QuantumVariable` and a real parameter $\gamma$. This function performs the application of the cost operator.

    """
    def cost_operator(qv, gamma):
        for pair in list(graph.edges()):
            #cx(qv[pair[0]], qv[pair[1]])
            rzz(3*gamma, qv[pair[0]], qv[pair[1]])
            rz(-gamma, qv[pair[0]])
            rz(-gamma, qv[pair[1]])
        for i in graph.nodes():
            if not i in solutions:
                rz(gamma, qv[i])

    return cost_operator



""" def create_maxIndep_mixer_reduced(graph, solutions):

    def RX_mixer(qv, beta):

        from qrisp import rx
        for i in graph.nodes():
            if not i in solutions:
                rx(2 * beta, qv[i])
    return RX_mixer


def init_function_reduced(graph, solutions):

    def init_state(qv):
        from qrisp import h
        for i in graph.nodes():
            if not i in solutions:
                h(qv[i])
        for i in solutions:
            x(qv[i])
    return init_state



#TODO:
def create_maxIndep_cl_cost_function_reduced(graph):

        #btw alternative formulation: for edge: check if string[edge[0]] != string[edge[1]] 
    def aClcostFct(res_dic):
        tot_energy = 0.001
        tot_counts = 0
        for state in res_dic.keys():
            # we assume solution is right
            temp = True
            energy = 0 
            for edge in graph.edges():
                if not state[edge[0]] != state[edge[1]]:
                    temp = False 
            
            # else we just add the number of marked as |1> nodes
            if temp: 
                intlist = [s for s in range(len(list(state))) if list(state)[s] == "1"]
                energy = -len(intlist)
            
            tot_energy += energy * res_dic[state]
            tot_counts += res_dic[state]

        #print(tot_energy/tot_counts)

        return tot_energy/tot_counts

    return aClcostFct 
"""