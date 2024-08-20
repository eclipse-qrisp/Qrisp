from qrisp import rz, rzz, x
import numpy as np
import copy
from qrisp.algorithms.qiro.qiroproblems.qiro_utils import * 

def create_maxIndep_replacement_routine( res, Graph, solutions= [], exclusions= []):
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
    # For multi qubit correlations
    orig_edges = [list(item) for item in Graph.edges()]

    # FOR SINGLE QUBIT CORRELATIONS
    orig_nodes = list(Graph.nodes())
    
    max_item = []
    max_item, sign = find_max(orig_nodes, orig_edges , res, solutions)
    newGraph = copy.deepcopy(Graph)

    # we just directly remove vertices from the graph 
    if isinstance(max_item, int):
        if sign > 0:
            to_remove = Graph.adj[max_item]
            newGraph.remove_nodes_from(to_remove)
            solutions.append(max_item)
            exclusions += to_remove

        elif sign < 0:
            newGraph.remove_node(max_item)
            exclusions.append(max_item)

    else:
        if sign > 0:
            newGraph.remove_nodes_from(max_item)
            exclusions += list(max_item)

        elif sign < 0:
            #remove 
            intersect = list(set( list(Graph.adj[max_item[0]].keys()) ) & set( list(Graph.adj[max_item[0]].keys()) ))
            newGraph.remove_nodes_from(intersect)
            exclusions += intersect 

    return newGraph, solutions, sign, exclusions


def create_maxIndep_cost_operator_reduced(Graph, solutions= []):
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
    def partialcostMixer(qv, gamma):
        for pair in list(Graph.edges()):
            #cx(qv[pair[0]], qv[pair[1]])
            rzz(3*gamma, qv[pair[0]], qv[pair[1]])
            rz(-gamma, qv[pair[0]])
            rz(-gamma, qv[pair[1]])
        for i in Graph.nodes():
            if not i in solutions:
                rz(gamma, qv[i])
        #return qv

    return partialcostMixer



""" def create_maxIndep_mixer_reduced(Graph, solutions):

    def RX_mixer(qv, beta):

        from qrisp import rx
        for i in Graph.nodes():
            if not i in solutions:
                rx(2 * beta, qv[i])
    return RX_mixer


def init_function_reduced(Graph, solutions):

    def init_state(qv):
        from qrisp import h
        for i in Graph.nodes():
            if not i in solutions:
                h(qv[i])
        for i in solutions:
            x(qv[i])
    return init_state



#TODO:
def create_maxIndep_cl_cost_function_reduced(Graph):

        #btw alternative formulation: for edge: check if string[edge[0]] != string[edge[1]] 
    def aClcostFct(res_dic):
        tot_energy = 0.001
        tot_counts = 0
        for state in res_dic.keys():
            # we assume solution is right
            temp = True
            energy = 0 
            for edge in Graph.edges():
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