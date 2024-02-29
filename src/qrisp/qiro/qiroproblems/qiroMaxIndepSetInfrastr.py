from qrisp import rz, rzz, x
import numpy as np
import copy
from qrisp.qiro.qiroproblems.qiro_utils import * 

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

    Examples:
    ---------

    In the example below we show how to employ the structure of this class for a MIS problem and compare it it a normal QAOA implementation

    ::

        # imports 
        from qrisp.qaoa.qiro_problem import QIROProblem
        from qrisp.qaoa.problems.create_rdm_graph import create_rdm_graph
        from qrisp.qaoa.problems.maxIndepSetInfrastr import maxIndepSetclCostfct, maxIndepSetCostOp
        from qrisp.qaoa.qiroproblems.qiroMaxIndepSetInfrastr import * 
        from qrisp.qaoa.qiro_mixers import qiro_init_function, qiro_RXMixer
        from qrisp import QuantumVariable
        import networkx as nx


        # First we define a graph via the number of nodes and the QuantumVariable arguments
        num_nodes = 13
        G = create_rdm_graph(num_nodes, 0.4, seed =  107)
        qarg = QuantumVariable(G.number_of_nodes())

        # set simulator shots
        mes_kwargs = {
            #below should be 5k
            "shots" : 5000
            }

        # assign the correct new update functions for qiro from above imports
        qiro_instance = QIROProblem(G, 
                                    replacement_routine=create_maxIndep_replacement_routine, 
                                    cost_operator= create_maxIndep_cost_operator_reduced,
                                    mixer= qiro_RXMixer,
                                    cl_cost_function= maxIndepSetclCostfct,
                                    init_function= qiro_init_function
                                    )

        # We run the qiro instance and get the results!
        res_qiro = qiro_instance.run_qiro(qarg=qarg, depth = 3, n_recursions = 2, mes_kwargs = mes_kwargs)
        # and also the final graph, that has been adjusted
        final_Graph = qiro_instance.problem

        # Lets see what the 5 best results are
        print("QIRO 5 best results")
        maxfive = sorted(res_qiro, key=res_qiro.get, reverse=True)[:5]
        costFunc = maxIndepSetclCostfct(G)
        for key, val in res_qiro.items():  
            if key in maxfive:
                
                print(key)
                print(costFunc({key:1}))
                

        # and compare them with the networkx result of the max_clique algorithm, where we might just see a better result than the heuristical NX algorithm!
        print("Networkx solution")
        print(nx.approximation.maximum_independent_set(G))

        
        
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
    Just add normal cost op here

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