
from qrisp import rz, rzz,rx, mcx, x, QuantumVariable, control
import numpy as np
import copy
from qrisp.qiro.qiroproblems.qiro_utils import * 
from qrisp.qaoa.problems.maxSetPackInfrastr import get_neighbourhood_relations
import networkx as nx


def trafo_maxPackToMIS(problem):
    """
    Function to transform a Maximum Set Packing problem into a Maximum independent set problem.

    Parameters
    ----------
    problem : List
        The problem definition in analogy to the QAOA maxSetPacking problem

    Returns
    -------
    G : nx.Graph
        The corresponding graph to be solved by an MIS implementation.


    """
    # the MIS solution to G is equivalent to the solution of the maxSetPackingProblem
    G = nx.Graph()
    G.add_nodes_from(list(range(len(problem[1]))))
    n_dict = get_neighbourhood_relations(problem)
    
    for val in n_dict.values():
        
        if len(val)<2:
            continue
        for index1 in range(len(val)-1):
            for index2 in range(index1+1, len(val)):
                G.add_edge(val[index1],val[index2])

    return G 










"""
does all of the set popping actually work? Since the amount of sets is equivalent to len of qarg?
--> setting to empty also doesnt work, since that implies that that they are always in

    # what to do about sols and exclusions in cost_op

where does the neighbourhood_dict fit it?

need to write cost function?


"""

""" def create_maxSetPack_replacement_routine( res, old_n_dict, problem, solutions, exclusions):

    sets ,universe = copy.deepcopy(problem[0]), problem[1]

    n_dict = copy.deepcopy(old_n_dict)

    # FOR SINGLE QUBIT CORRELATIONS
    orig_nodes = list(universe)
    combinations = []

    # add check for solution nodes      
    for index1 in range(len(universe)):
        for index2 in range(index1,len(universe)):
            combinations.append((index1,index2))

    # you are HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    max = 0
    max_item = []
    #get the max_edge and eval the sum and sign


    max_item, sign = find_max(orig_nodes, combinations, res, solutions)
    
    # we just directly remove vertices from the graph 
    if isinstance(max_item, int):
        if sign > 0:
            the_set = sets[max_item]
            for node in the_set:
                rels = n_dict[node]
                for rel in rels:
                    sets[rels] = [] 
            # add set to sol
            # go through n_dict and remove all other sets that contain elements in this set
            # what does remove mean in this context
            solutions.append(max_item)
            exclusions.append(max_item)

        elif sign < 0:
            # remove set from sets?
            sets[rels] = [] 
            exclusions.append(max_item)

    
    else:
        if sign > 0:
            
            
            # both are in: all elements can be deleted for other sets 
            # both are out: ??
            # --> add whatever is in second set to first set, and then dont consider the second set anymore?
            # keep the correlation somehow?? --> this is now in the cost_op
            # or work on the neighbourhood_dict?
            sets[0] += sets[1]
            sets[1] = []
            solutions.append(max_item)

            pass

        elif sign < 0:
            # remove all sets that contain elements of the schnittmenge, because one of the two is going to be in there
            # how to fix the negative correlation??
            intersect = list(sets[max_item[0]] & sets[max_item[1]])
            for item in intersect:
                rels = n_dict[item]
                for rel in rels: 
                    if not rel in max_item:
                        sets[rels] = [] 
            exclusions.append(rel)
            


    return sets, solutions, sign, exclusions

# what about other operatores??



def maxSetPackCost_reduced(problem, solutions = []):
    
    |  Create the cost/problem operator for this problem instance. The swapping rule is to swap a set in and out of the solution, if it is not intersecting with any other set.
    |  Idea - Per set: 

    * Create ancillas for every element, they represent these elements
    * Perform multi controlled x operations on each ancilla
    * Controls are given by sets with also contain the considered element
    * If all controls are "0" (see ``ctrl_state`` for ``mcx``-operation) we set this ancilla to "1"

    |  Then perform mcx on the qubit describing the set as follows:
    |  If all ancillas are "1" this means the qubits describing the sets contain no intersections with the considered set. We can then swap the set in (or out).
    |  Afterwards uncompute the ancillas.

    Parameters
    ----------
    sets : list(Lists)
        The sets the universe is seperated into as by the problem definition

    universe: Tuple
        The universe for the problem instance, i.e. all possible values (all graph vertices)

    Returns
    -------
    QuantumCircuit: qrisp.QuantumCircuit
        the Operator applied to the circuit-QuantumVariable



    universe, sets = list(range(problem[0])), problem[1]

    # get neigbhourhood relations from helper function
    nbh_rel = get_neighbourhood_relations(problem)

    # what to do about sols and exclusions
    def theCostOpEmbedded(qv, gamma):
        #check all sets
        for set_index in range(len(sets)):
            # get set elements and create an ancilla for every set element
            nodes = sets[set_index]
            ancillas = QuantumVariable(len(nodes))
            # go through all ancillas and, equivalently set elements
            for ancilla_index in range(len(ancillas)):
                # if the element is only in one set, we can set this ancilla to 1
                if len(nbh_rel[nodes[ancilla_index]])<2:
                    x(ancillas[ancilla_index])
                    continue
                # else save the sets with also contain the considered element
                nbh_sets_list = [ item for item in nbh_rel[nodes[ancilla_index]] if item != set_index]
                # perform mcx on ancilla, control given by the relevant set
                mcx([qv[nbh_sets_index] for nbh_sets_index in nbh_sets_list], ancillas[ancilla_index], ctrl_state= "0" * len(nbh_sets_list))
            # perform mcrx gate on the qubit describing the considered set
            with control(ancillas):
                rx(gamma, qv[set_index])  
            
            for item in solutions:
                if not isinstance(item, int):
                    cx(item[0],item[1])
            #mcrx_gate = RXGate(gamma).control(len(ancillas))
            #qv.qs.append(  mcrx_gate, [*ancillas, qv[set_index]])

            ancillas.uncompute()

    return theCostOpEmbedded

 """