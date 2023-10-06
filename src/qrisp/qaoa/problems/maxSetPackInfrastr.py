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

from qrisp import QuantumVariable,  mcx, rz, x, rx 

from qrisp import control
from collections.abc import Iterable
# this is pretty much maxIndependent set, but with a twist
# instead of swapping out singular qubits, you swap out whole predefined sets. 
# this means you apply the mixers to all elements in the sets

# we have a graph of 9 vertices

#######################
## reformulate using @auto_uncompute !!!
## https://www.qrisp.eu/reference/Core/Uncomputation.html




def maxSetPackCostOp(sets, universe):
    
    """
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

    Examples
    --------
    Definition of the sets, given as list of lists. Full universe ``sol`` is given as a tuple
    
    >>> sets = [[0,7,1],[6,5],[2,3],[5,4],[8,7,0],[1]]
    >>> sol = (0,1,2,3,4,5,6,7,8)

    The relations between the sets, i.e. which vertice is in which other sets

    >>> print(get_neighbourhood_relations(sets, len_universe=len(sol)))

    Assign the operators

    >>> cost_fun = maxSetPackclCostfct(sets=sets,universe=sol)
    >>> mixerOp = RZ_Mixer()
    >>> costOp = maxSetPackCostOp(sets=sets, universe=sol)
    """
    
    if not isinstance(sets, Iterable):
        raise Exception("Wrong structure of problem - clauses have to be iterable!")
    for clause in sets:
        if not isinstance(clause, Iterable):
            raise Exception("Wrong structure of problem - each set has to be a tuple!")
        for item in clause:
            if not isinstance(item, int):
                raise Exception("Wrong structure of problem - each literal has to an int!")

    # get neigbhourhood relations from helper function
    nbh_rel = get_neighbourhood_relations(sets, len(universe))


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
            #mcrx_gate = RXGate(gamma).control(len(ancillas))
            #qv.qs.append(  mcrx_gate, [*ancillas, qv[set_index]])

            ancillas.uncompute()

    return theCostOpEmbedded





def get_neighbourhood_relations(sets, len_universe):
    """
    helper function to return a dictionary describing neighbourhood relations in the sets, i.e. for each element in the universe, gives the info in which the element is contained in.


    Parameters
    ----------
    sets : list(Lists)
        The sets the universe is seperated into as by the problem definition

    len_universe: int
        The number of elements in the universe

    Returns
    -------
    neighbourhood relation dictionary :  dict
        |  keys: all universe elements (all graph vertices)
        |  values: per universe element the sets it is contained in

    """
    
    n_dict = {}
    for index_node in range(len_universe):
        adding_list = [index_set for index_set in range(len(sets)) if index_node in sets[index_set]]
        #if len(adding_list)>1: 
        n_dict[index_node] = adding_list
    return n_dict



def maxSetPackclCostfct(sets,universe):

    """
    create the classical cost function for the problem instance

    Parameters
    ----------
    sets : list(Lists)
        The sets the universe is seperated into as by the problem definition
        
    universe: Tuple
        The universe for the problem instance, i.e. all possible values (all graph vertices)

    Returns
    -------
    Costfunction : function
        the classical function for the problem instance, which takes a dictionary of measurement results as input
    """

    def setupaClCostfct(res_dic):
        energy = 0
        total_counts = 0
        for state in list(res_dic.keys()):
            # assume solution is correct
            list_universe = [True]*len(universe)
            temp = True
            obj = 0
            #get all sets marked by the solution
            intlist = [s for s in range(len(list(state))) if list(state)[s] == "1"]
            sol_sets = [sets[index] for index in intlist]
            
            for seto in sol_sets:
                for val in seto:
                    if list_universe[val]:
                        # if the value appears in the sets set this value to false
                        list_universe[val] = False
                    else: 
                        # is the value is False this element appeared in another solution set
                        # the sets then intersect and the solution is wrong
                        temp = False 
                        break
            if temp:
                obj -= len(intlist)
            energy += obj * res_dic[state]
            total_counts += res_dic[state]
        #print(energy/total_counts)

        return energy/total_counts
    
    return setupaClCostfct


def init_state(qv):
    # all in 0
    return qv


