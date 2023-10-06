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


from qrisp import QuantumVariable, mcx, rz, x, rx
from qrisp import control 
from _collections_abc import Iterable




def minSetCoverCostOp(sets, universe):
    """
    |  Create the cost/problem operator for this problem instance. Swapping rule is to swap a set in and out of the solution, if all elements are covered by other sets
    |  Idea - Per set: 

    * create ancillas for every element, they represent these elements
    * ancillas are set to "1" by default
    * perform multi controlled x operations on each ancilla
    * controls are given by sets with also contain the considered element
    * if all controls are "0" (see ``ctrl_state`` for ``mcx``-operation) we set this ancilla to "0"
    
    |  Then perform mcx on the qubit describing the set:
    |  If all ancillas are "1" this means that, for all elements in the list, there is already (atleast) one set in the bitstring describing the solution sets, which already contains this element. The set can then be swapped in (or out) of the solution, since all elements are covered by other sets
    |  Afterwards uncompute the ancillas

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

    Assign operators
    >>> cost_fun = minSetCoverclCostfct(sets=sets,universe = sol)
    >>> mixerOp = RZ_Mixer()
    >>> costOp = minSetCoverCostOp(sets=sets, universe=sol)
    """

    if not isinstance(sets, Iterable):
        raise Exception("Wrong structure of problem - clauses have to be iterable!")
    for clause in sets:
        if not isinstance(clause, Iterable):
            raise Exception("Wrong structure of problem - each set has to be a tuple!")
        for item in clause:
            if not isinstance(item, int):
                raise Exception("Wrong structure of problem - each literal has to an int!")

    # get neighbourhood relations from helper function
    nbh_rel = get_neighbourhood_relations(sets, len(universe))

    def theCostOpEmbedded(qv, gamma):
        # go through all sets
        for set_index in range(len(sets)):
            # get set elements and create an ancilla for every set element
            nodes = sets[set_index]
            ancillas = QuantumVariable(len(nodes))
            # go through all ancillas and, equivalently, set elements
            for ancilla_index in range(len(ancillas)):
                # if the considered element is in no other set, we cannot swap this set...
                if len(nbh_rel[nodes[ancilla_index]])<2:
                    #break ?
                    continue
                # set ancilla to one 
                x(ancillas[ancilla_index])
                # get relevant neighbourhood sets for this element
                nbh_sets_list = [item for item in nbh_rel[nodes[ancilla_index]] if item != set_index]
                # perform mcx on the ancilla, controlled by the relevant set qubits
                mcx([qv[nbh_sets_index] for nbh_sets_index in nbh_sets_list], ancillas[ancilla_index], ctrl_state= "0"*len(nbh_sets_list))
            # perform mcx on the set qubit if all elements are covered by other sets, i.e. all ancillas == "1"
            with control(ancillas):
                rx(gamma, qv[set_index])

            # uncompute ancillas
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


def minSetCoverclCostfct(sets, universe):

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
            obj = 0.01
            intlist = [s for s in range(len(list(state))) if list(state)[s] == "1"]
            sol_sets = [sets[index] for index in intlist]
            res = ()
            # get all elements that are contained within the solution sets
            for seto in sol_sets:
                res = tuple(set(res + tuple(seto))) 
            # this is problematic... wrong solutions are being disregarded...
            if res  != universe:
                continue
            obj += len(intlist)
            energy += obj * res_dic[state]
            total_counts += res_dic[state]
        #print(energy/total_counts)

        return energy/total_counts
    
    return setupaClCostfct



def init_state(qv):
    x(qv)
    return qv



