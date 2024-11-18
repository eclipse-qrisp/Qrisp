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


from qrisp import QuantumVariable
from qrisp.qaoa import QAOAProblem, RZ_mixer, create_max_indep_set_cl_cost_function, create_max_indep_set_mixer, max_indep_set_init_function, approximation_ratio
import networkx as nx


def test_QAOAtrain_func():

    G = nx.erdos_renyi_graph(9, 0.7, seed =  133)
    G_complement = nx.complement(G)
    qarg = QuantumVariable(G.number_of_nodes())

    qaoa_max_clique = QAOAProblem(cost_operator=RZ_mixer,
                                    mixer=create_max_indep_set_mixer(G_complement),
                                    cl_cost_function=create_max_indep_set_cl_cost_function(G_complement),
                                    init_function=max_indep_set_init_function)
    training_func = qaoa_max_clique.train_function(qarg=qarg, depth=5)

    qarg2 = QuantumVariable(G.number_of_nodes())
    training_func(qarg2)
    results = qarg2.get_measurement()

    cl_cost = create_max_indep_set_cl_cost_function(G_complement)

    cliques = list(nx.find_cliques(G))

    max = 0
    max_index = 0
    for index, clique in enumerate(cliques):
        if len(clique) > max:
            max = len(clique)
            max_index = index

    optimal_sol = "".join(["1" if index in cliques[max_index] else "0" for index in range(G.number_of_nodes())])

    # approximation ratio test
    assert approximation_ratio(results, optimal_sol, cl_cost)>=0.5 


    #Instanciate QAOA
    #mixer gets Graph as argument
    #cost function gets graph as argument 
    QAOAinstance = QAOAProblem(cost_operator= maxCliqueCostOp(giraf), mixer= RX_mixer, cl_cost_function=maxCliqueCostfct(giraf)) 
    QAOAinstance.set_init_function(init_function=init_state)
    qarg2 = QuantumVariable(giraf.number_of_nodes())
    training_func = QAOAinstance.train_function( qarg=qarg2, depth=5, mes_kwargs = {"shots" : 100000} )
    qarg3 = QuantumVariable(giraf.number_of_nodes())
    training_func(qarg3)
    theNiceQAOA = qarg3.get_measurement()


    import itertools
    def aClcostFct(state, G):
        # we assume solution is right
        temp = True
        energy = 0 
        intlist = [s for s in range(len(list(state))) if list(state)[s] == "1"]
        # get all combinations of vertices in graph that are marked as |1> by the solution 
        combinations = list(itertools.combinations(intlist, 2))
        # if any combination is not found in the list of G.edges(), the solution is wrong, and energy == 0
        for combination in combinations:
            if combination not in G.edges():
                temp = False
        # else we just add the number of marked as |1> nodes
        if temp: 
            energy = -len(intlist)
        return(energy)

    
    # find the nx solutions
    the_it = list(nx.find_cliques(giraf))
    the_itt = [set(it) for it in the_it]
    print(the_itt)
    
    #print the ideal solutions
    #print("5 most likely Solutions") 
    #check if condition is fullfilled for 5 best sols
    maxfive = sorted(theNiceQAOA, key=theNiceQAOA.get, reverse=True)[:5]
    print(maxfive)
    for name in theNiceQAOA.keys():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if name in maxfive:
            print(name)
            print(aClcostFct(name, giraf))
        if aClcostFct(name, giraf) < 0:
            intlist = [s for s in range(len(list(name))) if list(name)[s] == "1"]
            intlist2 = set(intlist)
            set_g = False
            for seto in the_itt:
                if intlist2.issubset(seto):
                    set_g = True
                    break
            assert set_g

                #assert set(intlist) in the_itt
            #assert aClcostFct(name, giraf) <= -1
