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


from qrisp.qaoa import QAOAProblem
from qrisp.qaoa.problems.maxCliqueInfrastr import maxCliqueCostfct,maxCliqueCostOp,init_state
from qrisp.qaoa.mixers import RX_mixer
from qrisp import QuantumVariable
import networkx as nx
import matplotlib.pyplot as plt
import  numpy as np


def test_QAOAmaxClique():

    giraf = nx.erdos_renyi_graph(9,0.7)
    #draw graph
    #nx.draw(giraf,with_labels = True)
    #plt.show() 


    qarg = QuantumVariable(giraf.number_of_nodes())



    #Instanciate QAOA
    #mixer gets Graph as argument
    #cost function gets graph as argument 
    QAOAinstance = QAOAProblem(cost_operator= maxCliqueCostOp(giraf), mixer= RX_mixer, cl_cost_function=maxCliqueCostfct(giraf)) 
    QAOAinstance.set_init_function(init_function=init_state)
    theNiceQAOA = QAOAinstance.run(qarg=qarg, depth= 5, mes_kwargs = {"shots" : 100000})

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

    # test for full solution...
    truth_list = []
    for index in range(10):

        integ = np.random.randint(1,7)
        #create random graph with either all nodes connected, or no nodes connected
        integ2 = np.random.randint(0,1)
        
        secondGraph = nx.erdos_renyi_graph(integ,integ2)
        secondQarg = QuantumVariable(secondGraph.number_of_nodes())
        secondInstance = QAOAProblem(cost_operator= maxCliqueCostOp(secondGraph), mixer= RX_mixer, cl_cost_function=maxCliqueCostfct(secondGraph)) 
        secondInstance.set_init_function(init_function=init_state)
        theNiceQAOA2 = secondInstance.run(qarg=secondQarg, depth= 3, max_iter = 50, mes_kwargs = {"shots" : 100000})
        maxOne = sorted(theNiceQAOA2, key=theNiceQAOA2.get, reverse=True)[:1]
        #if no nodes connected, maxClique sol should just be the first node, i.e. "1000" (more or less zeros)
        if integ2 == 0:
            string = "1"
            string += str(integ2) * (secondGraph.number_of_nodes()-1)
            testStr = [string]
        #if all nodes connected, maxClique sol should be all nodes, i.e. "1111" (more or less ones)
        if integ2 == 1:
            testStr = [str(integ2) * secondGraph.number_of_nodes()]

        
        if  testStr == maxOne:
            truth_list.append(1)
            
    assert sum(truth_list)/10 > 0.5
