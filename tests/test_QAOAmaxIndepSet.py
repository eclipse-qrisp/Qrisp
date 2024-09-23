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
from qrisp.qaoa.problems.maxIndepSetInfrastr import maxIndepSetCostOp, maxIndepSetclCostfct,  init_state
from qrisp.qaoa.mixers import RX_mixer
from qrisp import QuantumVariable
from qrisp.qaoa import approximation_ratio

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def test_QAOAmaxIndepSet():
    # start with a graph G - from np adjacency matrix

    giraf = nx.erdos_renyi_graph(9,0.2)
    #print(nx.maximum_independent_set(giraf))
    nx_sol = nx.maximal_independent_set(giraf)
    print(nx_sol)
    #draw graph


    qarg = QuantumVariable(giraf.number_of_nodes())


    #Instanciate QAOA
    #mixer gets Graph as argument
    #cost function gets graph as argument 
    QAOAinstance = QAOAProblem(cost_operator=RX_mixer, mixer=maxIndepSetCostOp(giraf), cl_cost_function=maxIndepSetclCostfct(giraf))
    QAOAinstance.set_init_function(init_function=init_state)
    theNiceQAOA = QAOAinstance.run(qarg=qarg,depth=5)



    import itertools
    def aClcostFct(state, G):
        
        # we assume solution is right
        temp = True
        energy = 0 
        intlist = [s for s in range(len(list(state))) if list(state)[s] == "1"]
        # get all combinations of vertices in graph that are marked as |1> by the solution 
        combinations = list(itertools.combinations(intlist, 2))
        # if any combination is found in the list of G.edges(), the solution is wrong, and energy == 0
        for combination in combinations:
            if combination in G.edges():
                temp = False
        # else we just add the number of marked as |1> nodes
        if temp: 
            energy = -len(intlist)
        return(energy)

    
    # write correct sol test --> this done in the cost function... 
    for state in theNiceQAOA.keys():
        if aClcostFct(state=state, G=giraf) < 0:
            intlist = [s for s in range(len(list(state))) if list(state)[s] == "1"]
            # get all combinations of vertices in graph that are marked as |1> by the solution 
            combinations = list(itertools.combinations(intlist, 2))
            for combination in combinations:
                assert combination not in giraf.edges()


    #print the ideal solutions
    #print("5 most likely Solutions") 
    maxfive = sorted(theNiceQAOA, key=theNiceQAOA.get, reverse=True)[:5]
    for name in theNiceQAOA.keys():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if name in maxfive:
            print(name)
            print(aClcostFct(name, giraf))

    # write approximation ratio test
    metric_dict ={}
    metric_dict["counts"] = theNiceQAOA
    metric_dict["cost_func"] = maxIndepSetclCostfct(giraf)

    this = ["1" if s in nx_sol else "0" for s in range(giraf.number_of_nodes()) ]
    this2 = "".join(this)
    metric_dict["optimal_sol"] = this2
    
    assert approximation_ratio(metric_dict["counts"], metric_dict["optimal_sol"], maxIndepSetclCostfct(giraf))>= 0.1

    # test for full solution...
    integ = np.random.randint(1,7)
    integ2 = np.random.randint(0,1)
    integ2 = 0
    giraf1 = nx.erdos_renyi_graph(integ,integ2)
    qarg2 = QuantumVariable(giraf1.number_of_nodes())
    QAOAinstance2 = QAOAProblem(cost_operator= RX_mixer, mixer= maxIndepSetCostOp(giraf1), cl_cost_function=maxIndepSetclCostfct(giraf1)) 
    QAOAinstance2.set_init_function(init_function=init_state)
    theNiceQAOA2 = QAOAinstance2.run(qarg=qarg2, depth= 5)
    maxOne = sorted(theNiceQAOA2, key=theNiceQAOA2.get, reverse=True)[:1]
    if integ2 == 0:
        testStr = [str(1-integ2) * giraf1.number_of_nodes()]
    if integ2 == 1:
        testStr = [str(1-integ2) * giraf1.number_of_nodes()]
        string = "1"
        string += str(1-integ2) * (giraf1.number_of_nodes()-1)
        testStr = [string]
    #print(testStr)
    #print(maxOne)
    assert testStr == maxOne
