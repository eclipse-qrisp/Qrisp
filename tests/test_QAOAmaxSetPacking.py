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
from qrisp.qaoa.problems.maxSetPackInfrastr import maxSetPackclCostfct,maxSetPackCostOp,get_neighbourhood_relations, init_state
from qrisp.qaoa.mixers import RZ_mixer
from qrisp import QuantumVariable
from qrisp.qaoa import approximation_ratio
import matplotlib.pyplot as plt
import itertools


def QAOAmaxSetPacking():
    # start with a graph G - from np adjacency matrix
    """
    This is an example of the maxSetPacking QAOA solution
    the problem is structured as follows:
    > Given a universe [n] and m subsets S (S_j in S, S_j is subset of [n]), find the maximum cardinality subcollection S' (S' is subset of S) of pairwise dijoint sets.

    We will not stick to mathematical assignment of variable names. 

    """

    # sets are given as list of tuples
    examplesets = [(0,1,7),(5,6),(2,3),(4,5),(7,8)]
    # full universe is given as a tuple
    sol = (0,1,2,3,4,5,6,7,8)

    # the realtions between the sets, i.e. with vertice is in which other sets
    #print(get_neighbourhood_relations(examplesets, len_universe=len(sol)))

    # assign the operators
    cost_fun = maxSetPackclCostfct(sets=examplesets,universe=sol)
    mixerOp = RZ_mixer
    costOp = maxSetPackCostOp(sets=examplesets, universe=sol)

    #initialize the qarg
    qarg = QuantumVariable(len(examplesets))

    # run the qaoa
    QAOAinstanceSecond = QAOAProblem(cost_operator=costOp ,mixer= mixerOp, cl_cost_function=cost_fun)
    QAOAinstanceSecond.set_init_function(init_function=init_state)
    InitTest = QAOAinstanceSecond.run(qarg=qarg, depth=5, mes_kwargs = {"shots" : 100000})


    # assign the cost_func for later usage
    def testCostFun(state, universe):
        list_universe = [True]*len(universe)
        temp = True
        obj = 0
        intlist = [s for s in range(len(list(state))) if list(state)[s] == "1"]
        sol_sets = [examplesets[index] for index in intlist]
        for seto in sol_sets:
            for val in seto:
                if list_universe[val]:
                    list_universe[val] = False
                else: 
                    temp = False 
                    break
        if temp:
            obj -= len(intlist)
        return obj




    # test if cost_func functions for all items . 
    for state in InitTest.keys():
        list_universe = [True]*len(sol)
        temp = True
        if testCostFun(state=state, universe = sol) < 0:
            intlist = [s for s in range(len(list(state))) if list(state)[s] == "1"]
            sol_sets = [examplesets[index] for index in intlist]
            for seto in sol_sets:
                for val in seto:
                    if list_universe[val]:
                        list_universe[val] = False
                    else: 
                        temp = False 
                        break
        assert temp

    

    #print the ideal solutions
    #print("5 most likely Solutions") 
    maxfive = sorted(InitTest, key=InitTest.get, reverse=True)[:5]
    """ for name in InitTest.keys():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if name in maxfive:
            print(name)
            print(testCostFun(name, universe=sol))  
    """
    #find ideal solution by brute force    
    temp_binStrings = list(itertools.product([1,0], repeat=len(examplesets)))
    binStrings = ["".join(map(str, item))  for item in temp_binStrings]
    
    mini = 0
    for index in range(len(binStrings)):
        val = testCostFun(binStrings[index], universe=sol)
        if val < mini:
            mini = val
            min_index = index

    optimal_sol = binStrings[min_index]
    #print("optimal solution")
    #print(optimal_sol)


    # write approximation ratio test
    metric_dict ={}
    metric_dict["counts"] = InitTest
    metric_dict["cost_func"] = maxSetPackclCostfct(sets=examplesets,universe=sol)
    metric_dict["optimal_sol"] = optimal_sol
    
    #print(approximation_ratio(metric_dict))
    """ if not approximation_ratio(metric_dict)>= 0.3:
        failure_list.append(approximation_ratio(metric_dict))
        counter +=1 """
    #assert approximation_ratio(metric_dict)>= 0.3



    import numpy as np

    randInt = np.random.randint(4,9)
    randInt2 = np.random.randint(0,1)
    secondSol = tuple(range(randInt))

    if randInt2 == 0:
        secondSets = [secondSol,secondSol[0:randInt-3],secondSol[randInt-3:randInt-1],secondSol[randInt-1:]] 
    else:
        #print(list(range(randInt)))
        #examplesets = [tuple(index) for index in list(range(randInt))]
        secondSets = [secondSol[0:randInt-3],secondSol[randInt-3:randInt-1],secondSol[randInt-1:]] 



    # assign the operators
    cost_fun = maxSetPackclCostfct(sets=secondSets,universe=secondSol)
    mixerOp = RZ_mixer
    costOp = maxSetPackCostOp(sets=secondSets, universe=secondSol)

    #initialize the qarg
    qarg2 = QuantumVariable(len(secondSets))

    # run the qaoa
    QAOAinstanceSecond = QAOAProblem(cost_operator=costOp ,mixer= mixerOp, cl_cost_function=cost_fun)
    QAOAinstanceSecond.set_init_function(init_function=init_state)
    InitTest2 = QAOAinstanceSecond.run(qarg=qarg2, depth=5, mes_kwargs = {"shots" : 100000})

    maxOne =  sorted(InitTest2, key=InitTest2.get, reverse=True)[:1]
    if randInt2 == 0:
        
        #print(maxOne)
        assert maxOne == ["0111"]

    else:
        #print(maxOne)
        assert maxOne == ["1"* 3 ]
    #print(counter)
    #print(failure_list)