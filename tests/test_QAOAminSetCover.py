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

import itertools
from qrisp.qaoa import QAOAProblem
from qrisp.qaoa.mixers import RZ_mixer
from qrisp.qaoa.problems.minSetCoverInfrastr import minSetCoverclCostfct,minSetCoverCostOp, init_state

from qrisp import QuantumVariable

def test_QAOAminSetCover():


    """
    This is an example of the minSetCoer QAOA solution
    the problem is structured as follows:
    > Given a universe [n] and m subsets S (S_j in S, S_j is subset of [n]), find the minimum cardinality subcollection S' (S' is subset of S) s.t. their union recovers [n]. 

    We will not stick to mathematical assignment of variable names. 

    """


    # sets are given as list of lists
    examplesets = [[0,1,2,3],[1,5,6,4],[0,2,6,3,4,5],[3,4,0,1],[1,2,3,0],[1]]
    # full universe is given as a tuple
    sol = (0,1,2,3,4,5,6)
 

    # assign operators
    cost_fun = minSetCoverclCostfct(sets=examplesets,universe = sol)
    mixerOp = RZ_mixer
    costOp = minSetCoverCostOp(sets=examplesets, universe=sol)

    #initialize variable
    qarg = QuantumVariable(len(examplesets))

    #+run qaoa
    QAOAinstance = QAOAProblem(cost_operator=costOp ,mixer= mixerOp, cl_cost_function=cost_fun)
    QAOAinstance.set_init_function(init_function=init_state)
    InitTest = QAOAinstance.run(qarg=qarg, depth=5)

    # create example cost_func
    def testCostFun(state,universe):
        obj = 0
        #literally just do a minus 1 op if state is equiv to a given condition
        intlist = [s for s in range(len(list(state))) if list(state)[s] == "1"]
        sol_sets = [examplesets[index] for index in intlist]
        res = ()
        for seto in sol_sets:
            res = tuple(set(res+ tuple(seto))) 
        if res == universe:
            obj += len(intlist)

        return obj

    # test if cost_func functions for all items . 
    for state in InitTest.keys():
        if testCostFun(state=state, universe = sol) > 0:
            intlist = [s for s in range(len(list(state))) if list(state)[s] == "1"]
            sol_sets = [examplesets[index] for index in intlist]
            res = ()
            for seto in sol_sets:
                res = tuple(set(res+ tuple(seto))) 
            assert sol == res

    

    #print the ideal solutions
    #print("5 most likely Solutions") 
    maxfive = sorted(InitTest, key=InitTest.get, reverse=True)[:5]
    for name in InitTest.keys():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if name in maxfive:
            print(name)
            print(testCostFun(name, universe=sol))  

    #find ideal solution by brute force    
    temp_binStrings = list(itertools.product([1,0], repeat=len(examplesets)))
    binStrings = ["".join(map(str, item))  for item in temp_binStrings]
    
    mini = len(examplesets)
    for index in range(len(binStrings)):
        val = testCostFun(binStrings[index], universe=sol)
        if 0 < val < mini :
            
            mini = val
            min_index = index

    optimal_sol = binStrings[min_index]
    print("optimal solution")
    print(optimal_sol)


    # write approximation ratio test
    # metric_dict ={}
    # metric_dict["counts"] = InitTest
    # metric_dict["cost_func"] = minSetCoverclCostfct(sets=examplesets,universe = sol)
    # metric_dict["optimal_sol"] = optimal_sol
    
    # print(approximation_ratio(metric_dict))
    #assert 0 < approximation_ratio(metric_dict)<= 3  

    import numpy as np

    randInteg = np.random.randint(4,8)
    
    randInteg2 = np.random.randint(0,1)
    
    sol = tuple(range(randInteg))
    if randInteg2 == 0:
        examplesets = [sol,sol[0:randInteg-3],sol[randInteg-3:randInteg-1],sol[randInteg-1:]] 
    else:
        print(randInteg)
        examplesets = [[index] for index in range(randInteg)]

    # assign the operators
    cost_fun = minSetCoverclCostfct(sets=examplesets,universe=sol)
    mixerOp = RZ_mixer
    costOp = minSetCoverCostOp(sets=examplesets, universe=sol)

    #initialize the qarg
    qarg2 = QuantumVariable(len(examplesets))

    # run the qaoa
    QAOAinstance = QAOAProblem(cost_operator=costOp ,mixer= mixerOp, cl_cost_function=cost_fun)
    QAOAinstance.set_init_function(init_function=init_state)
    InitTest = QAOAinstance.run(qarg=qarg2, depth=5)

    maxOne =  sorted(InitTest, key=InitTest.get, reverse=True)[:1]
    if randInteg2 == 0:
        print(maxOne)
        assert maxOne == ["1000"]
    else:
        print(maxOne)
        assert maxOne == ["1"* len(examplesets)]

