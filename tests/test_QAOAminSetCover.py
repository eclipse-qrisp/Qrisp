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
from qrisp.qaoa import QAOAProblem, RZ_mixer, approximation_ratio, create_min_set_cover_mixer, create_min_set_cover_cl_cost_function, min_set_cover_init_function
import itertools


def test_QAOAminSetCover():

    sets = [{0,1,2,3},{1,5,6,4},{0,2,6,3,4,5},{3,4,0,1},{1,2,3,0},{1}]
    universe = set.union(*sets)
    qarg = QuantumVariable(len(sets))

    qaoa_min_set_cover = QAOAProblem(cost_operator=RZ_mixer, 
                                    mixer= create_min_set_cover_mixer(sets, universe), 
                                    cl_cost_function=create_min_set_cover_cl_cost_function(sets, universe),
                                    init_function=min_set_cover_init_function)
    results = qaoa_min_set_cover.run(qarg=qarg, depth=5)

    cl_cost = create_min_set_cover_cl_cost_function(sets, universe)


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
    InitTest = QAOAinstance.run(qarg=qarg, depth=5, mes_kwargs = {"shots" : 100000})

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
    
    min = len(sets)
    min_index = 0
    for index in range(len(binStrings)):
        val = cl_cost({binStrings[index] : 1})
        if val < min:
            min = val
            min_index = index

    optimal_sol = binStrings[min_index]

    # approximation ratio test
    assert approximation_ratio(results, optimal_sol, cl_cost)>=0.5
    