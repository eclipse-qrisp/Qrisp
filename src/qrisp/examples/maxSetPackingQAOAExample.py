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
from qrisp.qaoa.mixers import RZ_Mixer
from qrisp.qaoa.problems.maxSetPackInfrastr import maxSetPackclCostfct,maxSetPackCostOp, get_neighbourhood_relations, init_state

from qrisp import QuantumVariable
"""
This is an example of the maxSetPacking QAOA solution
the problem is structured as follows:
> Given a universe [n] and m subsets S (S_j in S, S_j is subset of [n]), find the maximum cardinality subcollection S' (S' is subset of S) of pairwise dijoint sets.

We will not stick to mathematical assignment of variable names. 

"""

# sets are given as list of lists
sets = [[0,7,1],[6,5],[2,3],[5,4],[8,7,0],[1]]
# full universe is given as a tuple
sol = (0,1,2,3,4,5,6,7,8)

# the realtions between the sets, i.e. with vertice is in which other sets
print(get_neighbourhood_relations(sets, len_universe=len(sol)))

# assign the operators
cost_fun = maxSetPackclCostfct(sets=sets,universe=sol)
mixerOp = RZ_Mixer()
costOp = maxSetPackCostOp(sets=sets, universe=sol)

#initialize the qarg
qarg = QuantumVariable(len(sets))

# run the qaoa
QAOAinstance = QAOAProblem(cost_operator=costOp ,mixer= mixerOp, cl_cost_function=cost_fun)
QAOAinstance.set_init_function(init_function=init_state)
InitTest = QAOAinstance.run(qarg=qarg, depth=5)


# assign the cost_func
def testCostFun(state, universe):
    list_universe = [True]*len(universe)
    temp = True
    obj = 0
    intlist = [s for s in range(len(list(state))) if list(state)[s] == "1"]
    sol_sets = [sets[index] for index in intlist]
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

# print the most likely solutions
print("5 most likely Solutions") 
maxfive = sorted(InitTest, key=InitTest.get, reverse=True)[:5]
for name, age in InitTest.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
    if name in maxfive:

        print((name, age))
        print(testCostFun(name, universe=sol))  