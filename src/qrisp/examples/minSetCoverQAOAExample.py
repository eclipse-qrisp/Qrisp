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

from qrisp.algorithms.qaoa import *

from qrisp import QuantumVariable

"""
This is an example of the minSetCoer QAOA solution
the problem is structured as follows:
> Given a universe [n] and m subsets S (S_j in S, S_j is subset of [n]), find the minimum cardinality subcollection S' (S' is subset of S) s.t. their union recovers [n]. 

We will not stick to mathematical assignment of variable names. 

"""


# sets are given as list of lists
sets = [[0,1,2,3],[1,5,6,4],[0,2,6,3,4,5],[3,4,0,1],[1,2,3,0],[1]]
# full universe is given as a tuple
universe = (0,1,2,3,4,5,6)


# assign operators


#initialize variable
qarg = QuantumVariable(len(sets))

#+run qaoa
QAOAinstance = QAOAProblem(cost_operator=RZ_mixer,
                        mixer= create_min_set_cover_mixer(sets, universe),
                        cl_cost_function=create_min_set_cover_cl_cost_function(sets, universe),
                        init_function=min_set_cover_init_function)
InitTest = QAOAinstance.run(qarg=qarg, depth=5)

# create example cost_func
def testCostFun(state,universe):
    obj = 0
    #literally just do a minus 1 op if state is equiv to a given condition
    intlist = [s for s in range(len(list(state))) if list(state)[s] == "1"]
    sol_sets = [sets[index] for index in intlist]
    res = ()
    for seto in sol_sets:
        res = tuple(set(res+ seto)) 
    if res == universe:
        obj -= len(intlist)

    return obj


# print most likely solutions
print("5 most likely Solutions -- 2") 
maxfive = sorted(InitTest, key=InitTest.get, reverse=True)[:5]
for name, age in InitTest.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
    if name in maxfive:

        print((name, age))
        print(testCostFun(name, universe))