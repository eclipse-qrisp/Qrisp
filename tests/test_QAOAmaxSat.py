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
from qrisp.qaoa.problems.maxSatInfrastr import maxSatclCostfct, maxSatCostOp, init_state, clausesdecoder
from qrisp.qaoa.mixers import RX_mixer
from qrisp import QuantumVariable
import networkx as nx
import matplotlib.pyplot as plt


def test_QAOAmaxSat():

    #draw graph
    #nx.draw(giraf,with_labels = True)
    #plt.show() 

    problem = [6, [[1,2,-3],[1,4,-6], [4,5,6],[1,3,-4],[2,4,5],[1,3,5],[-2,-3,6]]]
    clauses11 = problem[1]

    #Clauses are decoded, s.t. the Cost-Optimizer can read them
    #numVars is the amount of considered variables, i.e. highest number (= Number of Qubits in Circuit aswell)
    decodedClauses = clausesdecoder( problem)
    #print(decodedClauses)

    qarg = QuantumVariable(6)

    #CostOperator-Generator has to be called with the clauses
    #CostFct-Generator has to be called with decodedClauses
    QAOAinstance = QAOAProblem(cost_operator=maxSatCostOp(problem), mixer=RX_mixer, cl_cost_function=maxSatclCostfct(problem))
    QAOAinstance.set_init_function(init_function=init_state)
    theNiceQAOA = QAOAinstance.run(qarg=qarg, depth=5)


    import itertools
    def testCostFun(state):
        obj = 0
        #literally just do a minus 1 op if state is equiv to a given condition
        for index in range(len(decodedClauses)):
            if state in decodedClauses[index]:
                obj -= 1 

        return obj

    
    #print 5 best sols 
    # assert that sol is in decoded clauses 
    maxfive = sorted(theNiceQAOA, key=theNiceQAOA.get, reverse=True)[:5]
    # assert that sol is in decoded clauses 
    for name in theNiceQAOA.keys():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if name in maxfive:
            print(name)
            print(testCostFun(name))
        if testCostFun(name) < 0:
            temp = False
            for index in range(len(decodedClauses)):
                if name in decodedClauses[index]:
                    temp  = True
            assert temp

