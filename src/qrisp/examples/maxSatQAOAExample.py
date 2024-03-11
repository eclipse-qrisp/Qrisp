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
from qrisp.qaoa.problems.maxSatInfrastr import maxSatclCostfct, maxSatCostOp, clausesdecoder, init_state
from qrisp.qaoa.mixers import RX_mixer
from qrisp import QuantumVariable

""" 
problem = [6 , [[1,2,-3],[1,4,-6], [4,5,6],[1,3,-4],[2,4,5],[1,3,5],[-2,-3,6]]]
Explanation
First index is number of variables
Clause : [1, 2, -4] 
fulfilled by : 1100, 1110 
--> if pos sign: the index has to be one; if neg sign the index has to be zero
--> indices not mentioned can be 0 or 1 
--> start with 1 in your clauses! because int 0 has no sign and this problematic 
--> we want to stay with mathematical formulation, so no handing over strings!

"""


problem = [6 , [[1,2,-3],[1,4,-6], [4,5,6],[1,3,-4],[2,4,5],[1,3,5],[-2,-3,6]]]

clauses11 = problem[1]

#Clauses are decoded, s.t. the Cost-Optimizer can read them
#numVars is the amount of considered variables, i.e. highest number (= Number of Qubits in Circuit aswell)
decodedClauses = clausesdecoder( problem)
#print(decodedClauses)

qarg = QuantumVariable(problem[0])

#CostOperator-Generator has to be called with the clauses
#CostFct-Generator has to be called with decodedClauses
QAOAinstance = QAOAProblem(cost_operator=maxSatCostOp(problem), mixer=RX_mixer, cl_cost_function=maxSatclCostfct(problem))
QAOAinstance.set_init_function(init_function=init_state)
theNiceQAOA = QAOAinstance.run(qarg=qarg, depth=5)

print("5 most likely Solutions") 
maxfive = sorted(theNiceQAOA, key=theNiceQAOA.get, reverse=True)[:5]
for name, age in theNiceQAOA.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
    if name in maxfive:
        print((name, age))
print()


print("Final energy value and associated solution values")
costfct = maxSatclCostfct(problem)
print()
print(costfct(theNiceQAOA))
#Final Result-dictionary
resDict = dict()
for index in decodedClauses:
    for index2 in index:
        if index2 in resDict.keys():
            resDict[index2] +=1
        else:
            resDict[index2] =1
print(resDict)

