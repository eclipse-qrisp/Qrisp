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


"""


problem = [6 , [[1,2,-3],[1,4,-6], [4,5,6],[1,3,-4],[2,4,5],[1,3,5],[-2,-3,6]]]

clauses11 = problem[1]


qarg = QuantumVariable(problem[0])


QAOAinstance = QAOAProblem(cost_operator=create_maxsat_cost_operator(problem), mixer=RX_mixer, cl_cost_function=create_maxsat_cl_cost_function(problem))

theNiceQAOA = QAOAinstance.run(qarg=qarg, depth=5)

print("5 most likely Solutions") 
maxfive = sorted(theNiceQAOA, key=theNiceQAOA.get, reverse=True)[:5]
for name, age in theNiceQAOA.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
    if name in maxfive:
        print((name, age))


print("Final energy value and associated solution values")
costfct = create_maxsat_cl_cost_function(problem)
print()
print(costfct(theNiceQAOA))
#Final Result-dictionary
resDict = dict()


#print(resDict)

