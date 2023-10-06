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
from qrisp.qaoa.problems.eThrTwoLinInfrastr import eTwoThrLinclCostfct, eThrTwocostOp, init_state
from qrisp.qaoa.mixers import RX_mixer 
from qrisp import QuantumVariable

# first three are the indices (a_1,j ; a_2,j ; a3,j), last one is b_L
clauses = [(1,3,4,5), (2,6,9,11), (1,5,8,33), (3,4,5,77), (2,4,6,7), (1,3,4,55),(1,3,6,49),(2,4,7,87),  (1,3,4,55),(1,3,6,43),(2,4,7,89), (1,3,4,57),(1,3,6,44),(2,4,7,8)]

#assign cost function
cl_cost_function=eTwoThrLinclCostfct(clauses)

#assign qarg
qarg = QuantumVariable(len(clauses))

# run the qaoa
QAOAinstance = QAOAProblem(cost_operator=eThrTwocostOp(clauses=clauses), mixer=RX_mixer, cl_cost_function=cl_cost_function)
QAOAinstance.set_init_function(init_function=init_state)
theNiceQAOA = QAOAinstance.run(qarg=qarg, depth=5)

max_val = max(theNiceQAOA.values())

#print ideal solution
for name, age in theNiceQAOA.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
    if age == max_val:
        print((name, age))

costfct = eTwoThrLinclCostfct(clauses)

#print final costfunction value
print(costfct(theNiceQAOA)) 