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
import numpy as np 


def test_eTwoThrLinQAOA():
    # first three are the indices (a_1,j ; a_2,j ; a3,j), last one is b_L
    #clauses = [(1,3,4,5), (2,6,9,11), (1,5,8,33), (3,4,5,77), (2,4,6,7), (1,3,4,55),(1,3,6,49),(2,4,7,87),  (1,3,4,55),(1,3,6,43),(2,4,7,89), (1,3,4,57),(1,3,6,44),(2,4,7,8)]
    for index in range(10):
        clauses = []
        for index2 in range(4):
            index2a = np.random.randint(0,1)
            if index2a % 2 == 0:
                int1 = np.random.randint(3,5)
                int2 = np.random.randint(0,3)
                int3 = np.random.randint(5,7)
                int4 = np.random.randint(30,39)
                clauses.append((int1,int2,int3,int4))
            else:
                int1 = np.random.randint(1,4)
                int2 = np.random.randint(0,1)
                int3 = np.random.randint(4,7)
                int4 = np.random.randint(31,40)
                clauses.append((int1,int2,int3,int4))

        #assign cost function
        cl_cost_function=eTwoThrLinclCostfct(clauses)
        print(clauses)

        #assign qarg
        qarg = QuantumVariable(7)

        # run the qaoa
        QAOAinstance = QAOAProblem(cost_operator=eThrTwocostOp(clauses=clauses), mixer=RX_mixer, cl_cost_function=cl_cost_function)
        QAOAinstance.set_init_function(init_function=init_state)
        theNiceQAOA = QAOAinstance.run(qarg=qarg, depth=5)

        import itertools
        def testCostFun(state):
            obj = 0
                #ljust do a minus 1 op if state is equiv to a given condition clause
            for index in range(len(clauses)):
                clause = clauses[index]
                sum = 0
                for index_aj in range(0,2):
                    sum += int(state[clause[index_aj]])
                if sum == clause[3] % 2:
                    obj -= 1 
            return obj
        
        
        #maxfive = sorted(theNiceQAOA, key=theNiceQAOA.get, reverse=True)[:1]
        for name in theNiceQAOA.keys():  # for name, age in dictionary.iteritems():  (for Python 2.x)
            if testCostFun(name) < 0:
                temp = False

                for clause in clauses:
                    sum = 0
                    for index_aj in range(0,2):
                        sum += int(name[clause[index_aj]])
                    if sum == clause[3] % 2:
                        temp  = True
                assert temp 
    
