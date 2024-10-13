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
from qrisp.qaoa import QAOAProblem, RX_mixer, approximation_ratio
from qrisp.qaoa.problems.eThrLinTwo import create_e3lin2_cl_cost_function, create_e3lin2_cost_operator, e3lin2_init_function
import itertools


def test_eTwoThrLinQAOA():

    clauses = [[0,1,2,1],[1,2,3,0],[0,1,4,0],[0,2,4,1],[2,4,5,1],[1,3,5,1],[2,3,4,0]]
    num_variables = 6
    qarg = QuantumVariable(num_variables)

    qaoa_e3lin2 = QAOAProblem(cost_operator=create_e3lin2_cost_operator(clauses),
                                    mixer=RX_mixer,
                                    cl_cost_function=create_e3lin2_cl_cost_function(clauses),
                                    init_function=e3lin2_init_function)
    results = qaoa_e3lin2.run(qarg=qarg, depth=5)

    cl_cost = create_e3lin2_cl_cost_function(clauses)

    # find optimal solution by brute force    
    temp_binStrings = list(itertools.product([1,0], repeat=num_variables))
    binStrings = ["".join(map(str, item)) for item in temp_binStrings]
    
    min = 0
    min_index = 0
    for index in range(len(binStrings)):
        val = cl_cost({binStrings[index] : 1})
        if val < min:
            min = val
            min_index = index

    optimal_sol = binStrings[min_index]
    
    # approximation ratio test
    assert approximation_ratio(results, optimal_sol, cl_cost)>=0.5