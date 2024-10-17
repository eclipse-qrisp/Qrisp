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
from qrisp.qaoa import QAOAProblem, RX_mixer, approximation_ratio, create_maxsat_cl_cost_function, create_maxsat_cost_operator
import itertools


def test_QAOAmaxSat():

    clauses = [[1,2,-3],[1,4,-6],[4,5,6],[1,3,-4],[2,4,5],[1,3,5],[-2,-3,6]]
    num_vars = 6
    problem = (num_vars, clauses)
    qarg = QuantumVariable(num_vars)

    qaoa_max_indep_set = QAOAProblem(cost_operator=create_maxsat_cost_operator(problem),
                                    mixer=RX_mixer,
                                    cl_cost_function=create_maxsat_cl_cost_function(problem))
    results = qaoa_max_indep_set.run(qarg=qarg, depth=5)

    cl_cost = create_maxsat_cl_cost_function(problem)

    # find optimal solution by brute force    
    temp_binStrings = list(itertools.product([1,0], repeat=num_vars))
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