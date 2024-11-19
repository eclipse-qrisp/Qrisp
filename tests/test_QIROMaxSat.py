"""
\********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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
from qrisp.algorithms.qaoa import create_maxsat_cl_cost_function, approximation_ratio
from qrisp.algorithms.qiro import QIROProblem, qiro_init_function, qiro_rx_mixer, create_maxsat_replacement_routine, create_maxsat_cost_operator_reduced
import itertools

def test_qiro_maxsat():

    clauses = [[1,2,-3],[1,4,-6],[4,5,6],[1,3,-4],[2,4,5],[1,3,5],[-2,-3,6],[1,7,8],[3,-7,-8],[3,4,8],[4,5,8],[1,2,7]]
    num_vars = 8
    problem = (num_vars, clauses)
    qarg = QuantumVariable(num_vars)

    cl_cost = create_maxsat_cl_cost_function(problem)    

    # assign the correct new update functions for qiro from above imports
    qiro_instance = QIROProblem(problem = problem,  
                                replacement_routine = create_maxsat_replacement_routine, 
                                cost_operator = create_maxsat_cost_operator_reduced,
                                mixer = qiro_rx_mixer,
                                cl_cost_function = create_maxsat_cl_cost_function,
                                init_function = qiro_init_function
                                )



    res_qiro = qiro_instance.run_qiro(qarg=qarg, depth = 3, n_recursions = 2,mes_kwargs={"shots":100000})

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
    assert approximation_ratio(res_qiro, optimal_sol, cl_cost)>=0.5

