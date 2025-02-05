"""
\********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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
from qrisp.qaoa import QAOAProblem, RZ_mixer, approximation_ratio, create_min_set_cover_mixer, create_min_set_cover_cl_cost_function, min_set_cover_init_function
import itertools


def test_QAOAminSetCover():

    sets = [{0,1,2,3},{1,5,6,4},{0,2,6,3,4,5},{3,4,0,1},{1,2,3,0},{1}]
    universe = set.union(*sets)
    qarg = QuantumVariable(len(sets))

    qaoa_min_set_cover = QAOAProblem(cost_operator=RZ_mixer, 
                                    mixer= create_min_set_cover_mixer(sets, universe), 
                                    cl_cost_function=create_min_set_cover_cl_cost_function(sets, universe),
                                    init_function=min_set_cover_init_function)
    results = qaoa_min_set_cover.run(qarg=qarg, depth=5)

    cl_cost = create_min_set_cover_cl_cost_function(sets, universe)

    # find optimal solution by brute force    
    temp_binStrings = list(itertools.product([1,0], repeat=len(sets)))
    binStrings = ["".join(map(str, item))  for item in temp_binStrings]
    
    min = len(sets)
    min_index = 0
    for index in range(len(binStrings)):
        val = cl_cost({binStrings[index] : 1})
        if val < min:
            min = val
            min_index = index

    optimal_sol = binStrings[min_index]

    # approximation ratio test
    assert approximation_ratio(results, optimal_sol, cl_cost)>=0.5
    