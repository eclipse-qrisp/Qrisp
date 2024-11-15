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
from qrisp.qaoa import QAOAProblem, RZ_mixer, approximation_ratio, create_max_indep_set_cl_cost_function, create_max_indep_set_mixer, max_indep_set_init_function
import networkx as nx
import itertools


def QAOAmaxSetPacking():

    sets = [{0,7,1},{6,5},{2,3},{5,4},{8,7,0},{1}]

    def non_empty_intersection(sets):
        return [(i, j) for (i, s1), (j, s2) in itertools.combinations(enumerate(sets), 2) if s1.intersection(s2)]

    # create constraint graph
    G = nx.Graph()
    G.add_nodes_from(range(len(sets)))
    G.add_edges_from(non_empty_intersection(sets))
    qarg = QuantumVariable(G.number_of_nodes())

    qaoa_max_indep_set = QAOAProblem(cost_operator=RZ_mixer,
                                    mixer=create_max_indep_set_mixer(G),
                                    cl_cost_function=create_max_indep_set_cl_cost_function(G),
                                    init_function=max_indep_set_init_function)
    results = qaoa_max_indep_set.run(qarg=qarg, depth=5)

    cl_cost = create_max_indep_set_cl_cost_function(G)

    # find optimal solution by brute force    
    temp_binStrings = list(itertools.product([1,0], repeat=len(sets)))
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
