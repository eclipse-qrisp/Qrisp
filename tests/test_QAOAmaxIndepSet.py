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


def test_QAOAmaxIndepSet():

    G = nx.erdos_renyi_graph(9, 0.5, seed =  133)
    qarg = QuantumVariable(G.number_of_nodes())

    qaoa_max_indep_set = QAOAProblem(cost_operator=RZ_mixer,
                                    mixer=create_max_indep_set_mixer(G),
                                    cl_cost_function=create_max_indep_set_cl_cost_function(G),
                                    init_function=max_indep_set_init_function)
    results = qaoa_max_indep_set.run(qarg=qarg, depth=5)

    cl_cost = create_max_indep_set_cl_cost_function(G)

    # check if all states are valid solutions 
    for state in results.keys():
            indices = [index for index, value in enumerate(state) if value == '1']
            combinations = list(itertools.combinations(indices, 2))

            for combination in combinations:
                assert combination not in G.edges()

    # approximation ratio test
    nx_sol = nx.maximal_independent_set(G)
    nx_sol = "".join(["1" if s in nx_sol else "0" for s in range(G.number_of_nodes())])
    assert approximation_ratio(results, nx_sol, cl_cost)>=0.5
