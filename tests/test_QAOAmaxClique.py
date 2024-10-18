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
from qrisp.qaoa import QAOAProblem, RZ_mixer, create_max_indep_set_cl_cost_function, create_max_indep_set_mixer, max_indep_set_init_function, approximation_ratio
import networkx as nx


def test_QAOAmaxClique():

    G = nx.erdos_renyi_graph(9, 0.7, seed =  133)
    G_complement = nx.complement(G)
    qarg = QuantumVariable(G.number_of_nodes())

    qaoa_max_clique = QAOAProblem(cost_operator=RZ_mixer,
                                    mixer=create_max_indep_set_mixer(G_complement),
                                    cl_cost_function=create_max_indep_set_cl_cost_function(G_complement),
                                    init_function=max_indep_set_init_function)
    results = qaoa_max_clique.run(qarg=qarg, depth=5)

    cl_cost = create_max_indep_set_cl_cost_function(G_complement)

    cliques = list(nx.find_cliques(G))

    max = 0
    max_index = 0
    for index, clique in enumerate(cliques):
        if len(clique) > max:
            max = len(clique)
            max_index = index

    optimal_sol = "".join(["1" if index in cliques[max_index] else "0" for index in range(G.number_of_nodes())])

    # approximation ratio test
    assert approximation_ratio(results, optimal_sol, cl_cost)>=0.5 


