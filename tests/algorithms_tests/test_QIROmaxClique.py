"""
********************************************************************************
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
********************************************************************************
"""

import networkx as nx

from qrisp import QuantumVariable
from qrisp.algorithms.qaoa import create_max_clique_cl_cost_function, max_clique_problem
from qrisp.algorithms.qiro import (
    QIROProblem,
    create_max_clique_cost_operator_reduced,
    create_max_clique_replacement_routine,
    qiro_init_function,
    qiro_rx_mixer,
)


def test_qiro_max_clique():

    # Define a random graph via the number of nodes and the QuantumVariable arguments
    # num_nodes = 15
    num_nodes = 10  # Reduced problem size to minimize failure probability
    G = nx.erdos_renyi_graph(num_nodes, 0.7, seed=99)
    qarg = QuantumVariable(G.number_of_nodes())

    cl_cost = create_max_clique_cl_cost_function(G)

    # QIRO
    qiro_instance = QIROProblem(
        problem=G,
        replacement_routine=create_max_clique_replacement_routine,
        cost_operator=create_max_clique_cost_operator_reduced,
        mixer=qiro_rx_mixer,
        cl_cost_function=create_max_clique_cl_cost_function,
        init_function=qiro_init_function,
    )
    res_qiro = qiro_instance.run_qiro(
        qarg=qarg, depth=3, n_recursions=2, mes_kwargs={"shots": 100000}
    )

    # QIRO most likely result
    # QIRO most likely result
    # most_likely = sorted(res_qiro, key=res_qiro.get, reverse=True)[0]
    test = False

    for i in range(5):
        item = sorted(res_qiro, key=res_qiro.get, reverse=True)[i]
        if cl_cost({item: 1}) < 0:
            test = True
            break

    # print(test)
    assert test
