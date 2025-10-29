"""
********************************************************************************
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
********************************************************************************
"""

import networkx as nx

from qrisp import QuantumVariable, jaspify
from qrisp.qaoa import (
    QAOAProblem,
    RX_mixer,
    create_maxcut_cost_operator,
    create_maxcut_sample_array_post_processor,
)


def test_jasp_QAOAmaxCut():

    @jaspify(terminal_sampling=True)
    def main():

        G = nx.erdos_renyi_graph(6, 0.7, seed=133)

        cl_cost = create_maxcut_sample_array_post_processor(G)

        qarg = QuantumVariable(G.number_of_nodes())

        qaoa_maxcut = QAOAProblem(
            cost_operator=create_maxcut_cost_operator(G),
            mixer=RX_mixer,
            cl_cost_function=cl_cost,
        )
        results = qaoa_maxcut.run(qarg, depth=5, max_iter=50, optimizer="SPSA")

        cut_value = cl_cost(results)

        return cut_value

    cut_value = main()
    assert cut_value < -3


def test_jasp_tqa_QAOAmaxCut():

    @jaspify(terminal_sampling=True)
    def main():

        G = nx.erdos_renyi_graph(6, 0.7, seed=133)

        cl_cost = create_maxcut_sample_array_post_processor(G)

        qarg = QuantumVariable(G.number_of_nodes())

        qaoa_maxcut = QAOAProblem(
            cost_operator=create_maxcut_cost_operator(G),
            mixer=RX_mixer,
            cl_cost_function=cl_cost,
        )
        results = qaoa_maxcut.run(
            qarg, depth=10, max_iter=0, init_type="tqa", optimizer="SPSA"
        )

        cut_value = cl_cost(results)

        return cut_value

    cut_value = main()
    # When using tqa initialization, discretized quantum annealing is performed. Converges to optimal solution for increasing depth even for 0 iterations of the optimizer.
    assert cut_value < -6
