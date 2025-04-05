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
from qrisp import *
from qrisp.qaoa import *
import networkx as nx

def test_TQA_warmstart():
    G = nx.Graph()
    edges = [(0, 5), (0, 3), (0, 7), (1, 3), (2, 4), (2, 7), (3, 4), (3, 7), (4, 6), (4, 7), (5, 7)]
    G.add_edges_from(edges)

    maxcut_instance = QAOAProblem(create_maxcut_cost_operator(G), RX_mixer, create_maxcut_cl_cost_function(G))

    count = 0
    repetitions=10
    for i in range (repetitions):
        
        benchmark_data1 = maxcut_instance.benchmark(lambda : QuantumVariable(len(G)),
                                depth_range = [3],
                                shot_range = [100000],
                                iter_range = [100],
                                optimal_solution = '00110110',
                                repetitions = 1
                                )

        #temp = benchmark_data1.rank(print_res = True)

        _,rndFO1=benchmark_data1.evaluate()

        approx1 = sum(rndFO1)/len(rndFO1)

        benchmark_data2 = maxcut_instance.benchmark(lambda : QuantumVariable(len(G)),
                                depth_range = [3],
                                shot_range = [100000],
                                iter_range = [100],
                                optimal_solution = '00110110',
                                init_type = 'tqa',
                                repetitions = 1
                                )

        #temp = benchmark_data2.rank(print_res = True)

        _,rndFO2=benchmark_data2.evaluate()

        approx2 = sum(rndFO2)/len(rndFO2)

        if approx1 > approx2:
            count += 1 


    # calculate p_empirical (the probability that the random start outperforms TQA warm start)
    p_empirical = count / repetitions

    assert p_empirical < 0.3

