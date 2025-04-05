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

from qrisp.qaoa import QAOAProblem, RX_mixer, create_maxcut_cl_cost_function,create_maxcut_cost_operator
from qrisp import QuantumArray, QuantumVariable
import networkx as nx
from operator import itemgetter

def test_maxcut():
    def maxcut_obj(x,G):
        cut = 0
        for i, j in G.edges():
            if x[i] != x[j]:                        
                cut -= 1    
        return cut

    depth = 5

    from qrisp.default_backend import def_backend

    qrisp_sim = def_backend
    qaoa_backend = qrisp_sim

    ###### Trivial case with 1 edge
    G1 = nx.Graph()
    G1.add_edge(0, 1)

    qarg_prep = lambda : QuantumArray(qtype = QuantumVariable(1), shape = len(G1))

    maxcut_instance1 = QAOAProblem(create_maxcut_cost_operator(G1), RX_mixer, create_maxcut_cl_cost_function(G1))

    res1 = maxcut_instance1.run(qarg_prep, depth, mes_kwargs={"backend" : qaoa_backend, "shots" : 100000}, max_iter = 50)
    best_cut1, best_solution1 = min([(maxcut_obj(x,G1),x) for x in res1.keys()], key=itemgetter(0))

    res_str1 = list(res1.keys())[0]
    best_cut1, best_solution1 = (maxcut_obj(res_str1,G1),res_str1)
    assert -best_cut1 == 1

    ####### Less trivial case
    G4 = nx.Graph()
    G4.add_edges_from([[0,3],[0,4],[1,3],[1,4],[2,3],[2,4]])

    qarg_prep = lambda : QuantumArray(qtype = QuantumVariable(1), shape = len(G4))

    maxcut_instance4 = QAOAProblem(create_maxcut_cost_operator(G4), RX_mixer, create_maxcut_cl_cost_function(G4))
    res4 = maxcut_instance4.run(qarg_prep, depth, mes_kwargs={"backend" : qaoa_backend, "shots" : 100000}, max_iter = 50)
    best_cut4, best_solution4 = min([(maxcut_obj(x,G4),x) for x in res4.keys()], key=itemgetter(0))

    res_str4 = list(res4.keys())[0]
    best_cut4, best_solution1 = (maxcut_obj(res_str4,G4),res_str4)
    assert -best_cut4 == 6
