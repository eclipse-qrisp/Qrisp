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
def test_RTE():
    import networkx as nx
    import numpy as np
    from qrisp.operators import X, Z
    from qrisp import QuantumVariable

    def generate_chain_graph(N):
        G = nx.Graph()
        G.add_edges_from([[k,k+1] for k in range(N-1)])
        return G

    def create_ising_hamiltonian(G, J, B):
        H = sum(-J*Z(i)*Z(j) for (i,j) in G.edges()) + sum(B*X(i) for i in G.nodes())
        return H

    def create_magnetization(G):
        H = (1.0/G.number_of_nodes())*sum(Z(i) for i in G.nodes())
        return H

    T_values = np.arange(0, 2, 0.05)
    precision_expectation_value = 0.005
    G=generate_chain_graph(6)
    H = create_ising_hamiltonian(G, J=1.0, B=1.0)
    M  = create_magnetization(G)
    n_max= 20
    M_values = []

    def psi(t):
        qv = QuantumVariable(G.number_of_nodes())
        RTE(H, qv, t, n_max)
        return qv  

    def magnetization_at_t(t, repeats=50):
        vals = []
        for _ in range(repeats):
            ev_M = M.expectation_value(psi, precision_expectation_value)
            vals.append(float(ev_M(t)))
        return sum(vals) / len(vals)

    for t in T_values:
        m_t = magnetization_at_t(t, repeats=5)  
        M_values.append(m_t)

    assert M_values[0] - M_values[-1] > 0.6