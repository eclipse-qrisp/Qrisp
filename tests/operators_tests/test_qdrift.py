"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

import jax.numpy as jnp
import networkx as nx
import numpy as np
from qrisp import QuantumVariable
from qrisp.jasp import jaspify
from qrisp.operators import X, Z

def test_qdrift_ising_chain():

    # Helper functions
    def generate_chain_graph(N):
        G = nx.Graph()
        G.add_edges_from((k, k+1) for k in range(N-1))
        return G

    def create_ising_hamiltonian(G, J, B):
        # H = -J ∑ Z_i Z_{i+1} - B ∑ X_i
        H = sum(-J * Z(i)*Z(j) for (i, j) in G.edges()) + sum(B * X(i) for i in G.nodes())
        return H

    def create_magnetization(G):
        return (1.0 / G.number_of_nodes()) * sum(Z(i) for i in G.nodes())

    # Simulation setup
    N = 100
    G = generate_chain_graph(6)
    H = create_ising_hamiltonian(G, J=1.0, B=1.0)
    U = H.qdrift()
    M = create_magnetization(G)
    T_values = np.arange(0, 1.0, 0.1)

    # Define the state evolution function using QDrift
    def psi(t):
        qv = QuantumVariable(G.number_of_nodes())
        U(qv, t, samples=N)
        return qv

    # Compute magnetization expectation values over time
    M_values = []
    for t in T_values:
        ev_M = M.expectation_value(psi, precision=0.05)
        M_values.append(float(ev_M(t)))

    assert M_values[0] - M_values[-1] > 0.5


def test_jasp_qdrift_ising_chain():

    # Helper functions
    def generate_chain_graph(N):
        G = nx.Graph()
        G.add_edges_from((k, k+1) for k in range(N-1))
        return G

    def create_ising_hamiltonian(G, J, B):
        # H = -J ∑ Z_i Z_{i+1} - B ∑ X_i
        H = sum(-J * Z(i)*Z(j) for (i, j) in G.edges()) + sum(B * X(i) for i in G.nodes())
        return H

    def create_magnetization(G):
        return (1.0 / G.number_of_nodes()) * sum(Z(i) for i in G.nodes())

    # Simulation setup
    N = 100
    G = generate_chain_graph(6)
    H = create_ising_hamiltonian(G, J=1.0, B=1.0)
    U = H.qdrift()
    M = create_magnetization(G)
    T_values = np.arange(0, 1.0, 0.1)

    # Define the state evolution function using QDrift
    def psi(t):
        qv = QuantumVariable(G.number_of_nodes())
        U(qv, t, samples=N)
        return qv

    # Compute magnetization expectation values over time
    @jaspify(terminal_sampling=True)
    def main(T_values):
        
        M_values = jnp.zeros(len(T_values))
        for i in range(len(T_values)):
            ev_M = M.expectation_value(psi, precision=0.01)(T_values[i])
            M_values = M_values.at[i].set(ev_M)

        return M_values
    
    M_values = main(T_values) 

    assert M_values[0] - M_values[-1] > 0.5