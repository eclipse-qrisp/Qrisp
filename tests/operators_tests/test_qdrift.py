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
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from qrisp import QuantumVariable
from qrisp.operators import X, Z, QubitOperator

# Helper functions
def generate_chain_graph(N):
    G = nx.Graph()
    G.add_edges_from((k, k+1) for k in range(N-1))
    return G

def create_ising_hamiltonian(G, J, B):
    # H = -J ∑ Z_i Z_{i+1} - B ∑ X_i
    H = sum(-J * Z(i)*Z(j) for (i,j) in G.edges()) + sum(B * X(i) for i in G.nodes())
    return H

def create_magnetization(G):
    return (1.0 / G.number_of_nodes()) * sum(Z(i) for i in G.nodes())

# Simulation setup
N=1000
G = generate_chain_graph(6)
H = create_ising_hamiltonian(G, J=1.0, B=1.0)
M = create_magnetization(G)
T_values = np.arange(0, 2, 0.05)

# Define the state evolution function using QDRIFT
def psi(t, use_arctan=True):
    qv = QuantumVariable(G.number_of_nodes())
    H.qdrift(qv, t, N=N, use_arctan=use_arctan)
    return qv

# Compute magnetization expectation values over time
M_values = []
for t in T_values:
    ev_M = M.expectation_value(psi, precision_expectation_value)
    M_values.append(float(ev_M(t)))


plt.scatter(T_values, M_values, color='#6929C4', marker="o", linestyle="solid", s=20, label=r"Ising chain")
plt.xlabel(r"Evolution time $T$", fontsize=15, color="#444444")
plt.ylabel(r"Magnetization $\langle M \rangle$", fontsize=15, color="#444444")
plt.legend(fontsize=15, labelcolor="#444444")
plt.tick_params(axis='both', labelsize=12)
plt.grid()
plt.show()