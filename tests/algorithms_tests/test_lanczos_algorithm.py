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
import numpy as np
from qrisp import QuantumVariable
from qrisp.lanczos import lanczos_alg
from qrisp.operators import X, Y, Z
from qrisp.vqe.problems.heisenberg import create_heisenberg_init_function


def test_lanczos_algorithm():

    L = 6
    G = nx.Graph()
    G.add_edges_from([(k, (k+1) % L) for k in range(L - 1)])

    # Define Hamiltonian e.g. Heisenberg with custom couplings
    H = (1/4)*sum((X(i)*X(j) + Y(i)*Y(j) + 0.5*Z(i)*Z(j)) for i,j in G.edges())

    energy_classical = H.ground_state_energy()
    print(f"Ground state energy: {energy_classical}")

    # Prepare initial state function (tensor product of singlets)
    M = nx.maximal_matching(G)
    U_singlet = create_heisenberg_init_function(M)

    def init_state():
        qv = QuantumVariable(H.find_minimal_qubit_amount())
        U_singlet(qv)
        return qv

    D = 6  # Krylov dimension
    energy, info = lanczos_alg(H, D, init_state, show_info=True)
    print(f"Ground state energy estimate: {energy}")

    assert np.abs(energy_classical-energy) < 1e-2