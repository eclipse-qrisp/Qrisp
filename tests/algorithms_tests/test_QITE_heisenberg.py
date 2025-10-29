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
import numpy as np
import sympy as sp

from qrisp import QuantumVariable
from qrisp.operators import X, Y, Z
from qrisp.qite import QITE
from qrisp.vqe.problems.heisenberg import create_heisenberg_init_function


def test_qite_heisenberg():

    # Create a graph
    N = 4
    G = nx.Graph()
    G.add_edges_from([(k, k + 1) for k in range(N - 1)])

    def create_heisenberg_hamiltonian(G):
        H = sum(X(i) * X(j) + Y(i) * Y(j) + Z(i) * Z(j) for (i, j) in G.edges())
        return H

    H = create_heisenberg_hamiltonian(G)
    E_tar = H.ground_state_energy()

    M = nx.maximal_matching(G)
    U_0 = create_heisenberg_init_function(M)

    qv = QuantumVariable(N)
    U_0(qv)
    E_0 = H.get_measurement(qv)

    def exp_H(qv, t):
        H.trotterization(method="commuting")(qv, t, 5)

    steps = 4
    s_values = np.linspace(0.01, 0.3, 10)

    theta = sp.Symbol("theta")
    optimal_s = [theta]
    optimal_energies = [E_0]

    for k in range(1, steps + 1):

        # Perform k steps of QITE
        qv = QuantumVariable(N)
        QITE(qv, U_0, exp_H, optimal_s, k)
        qc = qv.qs.compile()

        # Find optimal evolution time
        energies = [
            H.get_measurement(
                qv,
                subs_dic={theta: s_},
                precompiled_qc=qc,
                diagonalisation_method="commuting",
            )
            for s_ in s_values
        ]
        index = np.argmin(energies)
        s_min = s_values[index]

        optimal_s.insert(-1, s_min)
        optimal_energies.append(energies[index])

    assert np.abs(E_tar - optimal_energies[-1]) < 1e-1
