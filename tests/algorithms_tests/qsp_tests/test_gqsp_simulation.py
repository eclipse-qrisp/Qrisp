"""********************************************************************************
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

import numpy as np

from qrisp import QuantumFloat, multi_measurement, terminal_sampling
from qrisp.block_encodings import BlockEncoding
from qrisp.operators import X, Y, Z, QubitOperator


def post_selection(res_dict, N):
    filtered_dict = {k[0]: p for k, p in res_dict.items() \
                    if all(x == 0 for x in k[1:])}
    success_prob = sum(filtered_dict.values())
    filtered_dict = {k: p / success_prob for k, p in filtered_dict.items()}
    amps = np.sqrt([filtered_dict.get(k, 0) for k in range(N)])
    return amps, success_prob


# Issue #681
def test_gqsp_hamiltonian_simulation_nested_block_encoding():
    """Test the GQSP Hamiltonian simulation via qubitization with nested block encoding. 
    Defines a 4-qubit Ising Hamiltonian H and its polynomial transformation H2, constructs block encodings for both (BE and BE2).
    Applies the polynomial transformation to the block encoding of the original Ising Hamiltonian (BE_poly).
    Compares the amplitudes obtained form Hamiltonian simulation applied to BE_poly with those obtained from the direct Hamiltonian simulation applied to BE2.
    Ensures that the amplitudes are consistent and the success probability is high.
    """

    def create_ising_hamiltonian(L, J, B):
        H = sum(-J * Z(i) * Z(i + 1) for i in range(L-1))  \
            + sum(B * X(i) for i in range(L))
        return H

    L = 4
    H = create_ising_hamiltonian(L, 0.25, 0.5)
    H2 = 0.9 * H + 0.8 * H * H * H

    BE = BlockEncoding.from_operator(H)
    BE2 = BlockEncoding.from_operator(H2)
    BE_poly = BE.poly(np.array([0.0, 0.9, 0.0, 0.8]))

    # Prepare inital system state |psi> = |0>
    def operand_prep():
        return QuantumFloat(L)

    # Prepare state|psi(t)> = e^{itH} |psi>
    def psi(t, BE):
        BE_sim = BE.sim(t=t, N=10)
        operand = operand_prep()
        ancillas = BE_sim.apply(operand) 
        return operand, *ancillas

    @terminal_sampling
    def main(t, BE):
        return psi(t, BE)

    target_amps, success_prob = post_selection(res_dict = main(0.1, BE2), N=2**L)
    assert success_prob > 0.98

    amps, success_prob = post_selection(main(0.1, BE_poly), N=2**L)
    assert success_prob > 0.98
    assert np.linalg.norm(target_amps - amps) < 1e-3


# Disabled due to long runtime
# hamiltonian_simulation is tested in block encoding sim test
"""
from qrisp import *
from qrisp.gqsp import hamiltonian_simulation
from qrisp.operators import X, Y, Z
import numpy as np
import networkx as nx
import scipy as sp


def generate_chain_graph(N):
    coupling_list = [[k,k+1] for k in range(N-1)]
    G = nx.Graph()
    G.add_edges_from(coupling_list)
    return G

def create_ising_hamiltonian(G, J, B):
    H = sum(-J*Z(i)*Z(j) for (i,j) in G.edges()) + sum(B*X(i) for i in G.nodes())
    return H

def create_magnetization(G):
    H = (1/G.number_of_nodes())*sum(Z(i) for i in G.nodes())
    return H


def sim_classical(T_values, H, M):
    M_values = []
    E_values = []

    H_mat = H.to_array()
    M_mat = M.to_array()

    def _psi(t):
        psi0 = np.zeros(2**H.find_minimal_qubit_amount())
        psi0[0] = 1

        psi = sp.linalg.expm(-1.0j*t*H_mat) @ psi0
        psi = psi / np.linalg.norm(psi)

        return psi

    for t in T_values:

        psi = _psi(t)

        magnetization = (np.conj(psi) @ M_mat @ psi).real
        energy = (np.conj(psi) @ H_mat @ psi).real
        M_values.append(magnetization)
        E_values.append(energy)
        
    return np.array(M_values), np.array(E_values)


def sim_qsp(T_values, H, M):
    M_values = []
    E_values = []

    def operand_prep():
        return QuantumVariable(H.find_minimal_qubit_amount())

    def psi(t):
        BE = hamiltonian_simulation(H, t=t, N=10)
        operand = BE.apply_rus(operand_prep)()
        return operand

    @jaspify(terminal_sampling=True)
    def magnetization(t):
        magnetization = M.expectation_value(psi, precision=0.001)(t)
        return magnetization

    @jaspify(terminal_sampling=True)
    def energy(t):
        energy = H.expectation_value(psi, precision=0.001)(t)
        return energy

    for t in T_values:
        M_values.append(magnetization(t))
        E_values.append(energy(t))
    
    return np.array(M_values), np.array(E_values)

def test_qsp_hamiltonian_simulation():

    G = generate_chain_graph(6)
    H = create_ising_hamiltonian(G, 0.25, 0.5)
    M = create_magnetization(G)
    T_values = np.arange(0.1, 1.0, 0.1)
    M_classical, E_classical = sim_classical(T_values, H, M)
    M_qsp, E_qsp = sim_qsp(T_values, H, M)

    assert np.linalg.norm(E_classical - E_qsp) < 1e-2
    assert np.linalg.norm(M_classical - M_qsp) < 5e-2
"""
