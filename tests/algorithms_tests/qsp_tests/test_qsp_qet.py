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
from numpy.polynomial import Polynomial
import pytest
from qrisp import *
from qrisp.gqsp import GQET
from qrisp.operators import X, Y, Z
from qrisp.vqe.problems.heisenberg import create_heisenberg_init_function


def generate_1D_chain_graph(L):
    graph = nx.Graph()
    graph.add_edges_from([(k, (k+1)%L) for k in range(L-1)]) 
    return graph


def polyvalm(poly, A):
    roots = poly.roots()
    I = np.eye(A.shape[0])
    res = np.eye(A.shape[0])
    for i in range(len(roots)):
        res = (A - roots[i]*I) @ res
    return res


@pytest.mark.parametrize("L, poly", [
    (6, np.array([1., 1.])),
    (6, np.array([1., 2., 1.])),
])
def test_qsp_qet(L, poly):

    # Define Heisenberg Hamiltonian 
    G = generate_1D_chain_graph(L)
    H = sum((X(i)*X(j) + Y(i)*Y(j) + Z(i)*Z(j)) for i,j in G.edges())
    M = nx.maximal_matching(G)
    U0 = create_heisenberg_init_function(M)


    # Define initial state preparation function
    def psi_prep():
        operand = QuantumVariable(H.find_minimal_qubit_amount())
        U0(operand)
        return operand


    @RUS
    def transformed_psi_prep():
        operand = psi_prep()
        qbl, case = GQET(operand, H, poly, kind="Polynomial")
        success_bool = (measure(qbl) == 0) & (measure(case) == 0)
        return success_bool, operand
    

    @jaspify(terminal_sampling=True)
    def main(): 
        E = H.expectation_value(transformed_psi_prep, precision=0.001)()
        return E


    # Calculate the energy
    E0 = H.expectation_value(psi_prep, precision=0.001)()
    E1 = main()

    # Compare to classical values
    # Calculate energy for |psi_0>
    H_arr = H.to_array()
    psi_0 = psi_prep().qs.statevector_array()
    E0_numpy = (psi_0.conj() @ H_arr @ psi_0).real
    assert np.abs(E0 - E0_numpy) < 1e-2

    # Calculate energy for |psi> = poly(H) |psi0>
    psi = polyvalm(Polynomial(poly), H_arr) @ psi_0
    psi = psi / np.linalg.norm(psi)
    E1_numpy = (psi.conj() @ H_arr @ psi).real
    assert np.abs(E1 - E1_numpy) < 1e-2