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

from qrisp import QuantumVariable, x, QPE
from qrisp.operators import X, Y, Z, A, C, P0, P1
import numpy as np
from numpy.linalg import norm

def test_qubit_hamiltonian_commutator():

    def verify_commutator(H1, H2):
        commutator_hamiltonian = H1.commutator(H2)
        
        commutator_matrix = commutator_hamiltonian.to_sparse_matrix(factor_amount = 3).todense()
        
        H1_matrix = H1.to_sparse_matrix(factor_amount = 3).todense()
        H2_matrix = H2.to_sparse_matrix(factor_amount = 3).todense()
        
        assert norm(commutator_matrix - (np.dot(H1_matrix,H2_matrix) - np.dot(H2_matrix,H1_matrix))) < 1E-5

    operator_list = [lambda x : 1, X, Y, Z, A, C, P0, P1]

    counter = 0
    operator_list = [lambda x : 1, X, Y, Z, A, C, P0, P1]
    for O0 in operator_list: 
        for O1 in operator_list:
            for O2 in operator_list:
                for O3 in operator_list:
                    H1 = O0(0)*O1(1)
                    H2 = O2(0)*O3(1)
                    
                    if 1 in [H1, H2]:
                        continue
                    verify_commutator(H1, H2)
                    counter += 1

