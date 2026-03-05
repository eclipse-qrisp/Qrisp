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

import numpy as np
from qrisp import *
from qrisp.block_encodings import BlockEncoding
from qrisp.operators import X, Y, Z
import scipy as sp


def test_block_encoding_sim():

    def create_ising_hamiltonian(L, J, B):
        H = sum(-J * Z(i) * Z(i + 1) for i in range(L-1))  \
            + sum(B * X(i) for i in range(L))
        return H

    L = 4
    H = create_ising_hamiltonian(L, 0.25, 0.5)
    BE = BlockEncoding.from_operator(H)

    # Prepare inital system state |psi> = |0>
    def operand_prep():
        return QuantumFloat(L)

    # Prepare state|psi(t)> = e^{itH} |psi>
    def psi(t):
        BE_sim = BE.sim(t=t, N=8)
        operand = BE_sim.apply_rus(operand_prep)()
        return operand

    @terminal_sampling
    def main(t):
        return psi(t)

    res_dict = main(0.5)
    amps = np.sqrt([res_dict.get(key, 0) for key in range(2 ** L)])

    # Compare to classical solution
    H_mat = H.to_array()
    # Prepare state|psi(t)> = e^{itH} |psi>
    def psi_(t):
        # Prepare inital system state |psi> = |0>
        psi0 = np.zeros(2**H.find_minimal_qubit_amount())
        psi0[0] = 1

        psi = sp.linalg.expm(-1.0j * t * H_mat) @ psi0
        psi = psi / np.linalg.norm(psi)
        return psi

    c = np.abs(psi_(0.5))
    assert np.linalg.norm(c - amps) < 1e-6