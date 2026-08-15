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

from typing import TYPE_CHECKING

from qrisp.block_encodings.block_encoding_base import BlockEncoding

if TYPE_CHECKING:
    from jax.typing import ArrayLike


def apply_sim(self, t: "ArrayLike" = 1, N: int = 1) -> BlockEncoding:
    r"""Returns a BlockEncoding approximating Hamiltonian simulation of the operator.

    For a block-encoded Hamiltonian $H$, this method returns a BlockEncoding of an approximation of
    the unitary evolution operator $e^{-itH}$ for a given time $t$.

    The approximation is based on the Jacobi-Anger expansion into Bessel functions 
    of the first kind ($J_n$):

    .. math ::

        e^{-it\cos(\theta)} \approx \sum_{n=-N}^{N}(-i)^nJ_n(t)e^{in\theta}

    Parameters
    ----------
    t : ArrayLike
        The scalar evolution time $t$. The default is 1.0.
    N : int
        The truncation order $N$ of the expansion. A higher order provides 
        better approximation for larger $t$ or higher precision requirements. 
        Default is 1.

    Returns
    -------
    BlockEncoding
        A new BlockEncoding instance representing an approximation of the unitary $e^{-itH}$.

    Notes
    -----
    - **Precision**: The truncation error scales (decreases) super-exponentially with $N$. 
      For a fixed $t$, choosing $N > |t|$ ensures rapid convergence.
    - **Normalization**: The resulting operator is nearly unitary, meaning its 
      block-encoding normalization factor $\alpha$ will be close to 1.

    References
    ----------
    - Low & Chuang (2019) `Hamiltonian Simulation by Qubitization <https://quantum-journal.org/papers/q-2019-07-12-163/>`_.
    - Motlagh & Wiebe (2025) `Generalized Quantum Signal Processing <https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368>`_.

    Examples
    --------
    Generate an Ising Hamiltonian $H$ and apply Hamiltonian simulation $e^{-itH}$ to the inital system state $\ket{0}$.

    ::

        # For larger systems, restart the kernel and adjust simulator precision
        # import os
        # os.environ["QRISP_SIMULATOR_FLOAT_THRESH"] = "1e-10"

        import numpy as np
        from qrisp import QuantumFloat, terminal_sampling
        from qrisp.block_encodings import BlockEncoding
        from qrisp.operators import X, Y, Z

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
        amps = np.sqrt([res_dict.get(i, 0) for i in range(2 ** L)])
        print(amps)
        #[0.88288218 0.224682   0.22269639 0.05723058 0.22269632 0.05669449                   
        # 0.0570588  0.01457775 0.22468192 0.05717859 0.05669445 0.0145699
        # 0.05723059 0.01456992 0.01457775 0.00372438]

    Finally, compare the quantum simulation result with the classical solution:

    ::

        import scipy as sp

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
        print(c)
        #[0.88288217 0.22468197 0.22269638 0.05723056 0.22269638 0.05669446
        # 0.05705877 0.01457772 0.22468197 0.0571786  0.05669446 0.01456988
        # 0.05723056 0.01456988 0.01457772 0.00372439]

    """
    from qrisp.algorithms.gqsp import hamiltonian_simulation

    return hamiltonian_simulation(self, t, N)
