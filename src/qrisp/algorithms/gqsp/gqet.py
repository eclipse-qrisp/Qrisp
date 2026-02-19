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
import jax.numpy as jnp
from qrisp import QuantumBool
from qrisp.algorithms.gqsp.gqsp import GQSP
from qrisp.algorithms.gqsp.gqsp_angles import gqsp_angles
from qrisp.algorithms.gqsp.helper_functions import poly2cheb, _rescale_poly
from qrisp.block_encodings import BlockEncoding
from qrisp.operators import QubitOperator, FermionicOperator
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from jax.typing import ArrayLike


def GQET(
    H: BlockEncoding | FermionicOperator | QubitOperator,
    p: "ArrayLike",
    kind: Literal["Polynomial", "Chebyshev"] = "Polynomial",
    rescale: bool = True,
) -> BlockEncoding:
    r"""
    Returns a BlockEncoding representing a polynomial transformation of the operator via `Generalized Quantum Eigenvalue Transform <https://arxiv.org/pdf/2312.00723>`_.

    For a block-encoded **Hermitian** operator $H$ and a (complex) polynomial $p(z)$, this method returns
    a BlockEncoding of the operator $p(H)$.

    The Quantum Eigenvalue Transform is described as follows:

    * Given a Hermitian operator $H=\sum_i\lambda_i\ket{\lambda_i}\bra{\lambda_i}$ where $\lambda_i\in\mathbb R$ are the eigenvalues for the eigenstates $\ket{\lambda_i}$,
    * A quantum state $\ket{\psi}=\sum_i\alpha_i\ket{\lambda_i}$ where $\alpha_i\in\mathbb C$ are the amplitudes for the eigenstates $\ket{\lambda_i}$,
    * A (complex) polynomial $p(z)$,

    this transformation prepares a state proportional to

    .. math::

        p(H)\ket{\psi}=\sum_i p(\lambda_i)\ket{\lambda_i}\bra{\lambda_i}\sum_j\alpha_j\ket{\lambda_j}=\sum_i p(\lambda_i)\alpha_i\ket{\lambda_i}

    Parameters
    ----------
    H : BlockEncoding | FermionicOperator | QubitOperator
        The Hermitian operator to be transformed.
    p : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.
    kind : {"Polynomial", "Chebyshev"}
        The basis in which the coefficients are defined.

        - ``"Polynomial"``: $p(x) = \sum c_i x^i$

        - ``"Chebyshev"``: $p(x) = \sum c_i T_i(x)$, where $T_i$ are Chebyshev polynomials of the first kind.

        Default is ``"Polynomial"``.
    rescale : bool
        If True (default), the method returns a block-encoding of $p(H)$.
        If False, the method returns a block-encoding of $p(H/\alpha)$ where $\alpha$ is the normalization factor for the block-encoding of the operator $H$.

    Returns
    -------
    BlockEncoding
        A new BlockEncoding instance representing the transformed operator $p(H)$.

    Examples
    --------

    Define a Heisenberg Hamiltonian and apply a polynomial $p(H)$ to an initial system state.

    ::

        from qrisp import *
        from qrisp.gqsp import *
        from qrisp.operators import X, Y, Z
        from qrisp.vqe.problems.heisenberg import create_heisenberg_init_function
        import numpy as np
        import networkx as nx


        def generate_1D_chain_graph(L):
            graph = nx.Graph()
            graph.add_edges_from([(k, (k+1)%L) for k in range(L-1)])
            return graph


        # Define Heisenberg Hamiltonian
        L = 10
        G = generate_1D_chain_graph(L)
        H = sum((X(i)*X(j) + Y(i)*Y(j) + Z(i)*Z(j)) for i,j in G.edges())

        M = nx.maximal_matching(G)
        U0 = create_heisenberg_init_function(M)


        # Define initial state preparation function
        def psi_prep():
            operand = QuantumVariable(H.find_minimal_qubit_amount())
            U0(operand)
            return operand


        # Calculate the energy
        E = H.expectation_value(psi_prep, precision=0.001)()
        print(E)

    Define a polynomial and use GQET to obtain a BlockEncoding of $p(H)$.

    ::

        poly = jnp.array([1., 2., 1.])
        BE = GQET(H, poly, kind="Polynomial")

        def transformed_psi_prep():
            operand = BE.apply_rus(psi_prep)()
            return operand

    Calculate the expectation value of $H$ for the transformed state $p(H)\ket{\psi}$.

    ::

        @jaspify(terminal_sampling=True)
        def main():
            E = H.expectation_value(transformed_psi_prep, precision=0.001)()
            return E

        print(main())
        # -16.67236953920006

    Finally, we compare the quantum simulation to the classically calculated result:

    ::

        # Calculate energy for |psi_0>
        H_arr = H.to_array()
        psi_0 = psi_prep().qs.statevector_array()
        E_0 = (psi_0.conj() @ H_arr @ psi_0).real
        print("E_0", E_0)

        # Calculate energy for |psi> = poly(H) |psi0>
        I = np.eye(H_arr.shape[0])
        psi = (I + H_arr) @ (I + H_arr) @ psi_0
        psi = psi / np.linalg.norm(psi)
        E = (psi.conj() @ H_arr @ psi).real
        print("E", E)

    """

    ALLOWED_KINDS = {"Polynomial", "Chebyshev"}
    if kind not in ALLOWED_KINDS:
        raise ValueError(
            f"Invalid kind specified: '{kind}'. "
            f"Allowed kinds are: {', '.join(ALLOWED_KINDS)}"
        )

    if isinstance(H, (QubitOperator, FermionicOperator)):
        H = BlockEncoding.from_operator(H)

    # Rescaling of the polynomial to account for scaling factor alpha of block-encoding
    if rescale:
        p = _rescale_poly(H.alpha, p, kind=kind)
    if kind == "Polynomial":
        p = poly2cheb(p)

    BE_walk = H.qubitization()

    angles, new_alpha = gqsp_angles(p)

    def new_unitary(*args):
        GQSP(args[0], *args[1:], unitary=BE_walk.unitary, angles=angles)

    new_anc_templates = [QuantumBool().template()] + BE_walk._anc_templates
    return BlockEncoding(
        new_alpha,
        new_anc_templates,
        new_unitary,
        num_ops=BE_walk.num_ops,
        is_hermitian=False,
    )
