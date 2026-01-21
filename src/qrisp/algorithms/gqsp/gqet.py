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
from qrisp import (
    QuantumFloat,
    conjugate,
)
from qrisp.operators import QubitOperator
from qrisp.alg_primitives.prepare import prepare
from qrisp.alg_primitives.reflection import reflection
from qrisp.algorithms.gqsp.gqsp import GQSP
from qrisp.algorithms.gqsp.helper_functions import poly2cheb, cheb2poly
from qrisp.algorithms.quantum_backtracking.backtracking_tree import psi_prep
from qrisp.algorithms.cks import CKS_parameters, cheb_coefficients
from qrisp.jasp import qache, jrange
import jax
import jax.numpy as jnp


def GQET(qarg, H, p, kind="Polynomial"):
    r"""
    Performs `Generalized Quantum Eigenvalue Transform <https://arxiv.org/pdf/2312.00723>`_.
    Applies polynomial transformations on the eigenvalues of a Hermitian operator.

    The Quantum Eigenvalue Transform is described as follows:
    
    * Given a Hermitian operator $H=\sum_i\lambda_i\ket{\lambda_i}\bra{\lambda_i}$ where $\lambda_i\in\mathbb R$ are the eigenvalues for the eigenstates $\ket{\lambda_i}$, 
    * A quantum state $\ket{\psi}=\sum_i\alpha_i\ket{\lambda_i}$ where $\alpha_i\in\mathbb C$ are the amplitudes for the eigenstates $\ket{\lambda_i}$, 
    * A (complex) polynomial $p(z)$,

    this transformation prepares a state proportional to

    .. math::

        p(H)\ket{\psi}=\sum_i p(\lambda_i)\ket{\lambda_i}\bra{\lambda_i}\sum_j\alpha_j\ket{\lambda_j}=\sum_i p(\lambda_i)\alpha_i\ket{\lambda_i}

    Parameters
    ----------
    qarg : QuantumVariable
        The QuantumVariable representing the state to apply the GQET on.
    H : QubitOperator
        The Hermitian operator.
    p : ndarray
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.
    kind : str, optinal
        The kind of ``p``. Available are ``"Polynomial"`` or ``"Chebyshev"``. 
        The default is ``"Polynomial"``.

    Returns
    -------
    qbl : QuantumBool
        Auxiliary variable after GQSP protocol. Must be measured in state $\ket{0}$.
    case : QuantumFloat
        Auxiliary variable after GQSP protocol. Must be measured in state $\ket{0}$.

    Examples
    --------

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

    ::

        poly = np.array([1., 2., 1.])

        @RUS
        def transformed_psi_prep():

            operand = psi_prep()

            qbl, case = GQET(operand, H, poly, kind="Polynomial")

            success_bool = (measure(qbl) == 0) & (measure(case) == 0)
            return success_bool, operand

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

    # Rescaling of the polynomial to account for scaling factor alpha of pauli block-encoding
    H = H.hermitize().to_pauli()
    _, coeffs = H.unitaries()
    alpha = np.sum(np.abs(coeffs))
    scaling_exponents = np.arange(len(p))
    scaling_factors = np.power(alpha, scaling_exponents)

    # Convert to Polynomial for rescaling
    if kind=="Chebyshev":
        p = cheb2poly(p)

    p = p * scaling_factors

    p = poly2cheb(p)

    U, state_prep, n = H.pauli_block_encoding()

    # Qubitization step: RU^k is a block-encoding of T_k(H)
    def RU(case, operand):
        U(case, operand)
        reflection(case, state_function=state_prep)

    case = QuantumFloat(n)

    with conjugate(state_prep)(case):
        qbl = GQSP([case, qarg], RU, p, k=0)

    return qbl, case


def GQET_inversion(A, b, eps, kappa = None):
    """

    """

    if isinstance(A, tuple) and len(A) == 3:
        kappa = kappa
    else:
        kappa = np.linalg.cond(A)

    H = QubitOperator.from_matrix(A, reverse_endianness=True)

    j_0, beta = CKS_parameters(A, eps, kappa)
    p_odd = cheb_coefficients(j_0, beta)

    p_odd = p_odd * (-1) ** np.arange(len(p_odd))

    p = np.zeros(2 * len(p_odd))
    p[1::2] = p_odd    
    p = np.array(p)

    H = H.hermitize().to_pauli()
    _, coeffs = H.unitaries()
    alpha = np.sum(np.abs(coeffs))
    scaling_exponents = np.arange(len(p))
    scaling_factors = np.power(1/alpha, scaling_exponents)

    p = cheb2poly(p)
    p = p * scaling_factors
    p = poly2cheb(p)

    def psi_prep():
        operand = QuantumFloat(int(np.log2(b.shape[0])))
        prepare(operand, b)
        return operand

    operand = psi_prep()

    qbl, case = GQET(operand, H, p, kind="Chebyshev")

    return operand, qbl, case