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
    Quantum Linear System solver via Generalized Quantum Eigenvalue Transformation (GQET).

    This function implements a GQET-based quantum circuit that approximates the matrix
    inversion operation :math:`A^{-1}` applied to an input quantum state :math:`\\ket{b}`.
    Using a Chebyshev polynomial approximation to the inverse function within error
    tolerance ``eps``, it constructs a generalized eigenvalue transformation that
    effectively solves the Quantum Linear System Problem (QLSP)
    :math:`A \\vec{x} = \\vec{b}`.

    The block-encoding of the matrix :math:`A` can be provided either explicitly as a
    block-encoding tuple, or implicitly via a Hermitian matrix which will be
    block-encoded internally via a Pauli decomposition routine such as
    :meth:`QubitOperator.pauli_block_encoding <qrisp.operators.qubit.QubitOperator.pauli_block_encoding>`.
    In the former case, the condition number :math:`\\kappa` of :math:`A` must be
    provided explicitly.

    The inversion polynomial is constructed via
    :func:`CKS_parameters` and :func:`cheb_coefficients`, yielding an odd Chebyshev
    polynomial that approximates :math:`1/x` on the relevant spectral interval.
    This odd polynomial is then embedded into a full Chebyshev series and rescaled
    to the normalized Hamiltonian :math:`H / \\alpha` using :func:`_extend_and_rescale_cheb`.
    :math:`A^{-1} \\ket{b}`.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.csr_matrix or tuple
        Either the Hermitian matrix :math:`A` of size :math:`N \\times N` from
        the linear system :math:`A \\vec{x} = \\vec{b}`, or a 3-tuple
        ``(U, state_prep, n)`` representing a preconstructed block-encoding
        of :math:`A`. When a tuple is provided, the matrix itself is not required
        and ``kappa`` must be specified.
    b : numpy.ndarray or callable
        Either a vector :math:`\\vec{b}` of the linear system, or a
        callable that prepares the corresponding quantum state ``operand``.
    eps : float
        Target precision :math:`\epsilon`, such that the prepared state :math:`\ket{\\tilde{x}}` is within error
        :math:`\epsilon` of :math:`\ket{x}`.
    kappa : float, optional
        Condition number :math:`\\kappa` of :math:`A`. Required when ``A`` is
        a block-encoding tuple ``(U, state_prep, n)`` rather than a matrix.
        If ``A`` is provided as a matrix and ``kappa`` is ``None``, it is
        estimated as ``numpy.linalg.cond(A)``.

    Returns
    -------
    operand : QuantumVariable
        Operand variable after applying the GQET protocol.
    qbl : QuantumBool
        Auxiliary variable after GQET protocol. Must be measured in state $\ket{0}$.
    case : QuantumFloat
        Auxiliary variable after GQET protocol. Must be measured in state $\ket{0}$.

    Examples
    --------
    This example shows how to solve a QLSP using the ``GQET_inversion`` function directly,
    without employing the :ref:`RUS` protocol in Qrisp.

    First, define a Hermitian matrix :math:`A` and a right-hand side vector :math:`\\vec{b}`

    ::

        import numpy as np

        A = np.array([[0.73255474, 0.14516978, -0.14510851, -0.0391581],
                      [0.14516978, 0.68701415, -0.04929867, -0.00999921],
                      [-0.14510851, -0.04929867, 0.76587818, -0.03420339],
                      [-0.0391581, -0.00999921, -0.03420339, 0.58862043]])

        b = np.array([0, 1, 1, 1])

    Then, construct the GQET inversion circuit and perform a multi measurement:

    ::

        from qrisp.algorithms.gqet import GQET_inversion
        from qrisp import multi_measurement

        operand, qbl, case = GQET_inversion(A, b, 0.01)
        res_dict = multi_measurement([operand, qbl, case])

    This performs the measurement on all three QuantumVariables ``operand``, ``qbl``,
    and ``case``. Since the GQET (and all other LCU-based approaches) is correctly
    performed only when auxiliary QuantumVariables are measured in :math:`\\ket{0}`,
    we need to construct a new dictionary and collect the measurements when this
    condition is satisfied.

    ::

        new_dict = dict()
        success_prob = 0

        for key, prob in res_dict.items():
            # key = (operand_outcome, qbl_outcome, case_outcome)
            if key[1] == 0 and key[2] == 0:
                new_dict[key[0]] = prob
                success_prob += prob

        for key in new_dict.keys():
            new_dict[key] = new_dict[key] / success_prob

        for k, v in new_dict.items():
            new_dict[k] = v**0.5

    Finally, compare the quantum simulation result with the classical solution:

    ::

        q = np.array([new_dict.get(key, 0) for key in range(len(b))])
        c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)

        print("QUANTUM SIMULATION\\n", q, "\\nCLASSICAL SOLUTION\\n", c)
        # QUANTUM SIMULATION
        # [0.02908157 0.55499333 0.53005093 0.64045506] 
        # CLASSICAL SOLUTION
        # [0.02944539 0.55423278 0.53013239 0.64102936]

    This approach enables execution of the compiled GQET inversion circuit on
    :ref:`compatible quantum hardware or simulators <BackendInterface>`.
    """

    if isinstance(A, tuple) and len(A) == 3:
        kappa = kappa
    else:
        kappa = np.linalg.cond(A)

    H = QubitOperator.from_matrix(A, reverse_endianness=True)

    j_0, beta = CKS_parameters(A, eps, kappa)
    p_odd = cheb_coefficients(j_0, beta)

    p_odd = p_odd * (-1) ** np.arange(len(p_odd))

    H = H.hermitize().to_pauli()
    _, coeffs = H.unitaries()
    alpha = np.sum(np.abs(coeffs))

    # Extend the odd Chebyshev polynomial and rescale it for the normalized
    # Hamiltonian H / alpha.
    p = _extend_and_rescale_cheb(p_odd, alpha)

    if callable(b):
        b_prep = b

    else:
        def b_prep():
            operand = QuantumFloat(int(np.log2(b.shape[0])))
            prepare(operand, b)
            return operand

    operand = b_prep()

    qbl, case = GQET(operand, H, p, kind="Chebyshev")

    return operand, qbl, case

def _extend_and_rescale_cheb(p_odd, alpha):
    
    # Initialize full Chebyshev coefficient array.
    # We place the odd coefficients at indices 1, 3, 5, ... and keep even indices zero.
    p = np.zeros(2 * len(p_odd))
    p[1::2] = p_odd

    # Convert to Polynomial basis.
    p = cheb2poly(p)

    # Apply the rescaling x -> x / alpha in polynomial basis:

    scaling_exponents = np.arange(len(p))
    scaling_factors = (1.0 / alpha) ** scaling_exponents
    p_scaled = p * scaling_factors

    # Convert back to Chebyshev basis.
    p_scaled = poly2cheb(p_scaled)
    return p_scaled