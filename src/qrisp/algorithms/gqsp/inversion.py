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
from qrisp.algorithms.gqsp.helper_functions import poly2cheb, cheb2poly
from qrisp.algorithms.cks import CKS_parameters, cheb_coefficients
from qrisp.algorithms.gqsp.qet import QET
from qrisp.block_encodings import BlockEncoding


def inversion(A: BlockEncoding, eps: float, kappa: float) -> BlockEncoding:
    r"""
    Quantum Linear System Solver via Quantum Eigenvalue Transformation (QET).
    Returns a BlockEncoding approximating the matrix inversion of the operator.

    For a block-encoded matrix $A$, this function returns a BlockEncoding of an 
    operator $\tilde{A}^{-1}$ such that $\|\tilde{A}^{-1} - A^{-1}\| \leq \epsilon$. 
    The inversion is implemented via Quantum Eigenvalue Transformation (QET)         
    using a polynomial approximation of $1/x$ over the domain $D_{\kappa} = [-1, -1/\kappa] \cup [1/\kappa, 1]$.

    Parameters
    ----------
    A : BlockEncoding
        The block-encoded Hermitian matrix to be inverted. It is assumed that 
        the eigenvalues of $A$ lie within $D_{\kappa}$.
    eps : float
        The target precision $\epsilon$.
    kappa : float
        An upper bound for the condition number $\kappa$ of $A$. 
        This value defines the "gap" around zero where the function $1/x$ is not approximated.

    Returns
    -------
    BlockEncoding
        A new BlockEncoding instance representing an approximation of the inverse $A^{-1}$.

    Notes
    -----
    - **Complexity**: The polynomial degree scales as :math:`\mathcal{O}(\kappa \log(\kappa/\epsilon))`.

    References
    ----------
    - Childs et. al (2017) `Quantum algorithm for systems of linear equations with exponentially improved dependence on precision <https://arxiv.org/pdf/1511.02306>`_.

    Examples
    --------

    Define a QSLP and solve it using :meth:`inversion`.

    First, define a Hermitian matrix $A$ and a right-hand side vector $\vec{b}$.

    ::

        import numpy as np

        A = np.array([[0.73255474, 0.14516978, -0.14510851, -0.0391581],
                    [0.14516978, 0.68701415, -0.04929867, -0.00999921],
                    [-0.14510851, -0.04929867, 0.76587818, -0.03420339],
                    [-0.0391581, -0.00999921, -0.03420339, 0.58862043]])

        b = np.array([0, 1, 1, 1])

        kappa = np.linalg.cond(A)
        print("Condition number of A: ", kappa)
        # Condition number of A:  1.8448536035491883

    Generate a block-encoding of $A$ and use :meth:`inversion` to find a block-encoding approximating $A^{-1}$.

    ::

        from qrisp import *
        from qrisp.algorithms.gqsp import inversion
        from qrisp.operators import QubitOperator

        H = QubitOperator.from_matrix(A, reverse_endianness=True)
        BA = H.pauli_block_encoding()

        BA_inv = inversion(BA, 0.01, 2)

        # Prepares operand variable in state |b>
        def prep_b():
            operand = QuantumVariable(2)
            prepare(operand, b)
            return operand

        @terminal_sampling
        def main():
            operand = BA_inv.apply_rus(prep_b)()
            return operand

        res_dict = main()

    Finally, compare the quantum simulation result with the classical solution:

    ::

        for k, v in res_dict.items():
            res_dict[k] = v**0.5

        q = np.array([res_dict.get(key, 0) for key in range(len(b))])
        c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)

        print("QUANTUM SIMULATION\n", q, "\nCLASSICAL SOLUTION\n", c)
        # QUANTUM SIMULATION
        # [0.02844496 0.55538449 0.53010186 0.64010231] 
        # CLASSICAL SOLUTION
        # [0.02944539 0.55423278 0.53013239 0.64102936]

    """

    # The inversion polynomial is constructed using CKS_parameters and cheb_coefficients.
    # Since approximating 1/x over the relevant spectral interval [-1, -1/kappa] + [1/kappa,1] 
    # requires an odd Chebyshev series, cheb_coefficients returns an array containing only the odd-degree coefficients.
    # To remain compatible with the QET interface, this array is expanded into a full 
    # Chebyshev series by padding even-degree terms with zeros.
    j_0, beta = CKS_parameters(A, eps, kappa)
    p_odd = cheb_coefficients(j_0, beta)
    p_odd = p_odd * (-1) ** np.arange(len(p_odd))
    p = np.zeros(2 * len(p_odd))
    p[1::2] = p_odd

    # Set _rescale=False to apply p(A/α) instead of p(A).
    A_inv = QET(A, p, kind="Chebyshev", rescale=False)

    # Adjust scaling factor since (A/α)^{-1} = αA^{-1}.
    A_inv.alpha = A_inv.alpha / A.alpha
    return A_inv