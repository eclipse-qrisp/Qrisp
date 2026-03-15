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
import numpy.typing as npt
from numpy.polynomial import Chebyshev
from qrisp.algorithms.gqsp.gqsvt import GQSVT
from qrisp.block_encodings import BlockEncoding
from typing import Callable


def dalzell_inversion(A: BlockEncoding, prep_b: Callable, t: float, eps: float, kappa: float) -> BlockEncoding:
    r"""
    Performs the `Dalzell quantum algorithm <https://arxiv.org/pdf/2406.12086>`_
    to solve the Quantum Linear System Problem (QSLP) $A\vec{x}=\vec{b}$, using kernel reflection.
    When applied to a state $\ket{0}$, the algorithm prepares a state $\tilde{x}\propto A^{-1}\ket{b}$
    within target precision $\epsilon$ of the ideal solution $\ket{x}$.

    .. warning::

        The returned BlockEncoding must be applied to operands in state $\ket{0}$ (and not in state $\ket{b}$).

    Parameters
    ----------
    A : BlockEncoding
        The block-encoded matrix to be inverted. It is assumed that
        the eigenvalues of $A/\alpha$ lie within $D_{\kappa} = [-1, -1/\kappa] \cup [1/\kappa, 1]$.
    prep_b : Callable
        A function ``prep_b(*operands)`` preparing the right hand side $\ket{b}$.
    t : float
        An estimate $t$ for the norm $\|x\|_2=\|(A/\alpha)^{-1}b\|_2$ for normalized $\|b\|_2=1$.
        The success probability depends on the the ratio $t/\|x\|_2$. 
        The optimal choice is $t=\|x\|_2$. The estimate must lie in the interval $[1, \kappa]$.
    eps : float
        The target precision $\epsilon$.
    kappa : float
        An upper bound for the condition number $\kappa$ of $A$.
        This value defines the "gap" around zero such that all sigular values (eigenvalues) of $A/\alpha$
        lie in the domain $D_{\kappa}$.

    Returns
    -------
    BlockEncoding
        A new BlockEncoding instance representing the kernel reflection operator.

    Notes
    -----
    - **Complexity**: The query complexitiy of the algorithm (determined by the polynomial degree) scales as $\mathcal O(\kappa\log(1/\epsilon))$.
    - The polynomial is applied to operator $G_t=(\mathbb I - \ket{b'}\bra{b'})A_t$.
      Each application of $G_t$ requires 1 call to the block-encoding oracle for $A$
      and 2 calls to the state preparation oracle for $b$. 
      Hence, in contrast to other :meth:`inversion` methods that only prepare the initial state once, 
      the overall circuit complexity scales multiplicatively with the complexity of the state preparation oracle.

    Examples
    --------

    ::
    
        import numpy as np

        A = np.array([[ 0.78, -0.01, -0.16, -0.1 ],
            [-0.01,  0.57, -0.03,  0.08],
            [-0.16, -0.03,  0.69, -0.15],
            [-0.1 ,  0.08, -0.15,  0.88]])

        b = np.array([1, 1, 1, 1])

        kappa = np.linalg.cond(A)
        print("Condition number of A: ", kappa)
        # Condition number of A:  2.020268873491503

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.gqsp import dalzell_inversion

        BA = BlockEncoding.from_array(A)

        def prep_b(operand):
            prepare(operand, b)

        BA_inv = dalzell_inversion(BA, prep_b, 2.0, 0.01, 3.0)

        # Prepares operand variable in state |0>
        def operand_prep():
            operand = QuantumFloat(2)
            return operand

        @terminal_sampling
        def main():
            operand = operand_prep()
            ancillas = BA_inv.apply(operand)
            return operand, *ancillas

        res_dict = main()

        # Post-selection on ancillas being in |0> state
        filtered_dict = {k[0]: p for k, p in res_dict.items() \
                        if all(x == 0 for x in k[1:])}
        success_prob = sum(filtered_dict.values())
        print("Success probability:", success_prob)
        filtered_dict = {k: p / success_prob for k, p in filtered_dict.items()}

        amps = np.sqrt([filtered_dict.get(i, 0) for i in range(len(b))])

    ::

        c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)

        print("QUANTUM SIMULATION\n", amps, "\nCLASSICAL SOLUTION\n", c)
        # QUANTUM SIMULATION
        # [0.51718522 0.43564501 0.60746181 0.41680095] 
        # CLASSICAL SOLUTION
        # [0.51643917 0.4380885  0.60597747 0.41732523]

    """
    from qrisp import h, x, control, QuantumBool

    def prep_b_ext(*args):
        ext = args[0]
        ops_A = args[1:]
        h(ext)
        with control(ext[0], ctrl_state=0):
            prep_b(*ops_A)
            
    # Define BlockEncoding of A_t
    # A_t = |0><0| ⊗ A + (1 / t) |1><1| ⊗ |0><0|
    P0 = BlockEncoding.from_projector(0,0)
    P10 = BlockEncoding.from_projector((1,0), (1,0))
    A_t = P0.kron(A) + (1 / t) * P10
    
    # Define BlockEncoding of the kernel projector I - |b_t><b_t|
    P = BlockEncoding.from_projector(prep_b_ext, kernel=True, num_ops=2)

    # Define BlockEncoding of G_t = (I - |b_t><b_t|) A_t
    G_t = P @ A_t

    # Kernel Reflection
    # Rescale kappa to account for larger normalization factor of A_t
    kappa_t = kappa * (A.alpha + 1.0 / t) / A.alpha
    p = _kernel_reflection_poly(1.0 / kappa_t, eps)
    KR = GQSVT(G_t, p, kind="Chebyshev", parity="even", rescale=False)

    def new_unitary(*args):
        anc_ext = args[0]
        ancs_ = args[1 : 1 + KR.num_ancs]
        args_ = args[1 + KR.num_ancs:]

        x(anc_ext)
        KR.unitary(*ancs_, anc_ext, *args_)

    new_anc_templates = [QuantumBool().template()] + KR._anc_templates
    return BlockEncoding(KR.alpha, new_anc_templates, new_unitary, num_ops=KR.num_ops-1)


def _kernel_reflection_poly(delta: float, eps: float = 1e-3) -> npt.NDArray[np.float64]:
    """
    Constructs the exact Chebyshev polynomial for the Kernel Reflection 
    Polynomial $K_{\delta, \ell}(x)$ from `Dalzell (2024) <https://arxiv.org/pdf/2406.12086>`_ Eq 62.
    
    Parameters
    ----------
    delta : float
        The spectral gap threshold ($0 < \delta < 1$).
    eps : float
        The maximum allowed error on the interval $[\delta, 1]$. Defaults to 1e-3.
    
    Returns
    -------
    ndarray
        1-D array containing the coefficients of the Chebyshev series representing the smooth, bounded 
        approximation of the kernel reflection, ordered from lowest order term to highest.
    """

    # 1. Calculate the required degree parameter ell for the
    # Kernel Reflection Polynomial to achieve a target error eps.
    # Formula 51 (Dalzell 2024, arXiv:2406.12086):
    ell = int(np.ceil((1 / (2 * delta)) * np.log(2 / eps)))

    # 2. Calculate the Kernel Reflection Polynomial.
    # Formula 62 (Dalzell 2024, arXiv:2406.12086):
    # 2.1 Define the inner quadratic mapping z(x) in the Chebyshev basis:
    # z(x) = (1 + delta^2 - 2x^2) / (1 - delta^2)
    # Using the identity x^2 = 0.5*T_0(x) + 0.5*T_2(x), write z(x) exactly as:
    c_0 = (delta**2) / (1 - delta**2)
    c_2 = -1.0 / (1 - delta**2)
    # Instantiate z(x) as a Chebyshev series object:
    z_cheb = Chebyshev([c_0, 0.0, c_2])
    
    # 2.2 Define the outer Chebyshev polynomial T_ell(z).
    coeffs_T_ell = [0.0] * ell + [1.0]
    T_ell = Chebyshev(coeffs_T_ell)
    
    # 2.3 Compose the polynomials: T_ell_z(x) = T_ell(z(x)).
    T_ell_z = T_ell(z_cheb)
    
    # 2.4 Calculate the normalization constant.
    # Evaluate T_ell at the scalar value z_0 = (1 + delta^2) / (1 - delta^2).
    z_0 = (1 + delta**2) / (1 - delta**2)
    T_ell_z0 = T_ell(z_0)
    
    # 2.5 Construct the final normalized Kernel Reflection Polynomial.
    K_poly = -1.0 + 2.0 * (T_ell_z + 1.0) / (T_ell_z0 + 1.0)
    
    return K_poly.coef
