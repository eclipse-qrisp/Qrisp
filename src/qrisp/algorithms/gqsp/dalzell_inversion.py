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

from collections.abc import Callable
import numpy as np
import numpy.typing as npt
from numpy.polynomial import Chebyshev

from qrisp.algorithms.gqsp.qsvt import QSVT
from qrisp.block_encodings import BlockEncoding


def dalzell_inversion(A: BlockEncoding, prep_b: Callable, t: float, eps: float, kappa: float) -> BlockEncoding:
    r"""
    Performs the `Dalzell quantum algorithm <https://arxiv.org/pdf/2406.12086>`_ 
    to solve the Quantum Linear System Problem (QSLP) $A\vec{x}=\vec{b}$, using kernel reflection.
    When applied to a state $\ket{0}$, the algorithm prepares a state $\tilde{x}\propto A^{-1}\ket{b}$
    within target precision $\epsilon$ of the ideal solution $\ket{x}$.
    See also `Dalzell's talk <https://www.youtube.com/watch?v=OwqhdCioj4Y>`_ for more details.

    .. warning::

        The returned BlockEncoding must be applied to operands in state $\ket{0}$ (and not in state $\ket{b}$).

    Instead of solving the system $Ax=b$ directly, we consider the augmented system $A_t x_t = b'$, i.e.,
    
    .. math::

            A_t = \begin{pmatrix} A & 0 \\ 0 & t^{-1} \end{pmatrix} \begin{pmatrix} x \\ t \end{pmatrix} = \begin{pmatrix} b \\ 1 \end{pmatrix} = b'

    Equivalently, we can write the augmented system as $A\ket{x_t}=\ket{b'}$ where

    .. math::
    
        A_t = \ket{0}_a\bra{0}_a \otimes A + t^{-1} \ket{1}_a\bra{1}_a \otimes \ket{0}_s\bra{0}_s,\\
        \ket{b'} \propto \ket{0}_a\ket{b}_s + \ket{1}_a\ket{0}_s, \quad \ket{x_t} \propto \|x\|_2\ket{0}_a\ket{x}_s + t\ket{1}_a\ket{0}_s

    Here, $\ket{\cdot}_a$ denotes the state of the (1-qubit) ancilla variable, and $\ket{\cdot}_s$ denotes the state of the system variable.

    If $A_t x_t = b'$, then 
    
    .. math:: 
    
        G_tx_t=(\mathbb{I} - \ket{b'}\bra{b'})A_t x_t = (\mathbb{I} - \ket{b'}\bra{b'})\ket{b'} = 0
    
    Hence, the solution state $\ket{x_t}$ is a kernel state (sigular value 0) of the operator $G_t=(\mathbb{I} - \ket{b'}\bra{b'})A_t$.
    The **kernel reflection** operator $K(G_t)$ reflects about the solution state $\ket{x_t}$ to the augmented system, 
    and can be implemented via QSVT using a polynomial approximation of the kernel reflection function 
    $K(\sigma)\colon [0,1] \to [-1,1]$ which maps the sigular value 0 to 1 and all other singular values to -1.

    The algorithm consists of the following steps:

    - Introduce extra basis state $\ket{e_n}=\ket{1}_a\ket{0}_s$ and prepare it as the initial state.
    - Choose $t\simeq \|x\|_2$ and form the augmented linear system $A_tx_t=b'$.
    - Apply the kernel reflection operator $K(G_t)$, which **reflects about the solution state to the augmented system**.
    - Project the resulting state onto the $\ket{0}_a$ subspace to obtain $\ket{x}$.

    The projection step can be implemented by post-selecting on the ancilla qubit being in state $\ket{0}$ after applying the kernel reflection.
    The success probability of the algorithm depends on the choice of $t$. 
    If $t$ is chosen to be on the order of $\|x\|_2$, the algorithm succeeds with constant probability and only requires a constant number of repetitions.

    .. image:: /_static/chebyshev_kernel_reflection.png
       :align: center

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
        The optimal choice is $t=\|x\|_2$. The estimate must lie in the interval $[1, \kappa]$. See below for more details.
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
    - **Complexity**: The query complexitiy of the algorithm is determined by the polynomial degree and scales as $\mathcal O(\kappa\log(1/\epsilon))$.
      The success probability of the algorithm depends on the choice of $t$ relative to $\|x\|_2$: 

      - If the norm is known up to a constant factor, the algorithm succeeds with constant probability, hence only requires a constant number of repetitions.
      - If the norm is unknown, `methods for norm estimation <https://arxiv.org/pdf/2406.12086>`_ can be used to find a suitable $t$ with only a logarithmic overhead in the overall complexity.

      For further insights, see `Constant Factor Analysis of Optimal Quantum Linear Solvers in Practice <https://arxiv.org/abs/2604.22185>`_,
      and make sure to use :meth:`.resources() <qrisp.block_encodings.BlockEncoding.resources>` to compare the resources of different inversion methods in practice.
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

        @terminal_sampling
        def main():
            # Prepare operand variable in state |0>
            operand = QuantumFloat(2)
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

    Alternatively, use ``apply_rus`` to directly obtain the solution state without post-selection:
        
    ::

        @terminal_sampling
        def main():
            operand = BA_inv.apply_rus(lambda: QuantumFloat(2))()
            return operand

        res_dict = main()
        amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])
        print(amps)
        # [0.51816163 0.43295659 0.60721322 0.4187472 ]

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
    A_t = P0.kron(A) + (A.alpha / t) * P10
    
    # Define BlockEncoding of the kernel projector I - |b_t><b_t|
    P = BlockEncoding.from_projector(prep_b_ext, kernel=True, num_ops=2)

    # Define BlockEncoding of G_t = (I - |b_t><b_t|) A_t
    G_t = P @ A_t

    # Kernel Reflection
    # Rescale kappa to account for larger normalization factor of A_t
    kappa_t = kappa * (1.0 + 1.0 / t)
    p = _kernel_reflection_cheb(1.0 / kappa_t, eps)
    KR = QSVT(G_t, p, kind="Chebyshev", parity="even", rescale=False)

    def new_unitary(*args):
        anc_ext = args[0]
        ancs_ = args[1 : 1 + KR.num_ancs]
        args_ = args[1 + KR.num_ancs:]

        x(anc_ext)
        KR.unitary(*ancs_, anc_ext, *args_)

    new_anc_templates = [QuantumBool().template()] + KR._anc_templates
    return BlockEncoding(KR.alpha, new_anc_templates, new_unitary, num_ops=KR.num_ops-1)


def _kernel_reflection_cheb(delta: float, eps: float = 1e-3) -> npt.NDArray[np.float64]:
    """
    Constructs the Chebyshev polynomial for the Kernel Reflection 
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
    K_cheb = -1.0 + 2.0 * (T_ell_z + 1.0) / (T_ell_z0 + 1.0)
    
    return K_cheb.coef
