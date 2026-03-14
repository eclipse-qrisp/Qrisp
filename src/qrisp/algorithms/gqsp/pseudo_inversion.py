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
from qrisp.algorithms.cks import cks_coeffs, cks_params
from qrisp.algorithms.gqsp.gqsvt import GQSVT
from qrisp.algorithms.gqsp.qet import QET
from qrisp.algorithms.gqsp.helper_functions import chebyshev_approx
from qrisp.block_encodings import BlockEncoding
from scipy.special import erf


def pseudo_inversion(
    A: BlockEncoding,
    eps: float,
    kappa: float,
    delta: float = None,
) -> BlockEncoding:
    r"""
    Returns a BlockEncoding approximating the threshold matrix pseudo inversion of the operator.

    Let $A$ be a matrix with `Singular Value Decomposition <https://en.wikipedia.org/wiki/Singular_value_decomposition>`_

    .. math::

        A = U\Sigma V^{\dagger} = \sum_i \sigma_i u_iv_i^{\dagger}

    where $\sigma_i$ are the singular values, $u_i$ are the left singular vectors, 
    and $v_i$ are the right singular vectors.

    Given a threshold $\theta>0$, define a scalar thresholding function $f_{\theta}(\sigma)$ that 
    acts on the singular values:

    .. math::

        f_{\theta}(\sigma)=\begin{cases} \frac{1}{\sigma} & \sigma\geq\theta \\ 0 & \sigma<\theta \end{cases}

    The threshold pseudo inverse $A_{\theta}^{+}$ is construncted by applying this function to the singular values and recombinng the matrix:

    .. math::

        A_{\theta}^{+} = V f_{\theta}(\Sigma)U^{\dagger} = \sum_{\sigma_i\geq\theta}\frac{1}{\sigma_i}v_iu_i^{\dagger}
    
    For a block-encoded matrix $A$ with normalization factor $\alpha$, this function returns a BlockEncoding of an operator 
    $\tilde{A}_{\alpha\cdot\theta}^{+}$ such that $\|\tilde{A}_{\alpha\cdot\theta}^{+} - A_{\alpha\cdot\theta}^{+}\| \leq \epsilon$.

    The pseudo inverse is implemented via Generalized Quantum Singular Value Transform (GQSVT)
    using a polynomial approximation of $1/x$ over the domain $D_{\theta} = [-1, -\theta] \cup [\theta, 1]$, 
    and a smoothed rectangle filter over the domain $D_{\theta}' = [-\theta, \theta]$.

    .. image:: /_static/pseudo_inverse.png
       :align: center

    Parameters
    ----------
    A : BlockEncoding
        The block-encoded matrix to be pseudo inverted. It is assumed that
        the relevant eigenvalues of $A/\alpha$ lie within $D_{\theta}$.
    eps : float
        The target precision $\epsilon$.
    theta : float
        This threshold value defines the boundaries of the "gap" around zero 
        $[-\theta, \theta]\subset [-1,1]$ where the function $1/x$ is not approximated.
    delta : float
        The width of the transition region. The function will smoothly decay from 1 to 0 over the intervals 
        $[-\theta, -\theta + 2\delta]$ and $[\theta - 2\delta, \theta]$. 
        Defaults to $\delta = \theta/4$.

    Returns
    -------
    BlockEncoding
        A new BlockEncoding instance representing an approximation of the pseudo inverse $A_{\alpha\cdot\theta}^{+}$.

    Notes
    -----
    - **Complexity**: The polynomial degree scales as :math:`\mathcal{O}(\log(1/(\epsilon\theta))/\theta)`.

    References
    ----------
    - Childs et al. (2017) `Quantum algorithm for systems of linear equations with exponentially improved dependence on precision <https://arxiv.org/pdf/1511.02306>`_.
    - Gilyén et al. (2019) `Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics <https://dl.acm.org/doi/abs/10.1145/3313276.3316366>`_.

    Examples
    --------

    First, define a matrix $A$ and a right-hand side vector $\vec{b}$.

    ::

        import numpy as np

        N = 4
        A = 1.4 * np.eye(N, k=1) + 1.1 * np.eye(N, k=-1)
        A[0, N-1] = 1.1
        A[N-1, 0] = 1.4

        b = np.array([0, 1, 1, 1])

        print(A)
        #[[0.  1.4 0.  1.1]
        # [1.1 0.  1.4 0. ]
        # [0.  1.1 0.  1.4]
        # [1.4 0.  1.1 0. ]]
        
        _, S, _ = np.linalg.svd(A)
        print("Singular values of A: ", S)
        # Singular values of A:  [2.5 2.5 0.3 0.3]

    Generate a block-encoding of $A$ and use :meth:`pseudo_inversion` to find a block-encoding approximating $A_{1}^{+}$.

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.gqsp import pseudo_inversion

        # Define efficient block-encoding.
        def f0(x): x+=1
        def f1(x): x-=1
        BE = BlockEncoding.from_lcu(np.array([1.4, 1.1]), [f0, f1])
        # alpha = 1.4 + 1.1 = 2.5

        # Choose threshold theta > 0.3 / 2.5 = 0.12 
        # to cut off smallest singular values.
        BE_inv = pseudo_inversion(BE, 0.01, 0.4, 0.1)

        # Prepares operand variable in state |b>
        def prep_b():
            operand = QuantumFloat(2)
            prepare(operand, b)
            return operand

        @terminal_sampling
        def main():
            operand = prep_b()
            ancillas = BE_inv.apply(operand)
            return operand, *ancillas

        res_dict = main()

        # Post-selection on ancillas being in |0> state
        filtered_dict = {k[0]: p for k, p in res_dict.items() \
                        if all(x == 0 for x in k[1:])}
        success_prob = sum(filtered_dict.values())
        print(success_prob)
        filtered_dict = {k: p / success_prob for k, p in filtered_dict.items()}

        amps = np.sqrt([filtered_dict.get(i, 0) for i in range(4)])
        print(amps)

    Finally, compare the quantum simulation result with the classical solution:

    ::

        def threshold_pseudoinverse(A, threshold):
            U, S, Vh = np.linalg.svd(A, full_matrices=False)
            S_inv = np.zeros_like(S)
            valid_mask = S >= threshold
            S_inv[valid_mask] = 1.0 / S[valid_mask]
            return (Vh.conj().T * S_inv) @ U.conj().T

        c = (threshold_pseudoinverse(A, 0.4 * 2.5) @ b) 
        c = c / np.linalg.norm(c)

        print("QUANTUM SIMULATION\n", amps, "\nCLASSICAL SOLUTION\n", c)
        # QUANTUM SIMULATION
        # [0.63244232 0.31179432 0.63244232 0.32065201] 
        # CLASSICAL SOLUTION
        # [0.63245553 0.31622777 0.63245553 0.31622777]

    """

    p = _pseudo_inverse_cheb(kappa, delta, eps, 100)

    # Set _rescale=False to apply p(A/α) instead of p(A).
    A_pseudo_inv = GQSVT(A, p, kind="Chebyshev", rescale=False)

    # Adjust scaling factor since (A/α)^{-1} = αA^{-1}.
    A_pseudo_inv.alpha = A_pseudo_inv.alpha / A.alpha
    return A_pseudo_inv


def _smooth_rectangle(
    x: npt.NDArray[np.float64], 
    t: float, 
    delta: float,
) -> npt.NDArray[np.float64]:
    """
    Computes a smoothed rectangle (indicator) function using the error function.

    This function acts as a continuous, differentiable stand-in for 
    a harsh discontinuous step function. It evaluates to approximately 1 inside 
    the interval [-t, t] and transitions to 0 over a specified width. 
    Smoothing the jump prevents the Gibbs phenomenon (wild oscillations) when 
    subsequently fitting this target with a Chebyshev polynomial.

    Parameters
    ----------
    x : ndarray
        The input array of values (typically evaluating over the domain [-1, 1]).
    t : float
        The half-width of the inner interval. The function will be approximately 1 for x in [-t, t].
    delta :float
        The width of the transition region. The function will smoothly decay from 1 to 0 over the intervals 
        $[-t - \delta, -t + \delta]$ and $[t - \delta, t + \delta]$.

    Returns
    -------
    ndarray
        An array of evaluated function values, bounded between 0 and 1, with the same shape as the input array ``x``.
    """
    # kappa dictates the steepness of the transition. 
    # The factor of 2.0 is an empirical choice to ensure the curve settles 
    # completely to 0 or 1 within the delta region.
    kappa = 2.0 / delta 
    
    erf_plus = erf(kappa * (x + t))
    erf_minus = erf(kappa * (x - t))
    
    return 0.5 * (erf_plus - erf_minus)


def _pseudo_inverse_cheb(
    theta: float,
    delta: float = None,
    eps: float = 1e-3,
    max_N: int = 100,
) -> npt.NDArray[np.float64]:
    r"""
    Constructs a Chebyshev polynomial approximation of the pseudo-inverse.

    This function creates a polynomial that approximates $1/x$ over the domain 
    $[-1, \theta] \cup [\theta, 1]$ while smoothly dropping to zero around the 
    origin. It achieves this by multiplying an odd Chebyshev approximation of $1/x$ 
    (https://arxiv.org/pdf/1511.02306, Lemma 14) with an even, smooth "inverted rectangle" 
    filter that cuts off the region close to zero (https://arxiv.org/pdf/1806.01838, Lemma 29).

    Parameters
    ----------
    theta : float
        This threshold value defines the boundaries of the "gap" around zero 
        $[-\theta, \theta]\subset [-1,1]$ where the function $1/x$ is not approximated.
    delta : float, optional
        The width of the transition region for the smooth origin cutoff. 
        If None, it defaults to $\theta / 4$.
    eps : float, optional
        The target precision $\epsilon$ for the approximations. Defaults to 1e-3.
    max_N : int, optional
        The maximum polynomial degree to evaluate when interpolating the 
        even cutoff polynomial (the smooth rectangle). Defaults to 100.

    Returns
    -------
    ndarray
        1-D array containing the coefficients of the Chebyshev series representing the smooth, bounded 
        approximation of the pseudo-inverse, ordered from lowest order term to highest.
    """

    if delta is None:
        delta = theta / 4

    t = theta - delta

    # Define the target function for Chebyshev interpolation.
    target_func = lambda x: _smooth_rectangle(x, t, delta)
    cheb_even = 1 - chebyshev_approx(target_func, eps=eps, max_N=max_N)
    
    # The inversion polynomial is constructed using cks_params and cks_coeffs.
    # Since approximating 1/x over the relevant spectral interval [-1, -1/kappa] + [1/kappa, 1]
    # requires an odd Chebyshev series, cks_coeffs returns an array containing only the odd-degree coefficients.
    # This array is expanded into a full Chebyshev series by padding even-degree terms with zeros.
    j_0, beta = cks_params(eps, 1 / theta)
    coeffs_odd = cks_coeffs(j_0, beta)
    coeffs_odd = coeffs_odd * (-1) ** np.arange(len(coeffs_odd))
    coeffs = np.zeros(2 * len(coeffs_odd))
    coeffs[1::2] = coeffs_odd
    cheb_odd = Chebyshev(coeffs)

    cheb_pseudo_inverse = cheb_odd * cheb_even
    return cheb_pseudo_inverse.coef
