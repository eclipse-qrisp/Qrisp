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

from qrisp.block_encodings.block_encoding import BlockEncoding


def apply_pseudo_inv(
    self,
    eps: float,
    theta: float,
    delta: float | None = None,
) -> BlockEncoding:
    r"""
    Returns a BlockEncoding approximating the threshold `matrix pseudoinverse <https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse>`_ of the operator.

    Approximates the pseudoinverse of a matrix by ignoring singular values below a specified threshold.
    This regularizes ill-conditioned linear systems where tiny singular values would otherwise amplify noise and cause large numerical errors.
    By discarding these negligible values, this method trades a small amount of exact accuracy for a significantly more stable and reliable solution.

    Let $A$ be a matrix with `Singular Value Decomposition <https://en.wikipedia.org/wiki/Singular_value_decomposition>`_

    .. math::

        A = U\Sigma V^{\dagger} = \sum_i \sigma_i u_iv_i^{\dagger}

    where $\sigma_i$ are the singular values, $u_i$ are the left singular vectors, 
    and $v_i$ are the right singular vectors.

    Given a threshold $\theta>0$, define a scalar thresholding function $f_{\theta}(\sigma)$ that 
    acts on the singular values:

    .. math::

        f_{\theta}(\sigma)=\begin{cases} \frac{1}{\sigma} & \sigma\geq\theta \\ 0 & \sigma<\theta \end{cases}

    The threshold pseudo inverse $A_{\theta}^{+}$ is constructed by applying this function to the singular values and recombining the matrix:

    .. math::

        A_{\theta}^{+} = V f_{\theta}(\Sigma)U^{\dagger} = \sum_{\sigma_i\geq\theta}\frac{1}{\sigma_i}v_iu_i^{\dagger}
    
    For a block-encoded matrix $A$ with normalization factor $\alpha$, this function returns a BlockEncoding of an operator 
    $\tilde{A}_{\alpha\cdot\theta}^{+}$ such that $\|\tilde{A}_{\alpha\cdot\theta}^{+} - A_{\alpha\cdot\theta}^{+}\| \leq \epsilon$.

    The pseudo inverse is implemented via Quantum Singular Value Transform (QSVT)
    using a polynomial approximation of $1/x$ over the domain $D_{\theta} = [-1, -\theta] \cup [\theta, 1]$, 
    and a smoothed rectangle filter over the domain $D_{\theta}' = [-\theta, \theta]$.

    .. image:: /_static/chebyshev_pseudo_inversion.png
       :align: center

    Parameters
    ----------
    eps : float
        The target precision $\epsilon$.
    theta : float
        This threshold value defines the boundaries of the "gap" around zero 
        $[-\theta, \theta]\subset [-1,1]$ where the function $1/x$ is not approximated.
    delta : float, optional
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
    - It is assumed that the relevant singular values of $A/\alpha$ lie within $D_{\theta}$.

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

    Generate a block-encoding of $A$ and use :meth:`pseudo_inv` to find a block-encoding approximating $A_{1}^{+}$.

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
        BE_inv = BE.pseudo_inv(eps=0.01, theta=0.4, delta=0.1)

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

        amps = np.sqrt([filtered_dict.get(i, 0) for i in range(len(b))])
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

    from qrisp.algorithms.gqsp import pseudo_inversion

    return pseudo_inversion(self, eps, theta, delta)
