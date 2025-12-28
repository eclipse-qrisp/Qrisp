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
from qrisp.alg_primitives.reflection import reflection
from qrisp.algorithms.gqsp.gqsp import GQSP
from qrisp.jasp import qache
from scipy.special import jv
import jax
import jax.numpy as jnp


# https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368
@qache(static_argnames=["H", "N"])
def hamiltonian_simulation(qarg, H, t=1, N=1):
    r"""
    Performs Hamiltonian simulation.

    Based on Jacobi-Anger expansion

    .. math ::

        e^{-it\cos(\theta)} = \sum{n=-\infty}^{\infty}(-i)^nJ_n(t)e^{in\theta}

    where $J_n(t)$ are Bessel functions of the first kind.

    Parameters
    ----------
    qarg : QuantumVariable
        The QuantumVariable representing the state to apply Hamiltonian simulation on.
    H : QubitOperator
        The Hermitian operator.
    t : float, optional
        The time.
    N : int, optional
        The truncation index.

    Returns
    -------
    qbl : QuantumBool
        Auxiliary variable after GQSP protocol. Must be measured in state $\ket{0}$.
    case : QuantumFloat
        Auxiliary variable after GQSP protocol. Must be measured in state $\ket{0}$.

    Examples
    --------

    """
    H = H.hermitize().to_pauli()
    # Rescaling the time to account for scaling factor alpha of pauli block-encoding
    _, coeffs = H.unitaries()
    alpha = np.sum(np.abs(coeffs))
    t = t * alpha

    # Calculate coefficients of truncated Jacobi-Anger expansion
    # jax.scipy.jv is currently not implemented
    # To evaluate jv(m,s) for dynamic s, we evaluate scipy.jv at t=1.0 
    # and use the Bessel multiplication theorem to evaluate jv(m,s)
    j_val_at_1 = jv(np.arange(0, 2*N+1, 1), 1.0)
    j_val_at_t = bessel_multiplication(np.arange(0, N+1), t, j_val_at_1)
    # J_{-n}(t) = (-1)^nJ_n(t)
    j_values = jnp.concatenate(((j_val_at_t * (-1.) ** jnp.arange(0, N+1))[::-1], j_val_at_t[1:]))
    factors = (-1.j) ** jnp.arange(-N, N +1)
    coeffs = factors * j_values

    U, state_prep, n = H.pauli_block_encoding()

    # Qubitization step: RU^k is a block-encoding of T_k(H)
    def RU(case, operand):
        U(case, operand)
        reflection(case, state_function=state_prep)

    case = QuantumFloat(n)
    
    with conjugate(state_prep)(case):
        qbl = GQSP([case, qarg], RU, coeffs, k=N)

    return qbl, case


# jax.scipy.jv is currently not implemented
# To evaluate jv(m,s) for dynamic s, we evaluate scipy.jv at t=1.0
# and use the Bessel multiplication theorem to evaluate jv(m,s)
@jax.jit
def bessel_multiplication(m, s, jv_values_at_t, t=1.0):
    """
    Computes ``jv(m, s)`` using the Bessel Multiplication Theorem.

    .. math::

        J_{m}(\lambda t) = \lambda^m \sum_{k=0}^{\intfy}\frac{(1-\lambda^2)^k(t/2)^k}{k!}J_{m+k}(t)
    
    Parameters
    ----------
    m : ndarray
        Order of the Bessel function $J_m(s)$.
    s : float
        New argument to evaluate.
    j_values_at_t: ndarray
        Array of values $J_k(t)$ for $k=0,\dotsc,N$.
    t : float
        Fixed argument where values are known (default 1.0).

    Returns
    -------
    float 
        Approximation of $J_m(s)$, the Bessel function evaluated at order $m$ and value $s$. 

    Notes
    -----
    - Vectorized version of Bessel multiplication for m as an array.

    """
    # Use jax.vmap to map the single-m logic over an array of m values
    return jax.vmap(lambda m_val: _bessel_multiplication(m_val, s, jv_values_at_t, t))(m)


def _bessel_multiplication(m, s, jv_values_at_t, t=1.0):
    lam = s / t
    lam_sq_diff = 1.0 - lam**2
    t_half = t / 2.0
    
    max_k = jv_values_at_t.shape[0]

    def body_fun(k, val):
        coeff, total_sum = val
        idx = m + k
        
        # Guard the index to stay within bounds for the array access
        # Values outside the valid range for a specific 'm' are effectively ignored
        valid_mask = idx < max_k
        safe_idx = jnp.where(valid_mask, idx, 0)
        
        term = coeff * jv_values_at_t[safe_idx]
        total_sum += jnp.where(valid_mask, term, 0.0)
        
        new_coeff = coeff * lam_sq_diff * t_half / (k + 1)
        return (new_coeff, total_sum)

    init_state = (1.0, 0.0)
    # Loop over the maximum possible number of terms to keep loop bounds static
    _, final_sum = jax.lax.fori_loop(0, max_k, body_fun, init_state)
    
    return (lam**m) * final_sum