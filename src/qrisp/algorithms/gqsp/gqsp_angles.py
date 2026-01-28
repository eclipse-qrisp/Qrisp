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

from functools import partial
import numpy as np
import jax
from jax import Array
import jax.numpy as jnp
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from jax.typing import ArrayLike


# https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368
@jax.jit
def _complementary_objective(a: "ArrayLike", b: "ArrayLike") -> Array:
    """
    Computes the complementary objective function for two given polynomials.

    Parameters
    ----------
    a : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.   
    b : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.   

    Returns
    -------
    Array
        The scalar objective function value as 0-D Array.
    
    """
    d = len(a) - 1
    delta = jnp.zeros(2*d + 1)
    delta = delta.at[d].set(1)
    r = jnp.convolve(a, jnp.conjugate(a[::-1]), mode='full') + jnp.convolve(b, jnp.conjugate(b[::-1]), mode='full') - delta
    return jnp.linalg.norm(r)


@partial(jax.jit, static_argnames=['N'])
def _maximum(b: "ArrayLike", N: int = 1024) -> Array:
    r"""
    Finds the maximum absolute value that a given polynomial assumes on the unit circle.

    Parameters
    ----------
    b : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.   
    N : int
        The number of roots of unity to evaluate the polynomial.

    Returns
    -------
    Array
        The scalar maximum absolute value as 0-D array.

    """
    # 1. Evaluate b(z) at N-th roots of unity
    # Using standard FFT (maps coefficients to point values on the circle).
    values = jnp.fft.fft(b, n=N)
    return jnp.max(jnp.abs(values))


@jax.jit
def _complementary_polynomial(b: "ArrayLike") -> Array:
    r"""
    Finds a complementary polynomial $a$ such that $|a|^2 + |b|^2 = 1$ on the unit circle.
    
    This function implements spectral factorization via the Cepstral method.
    It constructs the unique outer polynomial (analytic and non-zero inside
    the unit disk) that satisfies the power-sum identity.

    This function calculates the spectral factor $a(z)$ by constructing an analytic function in the disk
    whose real part on the boundary is $\log{|a|}$. The projection in Step 4 is the discreate equivalent of the Schwarz integral:

    .. math ::

        G(z) = \frac{1}{2\pi}\int_{0}^{2\pi}\log{|a(e^{i\theta})|}\frac{e^{i\theta}+z}{e^{i\theta}-z}\mathrm d\theta

    The resulting $a(z)=\exp(G(z))$ is guaranteed to be the unique outer polynomial with a positive real mean $a_0$ (if b is real).

    Note: The polynomial $b(z)$ must satisfy $|b(z)| \leq 1$ on the unit disk. This algorithm is unstable if $|b(z)|=1$ for an N-th root of unity $z_k=\exp(2\pi i k/N)$, 
    since $\log(|a(z)|) = \log(1-|b(z)|^2)/2 has a singularity is this case. This can be mitigated by rescaling $b(z)$ such that $|b(z)|<1$ on the unit disk. 

    Parameters
    ----------
    b : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.   

    Returns
    -------
    a : Array
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.

    """
    d = b.shape[0] - 1
    # The degree of b is d. The degree of |b|^2 is 2d.
    # Choose N as a power of two larger than 2d to avoid aliasing. Multiply by factor 8 for increased precison.
    N = 8 * (1 << (2 * d + 2).bit_length())
    
    # 1. Evaluate b(z) at N-th roots of unity
    # Using standard FFT (maps coefficients to point values on the circle).
    b_points = jnp.fft.fft(b, n=N)

    # 2. Compute log-magnitude of a(z)
    # log|a| = 0.5 * log(1 - |b|^2)
    mag_sq = jnp.abs(b_points)**2
    log_a_mag = 0.5 * jnp.log(jnp.clip(1 - mag_sq, min=1e-10, max=1.0))
    
    # 3. Transform to the Cepstral domain
    # The IFFT of the log-magnitude gives the "real Cepstrum".
    cepstrum = jnp.fft.ifft(log_a_mag)

    # 4. Apply analytic projection (Schwarz/Hilbert transform in Cepstral domain)
    # An outer function's log-magnitude and phase are related by the Hilbert 
    # Transform. In the Cepstral domain, this means zeroing negative frequencies 
    # (indices > N/2) and doubling positive ones (indices < N/2).
    mid = N // 2
    a_cep_analytic = jnp.zeros(N, dtype=jnp.complex128)
    
    a_cep_analytic = a_cep_analytic.at[0].set(cepstrum[0]) # DC
    a_cep_analytic = a_cep_analytic.at[1:mid].set(2 * cepstrum[1:mid]) # Positive frequencies
    a_cep_analytic = a_cep_analytic.at[mid].set(cepstrum[mid]) # Nyquist
    
    # 5. Recovery of coefficients
    a_points = jnp.exp(jnp.fft.fft(a_cep_analytic))
    a_coeffs = jnp.fft.ifft(a_points)
    
    return a_coeffs[:d+1]


@jax.jit
def _inlft(a: "ArrayLike", b: "ArrayLike") -> Array:
    r"""
    Perform inverse non-linear Fourier transform.

    .. math ::

        F_k = \frac{b_k(0)}{a_k^*(0)}, 
        \quad a_{k+1}^*(z) = \frac{a_k^*(z)+\bar{F_k}b_k(z)}{\sqrt{1+|F_k|^2}}, 
        \quad b_{k+1}(z) = \frac{b_k(z)-F_ka_k^*(z)}{\sqrt{1+|F_k|^2}}

    Parameters
    ----------
    a : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.
    b : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.

    Returns
    -------
    F : Array
        1-D array containing the sequence, ordered from lowest order term to highest.
    
    """
    d = len(a) - 1

    a_star = jnp.conjugate(a)

    F = jnp.zeros(d+1, dtype=complex)

    for k in range(d+1):
        Fk = b[0] / a_star[0]
        F = F.at[k].set(Fk)

        s = jnp.sqrt(1.0 + jnp.abs(Fk)**2)
        a_star_new = (a_star + jnp.conjugate(Fk) * b) / s
        b_new = jnp.roll( (b - Fk * a_star) / s, -1) # divide by z
        a_star = a_star_new
        b = b_new

    return F


# https://arxiv.org/pdf/2503.03026
def gqsp_angles(p: "ArrayLike") -> Tuple[Array, Array, Array]:
    r"""
    Computes the GQSP angles for a given polynomial.

    Parameters
    ----------
    p : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.

    Returns
    -------
    theta : Array
        1-D array of angles $(\theta_0,\dotsc,\theta_d)$.
    phi : Array
        1-D array of angles $(\phi_0,\dotsc,\phi_d)$.
    lambda : Array
        The scalar angle $\lambda$ as 0-D array.

    Notes
    -----
    - The resulting angles correspond to a rescaled version of the input polynomial.

    """

    # Comupute the maximum of |p(z)| for |z|=1
    M = _maximum(p, N=1024)

    # Rescale p(z)
    # Divide by M such that |p(z)|<=1 for |z|=1 and QSP success probability is maximized
    p = p / M 
    # Multiply by 0.99 to ensure that |p(z)|<1 for |z|=1 for numerical stability of completion algorithm
    # This comes at the expense of a slightly smaller QSP success probability 
    p = 0.99 * p
    # Switch (Q,P) -> (P, iQ)
    p = -1.j * p

    # Find completion q(z) of p(z) such that |p(z)|^2 + |q(z)|^2 = 1 for |z|=1
    q = _complementary_polynomial(p)

    # INLFT
    F = _inlft(q, p)

    # Compute GQSP angles
    thres = 1e-10
    # pre-factor
    psi = jnp.where(jnp.abs(F)<thres, 0, jnp.where(jnp.abs(np.imag(F))<thres, -jnp.pi/4, -(1/2)*jnp.arctan(jnp.real(F) / jnp.imag(F))))

    # Theorem 9, formula (4) in https://arxiv.org/pdf/2503.03026
    phi = jnp.arctan(-1.j * jnp.exp(-2.j * psi) * F)
    psi_ = jnp.concatenate((psi, jnp.array([0])))
    theta = jnp.roll(psi_, -1)[:-1] - psi
    lambda_ = psi[0]

    # Switch (Q,P) -> (P, iQ)
    phi = phi.at[-1].set(phi[-1] + np.pi/2)
    theta = theta.at[-1].set(-theta[-1])

    phi = jnp.real(phi)
    theta = jnp.real(theta)
    lambda_ = jnp.real(lambda_)

    return theta, phi, lambda_