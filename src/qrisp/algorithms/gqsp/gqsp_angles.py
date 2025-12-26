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
import jax.numpy as jnp


# https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368
@jax.jit
def _complementary_objective(a, b):
    """
    Computes the complementary objective function for two given polynomials.

    Parameters
    ----------
    a : ndarray
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.   
    b : ndarray
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.   

    Returns
    -------
    float
        The objective function value.
    
    """
    d = len(a) - 1
    delta = jnp.zeros(2*d + 1)
    delta = delta.at[d].set(1)
    r = jnp.convolve(a, jnp.conjugate(a[::-1]), mode='full') + jnp.convolve(b, jnp.conjugate(b[::-1]), mode='full') - delta
    return jnp.linalg.norm(r)


@partial(jax.jit, static_argnames=['N'])
def _maximum(b, N=1024):
    r"""
    Finds the maximum absolute value that a given polynomial assumes on the unit circle.

    Parameters
    ----------
    b : ndarray
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.   

    Returns
    -------
    float
        The maximum absolute value.

    """
    # 1. Evaluate b(z) at N-th roots of unity
    # Using standard FFT (maps coefficients to point values on the circle).
    values = jnp.fft.fft(b, n=N)
    return jnp.max(jnp.abs(values))


@jax.jit
def _complementary_polynomial(b):
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
    b : ndarray
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.   

    Returns
    -------
    a : ndarray
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
    log_a_mag = 0.5 * jnp.log(jnp.clip(1 - mag_sq, a_min=1e-10, a_max=1.0))
    
    # 3. Transform to the Cepstral domain
    # The IFFT of the log-magnitude gives the "real Cepstrum".
    cepstrum = jnp.fft.ifft(log_a_mag)

    # 4. Apply analytic projection (Schwarz/Hilbert transform in Cepstral domain)
    # An outer function's log-magnitude and phase are related by the Hilbert 
    # Transform. In the Cepstral domain, this means zeroing negative frequencies 
    # (indices > N/2) and doubling positive ones (indices < N/2).
    mid = N // 2
    a_cep_analytic = jnp.zeros(N, dtype=jnp.complex64)
    
    a_cep_analytic = a_cep_analytic.at[0].set(cepstrum[0]) # DC
    a_cep_analytic = a_cep_analytic.at[1:mid].set(2 * cepstrum[1:mid]) # Positive frequencies
    a_cep_analytic = a_cep_analytic.at[mid].set(cepstrum[mid]) # Nyquist
    
    # 5. Recovery of coefficients
    a_points = jnp.exp(jnp.fft.fft(a_cep_analytic))
    a_coeffs = jnp.fft.ifft(a_points)
    
    return a_coeffs[:d+1]


@jax.jit
def _angles(p, q):
    r"""
    Computes the GQSP angles for two given (complementary) polynomials.

    Given two polynomials such that

    * $p,q\in\mathbb C[x]$, $\deg p, \deg q \leq d$,
    * for all $x\in\mathbb R$, $|p(e^{ix})|^2+|q(e^{ix})|^2=1$,

    this method computes the angles $\theta,\phi\in\mathbb R^{d+1}$, $\lambda\in\mathbb R$.

    This function is JAX-traceable.

    Parameters
    ----------
    p : ndarray
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.
    q : ndarray
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.

    Returns
    -------
    theta_arr : ndarray
        The angles $(\theta_0,\dotsc,\theta_d)$.
    phi_arr : ndarray
        The angles $(\phi_0,\dotsc,\phi_d)$.
    lambda : float
        The angle $\lambda$.

    """

    d = len(p) - 1
    theta_arr = jnp.zeros(d + 1)
    phi_arr = jnp.zeros(d + 1)

    S = jnp.vstack([p, q])

    # Add a small perturbation to denominators to avoid division by zero when computing angles
    eps = 1e-10
    theta_arr = theta_arr.at[d].set(jnp.arctan(jnp.abs(S[1][d] / (S[0][d] + eps))))
    phi_arr = phi_arr.at[d].set(jnp.angle(S[0][d] / (S[1][d] + eps)))

    def cond_fun(vals):
        d, S, theta_arr, phi_arr = vals
        return d > 0

    def body_fun(vals):
        d, S, theta_arr, phi_arr = vals
    
        theta = theta_arr[d]
        phi = phi_arr[d]
        # R(theta, phi, 0)^dagger
        R = jnp.array([[jnp.exp(-phi*1j) * jnp.cos(theta), jnp.sin(theta)],[jnp.exp(-phi*1j) * jnp.sin(theta), -jnp.cos(theta)]])
        S = R @ S
        S = jnp.vstack([S[0][1:d+1],S[1][0:d]])
        
        d = d-1
        theta_arr = theta_arr.at[d].set(jnp.arctan(jnp.abs(S[1][d] / (S[0][d] + eps))))
        phi_arr = phi_arr.at[d].set(jnp.angle(S[0][d] / (S[1][d] + eps)))

        return d, S, theta_arr, phi_arr
    
    #d, S, theta_arr, phi_arr = jax.lax.while_loop(cond_fun, body_fun, (d, S, theta_arr, phi_arr))
    vals = (d, S, theta_arr, phi_arr)
    while(cond_fun(vals)):
        vals = body_fun(vals)

    d, S, theta_arr, phi_arr = vals
    lambda_ = jnp.angle(S[1][0])

    return theta_arr, phi_arr, lambda_


def _gqsp_angles(p):
    r"""
    Computes the GQSP angles for a given polynomial.

    Note: The resulting angles correspond to a properly rescaled version of the input polynomial.

    Parameters
    ----------
    p : ndarray
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.

    Returns
    -------
    theta_arr : ndarray
        The angles $(\theta_0,\dotsc,\theta_d)$.
    phi_arr : ndarray
        The angles $(\phi_0,\dotsc,\phi_d)$.
    lambda : float
        The angle $\lambda$.

    """

    # Comupute the maximum of |p(z)| for |z|=1
    M = _maximum(p, N=1024)

    # Rescale p(z)
    # Divide by M such that |p(z)|<=1 for |z|=1 and QSP success probability is maximized
    p = p / M 
    # Multiply by 0.99 to ensure that |p(z)|<1 for |z|=1 for numerical stability of completion algorithm
    # This comes at the expense of a slightly smaller QSP success probability 
    p = 0.99 * p

    # Find completion q(z) of p(z) such that |p(z)|^2 + |q(z)|^2 = 1 for |z|=1
    q = _complementary_polynomial(p)

    # Compute GQSP angles
    theta, phi, lambda_ = _angles(p, q)

    return theta, phi, lambda_