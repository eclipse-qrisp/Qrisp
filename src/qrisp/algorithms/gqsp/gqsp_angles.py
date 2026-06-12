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

from functools import partial
import numpy as np
import jax
from jax import Array
import jax.numpy as jnp
from typing import Literal, Tuple, TYPE_CHECKING

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
    delta = jnp.zeros(2 * d + 1)
    delta = delta.at[d].set(1)
    r = (
        jnp.convolve(a, jnp.conjugate(a[::-1]), mode="full")
        + jnp.convolve(b, jnp.conjugate(b[::-1]), mode="full")
        - delta
    )
    return jnp.linalg.norm(r)


@partial(jax.jit, static_argnames=["N"])
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
    mag_sq = jnp.abs(b_points) ** 2
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

    a_cep_analytic = a_cep_analytic.at[0].set(cepstrum[0])  # DC
    a_cep_analytic = a_cep_analytic.at[1:mid].set(
        2 * cepstrum[1:mid]
    )  # Positive frequencies
    a_cep_analytic = a_cep_analytic.at[mid].set(cepstrum[mid])  # Nyquist

    # 5. Recovery of coefficients
    a_points = jnp.exp(jnp.fft.fft(a_cep_analytic))
    a_coeffs = jnp.fft.ifft(a_points)

    return a_coeffs[: d + 1]


@jax.jit
def _inlft(a: "ArrayLike", b: "ArrayLike") -> Array:
    r"""
    Computes the inverse non-linear Fourier transform using the layer stripping algorithm.

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
        1-D array containing the non-linear Fourier sequence, ordered from lowest order term to highest.

    """

    # Layer stripping algorithm (https://arxiv.org/pdf/2505.12615, Equation (28))
    # Define the single step of the scan loop
    def step(carry, _):
        a_star, b_arr = carry

        # Calculate F_k
        Fk = b_arr[0] / a_star[0]

        # Calculate intermediate scalar 's'
        s = jnp.sqrt(1.0 + jnp.abs(Fk) ** 2)

        # Update arrays
        a_star_new = (a_star + jnp.conjugate(Fk) * b_arr) / s
        b_new = jnp.roll((b_arr - Fk * a_star) / s, -1)  # divide by z

        # Return the new carry state and the output we want to accumulate (Fk)
        return (a_star_new, b_new), Fk

    # Initial state for the carry
    initial_carry = (jnp.conjugate(a), b)

    # Execute the scan loop for the length of 'a'
    # jax.lax.scan returns the final state (which we ignore with '_') and the stacked outputs
    _, F = jax.lax.scan(step, initial_carry, None, length=len(a))

    return F


@jax.jit
def _gqsp_angles_from_nlft_sequence(F: Array) -> Tuple[Array, Array, Array]:
    r"""
    Computes the GQSP angles form the non-linear Fourier sequence.

    Parameters
    ----------
    F : ArrayLike
        1-D array containing the non-linear Fourier sequence, ordered from lowest order term to highest.

    Returns
    -------
    angles : tuple of (Array, Array, Array)
        A collection containing:

        - **theta** (Array): 1-D array of angles $(\theta_0,\dotsc,\theta_d)$.

        - **phi** (Array): 1-D array of angles $(\phi_0,\dotsc,\phi_d)$.

        - **lambda** (Array): The scalar angle $\lambda$ as 0-D array.

    """
    thres = 1e-10
    # pre-factor
    psi = jnp.where(
        jnp.abs(F) < thres,
        0,
        jnp.where(
            jnp.abs(np.imag(F)) < thres,
            -jnp.pi / 4,
            -(1 / 2) * jnp.arctan(jnp.real(F) / jnp.imag(F)),
        ),
    )

    # Theorem 9, formula (4) in https://arxiv.org/pdf/2503.03026
    phi = jnp.arctan(-1.0j * jnp.exp(-2.0j * psi) * F)
    psi_ = jnp.concatenate((psi, jnp.array([0])))
    theta = jnp.roll(psi_, -1)[:-1] - psi
    lambda_ = psi[0]

    # Switch (Q,P) -> (P, iQ)
    phi = phi.at[-1].set(phi[-1] + np.pi / 2)
    theta = theta.at[-1].set(-theta[-1])

    theta = jnp.real(theta)
    phi = jnp.real(phi)
    lambda_ = jnp.real(lambda_)
    return theta, phi, lambda_


@jax.jit
def _xqsp_angles_from_nlft_sequence(F: Array) -> Array:
    r"""
    Computes the XQSP angles form the non-linear Fourier sequence.

    Parameters
    ----------
    F : ArrayLike
        1-D array containing the non-linear Fourier sequence, ordered from lowest order term to highest.

    Returns
    -------
    angles : Array
        1-D array of angles $(\phi_0,\dotsc,\phi_d)$.

    Raises
    ------
    NotImplementedError
        Always raised until the XQSP convention is mathematically verified.
    """
    raise NotImplementedError(
            "The XQSP angle calculation is currently unverified and disabled. "
            "Please use QSP, GQSP, or QSVT conventions instead."
        )
    #return jnp.arctan(-jnp.imag(F))


@jax.jit
def _yqsp_angles_from_nlft_sequence(F: Array) -> Array:
    r"""
    Computes the YQSP angles form the non-linear Fourier sequence.

    Parameters
    ----------
    F : ArrayLike
        1-D array containing the non-linear Fourier sequence, ordered from lowest order term to highest.

    Returns
    -------
    angles : Array
        1-D array of angles $(\phi_0,\dotsc,\phi_d)$.

    Raises
    ------
    NotImplementedError
        Always raised until the YQSP convention is mathematically verified.
    """
    raise NotImplementedError(
            "The YQSP angle calculation is currently unverified and disabled. "
            "Please use QSP, GQSP, or QSVT conventions instead."
        )
    #return jnp.arctan(jnp.real(F))


def poly_to_nlft_sequence(p: "ArrayLike") -> Array:
    r"""
    Computes the non-linear Fourier sequence for a given polynomial.

    Parameters
    ----------
    p : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.

    Returns
    -------
    F : Array
        1-D array containing the non-linear Fourier sequence, ordered from lowest order term to highest.

    """
    # Comupute the maximum of |p(z)| for |z|=1
    M = _maximum(p, N=1024)

    # Rescale p(z)
    # Divide by M such that |p(z)|<=1 for |z|=1 and QSP success probability is maximized
    p = p / M
    # Multiply by 0.99 to ensure that |p(z)|<1 for |z|=1 for numerical stability of completion algorithm
    # This comes at the expense of a slightly smaller QSP success probability
    p = 0.99 * p
    alpha = M / 0.99
    # Switch (Q,P) -> (P, iQ)
    p = -1.0j * p

    # Find completion q(z) of p(z) such that |p(z)|^2 + |q(z)|^2 = 1 for |z|=1
    q = _complementary_polynomial(p)

    F = _inlft(q, p)
    return F, alpha


# https://arxiv.org/pdf/2503.03026
def gqsp_angles(p: "ArrayLike") -> Tuple[Tuple[Array, Array, Array], Array]:
    r"""
    Computes the GQSP angles for a given polynomial.

    Parameters
    ----------
    p : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.

    Returns
    -------
    angles : tuple of (Array, Array, Array)
        A collection containing:

        - **theta** (Array): 1-D array of angles $(\theta_0,\dotsc,\theta_d)$.

        - **phi** (Array): 1-D array of angles $(\phi_0,\dotsc,\phi_d)$.

        - **lambda** (Array): The scalar angle $\lambda$ as 0-D array.

    alpha : Array
        The scalar scaling factor as 0-D array.

    Notes
    -----
    - The resulting angles correspond to a rescaled version of the input polynomial.

    """
    F, alpha = poly_to_nlft_sequence(p)
    theta, phi, lambda_ = _gqsp_angles_from_nlft_sequence(F)

    return (theta, phi, lambda_), alpha


# https://arxiv.org/pdf/2503.03026 
# Not verified to be correct.
def xqsp_angles(p: "ArrayLike") -> Tuple[Array, Array]:
    r"""
    Computes the XQSP angles for a given polynomial.

    Parameters
    ----------
    p : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.

    Returns
    -------
    angles : Array
        1-D array of angles $(\phi_0,\dotsc,\phi_d)$.
    alpha : Array
        The scalar scaling factor as 0-D array.

    Raises
    ------
    NotImplementedError
        Always raised until the XQSP convention is mathematically verified.

    Notes
    -----
    - The resulting angles correspond to a rescaled version of the input polynomial.

    """
    raise NotImplementedError(
            "The XQSP angle calculation is currently unverified and disabled. "
            "Please use QSP, GQSP, or QSVT conventions instead."
        )
    #F, alpha = poly_to_nlft_sequence(p)
    #phi = _xqsp_angles_from_nlft_sequence(F)
    #return phi, alpha


# https://arxiv.org/pdf/2503.03026 
# Not verified to be correct.
def yqsp_angles(p: "ArrayLike") -> Tuple[Array, Array]:
    r"""
    Computes the YQSP angles for a given polynomial.

    Parameters
    ----------
    p : ArrayLike
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.

    Returns
    -------
    angles : Array
        1-D array of angles $(\phi_0,\dotsc,\phi_d)$.
    alpha : Array
        The scalar scaling factor as 0-D array.

    Raises
    ------
    NotImplementedError
        Always raised until the YQSP convention is mathematically verified.

    Notes
    -----
    - The resulting angles correspond to a rescaled version of the input polynomial.

    """
    raise NotImplementedError(
            "The YQSP angle calculation is currently unverified and disabled. "
            "Please use QSP, GQSP, or QSVT conventions instead."
        )
    #F, alpha = poly_to_nlft_sequence(p)
    #phi = _yqsp_angles_from_nlft_sequence(F)
    #return phi, alpha


def laurent_to_analytic_coeffs(
    target_coeffs: "ArrayLike", parity: Literal["even", "odd"] = "odd"
) -> Array:
    """
    Converts a target polynomial from the Laurent QSP framework to the Analytic QSP framework.

    In standard QSP, signal operators enforce strict parity (even or odd). Analytic QSP
    (like GQSP) natively builds polynomials in strictly positive powers of a variable
    without parity constraints.

    This helper performs the algebraic mapping required to trick an analytic solver
    into solving a Laurent polynomial. Specifically, it factors out x^{-d} and
    substitutes y = x^2 to create a dense analytic polynomial A(y).

    Mathematical Example (Odd):
        Target: P(x) = a_1*x + a_3*x^3  (Coeffs: [0, a_1, 0, a_3], degree d=3)
        1. Factor out x^{-d}: P(x) = x^{-3} * (a_1*x^4 + a_3*x^6)
        2. Substitute y = x^2: A(y) = a_1*y^2 + a_3*y^3
        3. Expanded A(y): 0*y^0 + 0*y^1 + a_1*y^2 + a_3*y^3
        4. Analytic Coeffs: [0.0, 0.0, a_1, a_3]

    Mathematical Example (Even):
        Target: P(x) = a_0 + a_2*x^2  (Coeffs: [a_0, 0, a_2], degree d=2)
        1. Factor out x^{-d}: P(x) = x^{-2} * (a_0*x^2 + a_2*x^4)
        2. Substitute y = x^2: A(y) = a_0*y + a_2*y^2
        3. Expanded A(y): 0*y^0 + a_0*y^1 + a_2*y^2
        4. Analytic Coeffs: [0.0, a_0, a_2]

    Once the analytic solver finds the angles for A(y), Lemma 2 from Laneve (2025)
    is used to shift the phases, effectively multiplying the x^{-d} shift back
    into the quantum circuit.

    Parameters
    ----------
    target_coeffs : jax.Array
        The standard coefficients of the target polynomial (a_0, a_1, a_2, ...).
    parity : Literal["even", "odd"]
        The structural parity of the target polynomial ('even' or 'odd'). Defaults to 'odd'.
        Must be known at compile time for JAX tracing.

    Returns
    -------
    jax.Array
        The mapped Analytic QSP coefficient array A(y). The first half consists
        of zeros (representing the skipped lower-degree y terms), and the second
        half consists of the extracted parity coefficients.
    """
    if parity == "odd":
        # Extract odd coefficients: indices 1, 3, 5...
        extracted_coeffs = target_coeffs[1::2]
        num_zeros = extracted_coeffs.shape[0]
    elif parity == "even":
        # Extract even coefficients: indices 0, 2, 4...
        extracted_coeffs = target_coeffs[0::2]
        num_zeros = extracted_coeffs.shape[0] - 1
    else:
        raise ValueError("Parity must be either 'even' or 'odd'.")

    # Create an array of zeros matching the input dtype
    zeros = jnp.zeros(num_zeros, dtype=target_coeffs.dtype)

    # Concatenate the zeros with the extracted coefficients
    analytic_coeffs = jnp.concatenate((zeros, extracted_coeffs))

    return analytic_coeffs


def qsp_angles(
    p: "ArrayLike",
    parity: Literal["even", "odd"] = "odd",
    signal_basis: Literal["X", "Z"] = "Z",
) -> Tuple[Array, Array]:
    r"""
    Computes the QSP angles for a given polynomial.

    Parameters
    ----------
    p : ArrayLike
        1-D array containing the polynomial coefficients in Chebyshev basis, ordered from lowest order term to highest.
    parity : Literal["even", "odd"]
        The structural parity of the target polynomial ('even' or 'odd'). Defaults to 'odd'.
        Must be known at compile time for JAX tracing.
    signal_basis : Literal["X", "Z"]
        The signal basis for the QSP angles ('X' or 'Z'). Defaults to 'Z'.

    Returns
    -------
    angles : Array
        1-D array of angles $(\phi_0,\dotsc,\phi_d)$.
    alpha : Array
        The scalar scaling factor as 0-D array.

    Notes
    -----
    - The resulting angles correspond to a rescaled version of the input polynomial.

    """
    p_analytic = laurent_to_analytic_coeffs(p, parity=parity)
    (theta, phi, lambda_), alpha = gqsp_angles(p_analytic)

    if signal_basis == "X":
        phi = phi.at[0].set(phi[0] - np.pi / 4)
        phi = phi.at[-1].set(phi[-1] + np.pi / 4)

    return phi, alpha


def qsvt_angles(p: "ArrayLike", parity: Literal["even", "odd"] = "odd") -> Tuple[Array, Array]:
    r"""
    Computes the QSVT angles for a given polynomial.

    Parameters
    ----------
    p : ArrayLike
        1-D array containing the polynomial coefficients in Chebyshev basis, ordered from lowest order term to highest.
    parity : Literal["even", "odd"]
        The structural parity of the target polynomial ('even' or 'odd'). Defaults to 'odd'.
        Must be known at compile time for JAX tracing.

    Returns
    -------
    angles : Array
        1-D array of angles $(\phi_0,\dotsc,\phi_d)$.
    alpha : Array
        The scalar scaling factor as 0-D array.

    Notes
    -----
    - The resulting angles correspond to a rescaled version of the input polynomial.

    """
    phi_qsp, alpha = qsp_angles(p, parity=parity, signal_basis="X")

    d = len(phi_qsp) - 1

    phi_qsvt = jnp.zeros(d + 1)
    phi_qsvt = phi_qsvt.at[0].set(phi_qsp[0] + (2 * d - 1) * np.pi / 4)
    phi_qsvt = phi_qsvt.at[1:].set(phi_qsp[1:] - np.pi / 2)
    phi_qsvt = phi_qsvt.at[d].set(phi_qsp[d] - np.pi / 4)
    return phi_qsvt, alpha
