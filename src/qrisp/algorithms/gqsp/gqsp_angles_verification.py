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

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.chebyshev as cheb
import numpy.polynomial.polynomial as poly
import numpy.typing as npt
from typing import Union


def plot_reconstruction_vs_target(
    domain: npt.NDArray[np.float64],
    target_values: npt.NDArray[np.float64],
    reconstructed_values: npt.NDArray[np.float64],
    domain_label: str = "Singular Value (x)",
    title: str = "Input Target vs. Reconstructed Polynomial",
):
    """
    Plots the target polynomial against the reconstructed QSP/GQSP response,
    along with a subplot showing the absolute residual error.

    Parameters:
    -----------
    domain : numpy.ndarray
        The x-axis values (e.g., x in [-1, 1] for QSP, or omega in [0, pi] for GQSP).
    target_values : numpy.ndarray
        The expected theoretical values of the polynomial.
    reconstructed_values : numpy.ndarray
        The actual reconstructed values (scaled by alpha, and mapped to the correct real/imag/mag axis).
    domain_label : str
        The label for the X-axis.
    title : str
        The main title of the plot.

    Examles
    -------

    ::

        import numpy as np
        import numpy.polynomial.polynomial as poly
        from qrisp.algorithms.gqsp.gqsp_angles import gqsp_angles
        from qrisp.algorithms.gqsp.gqsp_angles_verification import *

        # 1. Generate Domain & Target
        omega_domain = np.linspace(0, np.pi, 400)
        z_circle = np.exp(1j * omega_domain)

        target_coeffs = np.array([0.0, 1.1, 0.0, -1.0, 0.4, 0.7, 0.4, 0.4, 0.6])
        # Target is standard polynomial evaluated on z_circle
        target_vals_complex = poly.polyval(z_circle, target_coeffs)
        target_magnitude = np.abs(target_vals_complex)

        # 2. Reconstruct (using evaluate_gqsp_polynomial)
        (theta, phi, lambd), alpha = gqsp_angles(target_coeffs)
        U00_complex = evaluate_gqsp_polynomial(theta, phi, lambd, z_circle)
        reconstructed_magnitude = np.abs(U00_complex) * alpha

        # 3. Plot!
        plot_reconstruction_vs_target(
            domain=omega_domain,
            target_values=target_magnitude,
            reconstructed_values=reconstructed_magnitude,
            domain_label=r"Angle $\omega$ (where $z = e^{i\omega}$)",
            title="GQSP Reconstruction (Magnitude on Unit Circle)"
        )
    """
    # Calculate absolute error
    residuals = np.abs(reconstructed_values - target_values)
    max_error = np.max(residuals)

    # Setup the figure with 2 subplots (Main plot and Error plot)
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # --- Top Panel: Overlay ---
    ax1.plot(
        domain,
        target_values,
        label="Target (Input)",
        color="black",
        linewidth=5,
        alpha=0.3,
    )
    ax1.plot(
        domain,
        reconstructed_values,
        label="Reconstructed",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    ax1.set_title(title, fontsize=14)
    ax1.set_ylabel("Amplitude", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best", fontsize=11)

    # --- Bottom Panel: Residuals ---
    ax2.plot(
        domain,
        residuals,
        label=f"Absolute Error (Max: {max_error:.2e})",
        color="blue",
        linewidth=1.5,
    )
    ax2.fill_between(domain, 0, residuals, color="blue", alpha=0.1)

    ax2.set_xlabel(domain_label, fontsize=12)
    ax2.set_ylabel("Error $\Delta$", fontsize=12)
    ax2.set_yscale("log")  # Log scale is best for viewing numerical precision errors
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.show()


def evaluate_nlft_sequence(
    F_sequence: npt.NDArray[np.complex128],
    z_values: Union[npt.NDArray[np.complex128], npt.NDArray[np.float64]],
) -> npt.NDArray[np.complex128]:
    """
    Evaluates the Non-Linear Fourier Transform (NLFT) sequence by classically
    reconstructing the matrix response using forward synthesis on the unit circle.

    This function simulates the layer-by-layer mixing of the NLFT. Note that
    this specific implementation returns the top-right matrix element U_{01}(z),
    which typically encodes the complementary polynomial in QSP/NLFT frameworks.

    Parameters
    ----------
    F_sequence : numpy.ndarray
        A 1D array of complex numbers representing the reflection coefficients
        (the NLFT sequence) generated from layer-peeling algorithms.
    z_values : numpy.ndarray
        A 1D array of complex numbers (or floats) representing the evaluation
        points. Typically, these lie on the unit circle z = exp(i * omega).

    Returns
    -------
    numpy.ndarray
        A 1D complex128 array containing the evaluated complementary response
        U_{01}(z) for each input z in `z_values`.
    """
    z = np.asarray(z_values, dtype=complex)
    num_z = len(z)

    # Initialize the state (Identity matrix)
    U = np.zeros((num_z, 2, 2), dtype=complex)
    U[:, 0, 0] = 1.0
    U[:, 1, 1] = 1.0

    # Forward NLFT synthesis
    for F_k in F_sequence:
        # 1. The z-shift matrix
        Z = np.zeros((num_z, 2, 2), dtype=complex)
        Z[:, 0, 0] = z
        Z[:, 1, 1] = 1.0

        # 2. The NLFT SU(2) mixing layer
        norm = 1.0 / np.sqrt(1.0 + np.abs(F_k) ** 2)
        L = np.zeros((num_z, 2, 2), dtype=complex)
        L[:, 0, 0] = norm
        L[:, 0, 1] = F_k * norm
        L[:, 1, 0] = -np.conj(F_k) * norm
        L[:, 1, 1] = norm

        # Apply layer: U = L @ Z @ U
        # Note: Depending on the array endianness from the generator, you might
        # need to reverse the sequence of F_k.
        U = np.matmul(L, np.matmul(Z, U))

    return U[:, 0, 1]


def assert_nlft_sequence_match_target(
    F_sequence: npt.NDArray[np.complex128],
    alpha: float,
    target_coeffs: npt.NDArray[np.float64],
    num_test_points: int = 500,
    atol: float = 1e-6,
    check_magnitude_only: bool = True,
) -> None:
    """
    Tests if a Non-Linear Fourier Transform (NLFT) sequence accurately reconstructs
    the target polynomial on the complex unit circle.

    Parameters
    ----------
    F_sequence : numpy.ndarray
        A 1D array of complex numbers representing the reflection coefficients.
    alpha : float
        The scalar scaling factor used during sequence generation.
    target_coeffs : numpy.ndarray
        The standard polynomial coefficients (e.g., [a_0, a_1, a_2] for a_0 + a_1*z + a_2*z^2).
    num_test_points : int
        Number of points to evaluate on the unit circle [0, 2pi). Default is 500.
    atol : float
        Absolute tolerance for the maximum error. Default is 1e-6.
    check_magnitude_only : bool
        If True (default), checks if the absolute magnitudes match |U_01(z)| == |P(z)|,
        bypassing z^d phase shifts common in raw NLFT synthesis. If False, checks
        exact complex plane equality.

    Raises
    ------
    AssertionError
        If the maximum absolute error between the target and the reconstructed
        response exceeds `atol`.
    """
    # Evaluate over the full unit circle
    omegas = np.linspace(0, 2 * np.pi, num_test_points)
    z_circle = np.exp(1j * omegas)

    # 1. Evaluate target analytically using standard polynomial basis
    expected_values = poly.polyval(z_circle, target_coeffs)

    # 2. Evaluate classical matrix reconstruction
    u01_response = evaluate_nlft_sequence(F_sequence, z_circle)

    # 3. Scale back by alpha
    reconstructed_complex = u01_response * alpha

    # 4. Calculate error
    if check_magnitude_only:
        # Just compare |U_01| to |P(z)|
        diff = np.abs(reconstructed_complex) - np.abs(expected_values)
        max_error = np.max(np.abs(diff))
        error_type = "Magnitude Error"
    else:
        # Compare distance in the complex plane: |U_01 - P(z)|
        diff = reconstructed_complex - expected_values
        max_error = np.max(np.abs(diff))
        error_type = "Complex Plane Error"

    # 5. Assert with detailed error message
    error_msg = (
        f"\nNLFT Sequence Verification Failed!\n"
        f"Maximum {error_type}: {max_error:.2e} (Tolerance: {atol})\n"
        f"Sequence Length (d): {len(F_sequence)}\n"
        f"Ensure array endianness (F_sequence[::-1]) is correct for your convention."
    )

    assert max_error <= atol, error_msg


def evaluate_gqsp_polynomial(
    theta_angles: npt.NDArray[np.float64],
    phi_angles: npt.NDArray[np.float64],
    lambd: float,
    z_values: Union[npt.NDArray[np.complex128], npt.NDArray[np.float64]],
) -> npt.NDArray[np.complex128]:
    """
    Evaluates the Generalized Quantum Signal Processing (GQSP) polynomial response
    in the complex unit circle basis.

    This function simulates the GQSP unitary sequence as defined in Theorem 9
    of Laneve (arXiv:2503.03026) and returns the top-left matrix element
    U_{00}(z), which encodes the target polynomial.

    The sequence is defined as:
    U(z) = A_0 * W(z) * A_1 * W(z) * ... * W(z) * A_d

    Where:
    - W(z) = diag(z, 1) is the analytic signal operator.
    - A_0 = exp(i * lambd * Z) * exp(i * phi_0 * X) * exp(i * theta_0 * Z)
    - A_k = exp(i * phi_k * X) * exp(i * theta_k * Z) for k > 0

    Parameters
    ----------
    theta_angles : numpy.ndarray
        A 1D array of float64 representing the Z-rotation angles.
        Expected length is d + 1, where d is the polynomial degree.
    phi_angles : numpy.ndarray
        A 1D array of float64 representing the X-rotation angles.
        Expected length is d + 1.
    lambd : float
        The initial global Z-rotation phase factor (lambda).
    z_values : numpy.ndarray
        A 1D array of complex numbers representing the evaluation points.
        For standard GQSP, these should lie on the unit circle (z = exp(i * omega)).

    Returns
    -------
    numpy.ndarray
        A 1D complex128 array containing the evaluated polynomial response U_{00}(z)
        for each input z in `z_values`.

    Raises
    ------
    ValueError
        If `theta_angles` and `phi_angles` do not have the same length.
    """

    if len(theta_angles) != len(phi_angles):
        raise ValueError(
            "theta_angles and phi_angles must have the same length (d + 1)."
        )

    z = np.asarray(z_values, dtype=complex)
    num_z = len(z)
    d = len(phi_angles) - 1

    # Inner helper functions for vectorized Pauli rotations
    def exp_iX(angle: float) -> npt.NDArray[np.complex128]:
        mat = np.zeros((num_z, 2, 2), dtype=complex)
        mat[:, 0, 0] = np.cos(angle)
        mat[:, 1, 1] = np.cos(angle)
        mat[:, 0, 1] = 1j * np.sin(angle)
        mat[:, 1, 0] = 1j * np.sin(angle)
        return mat

    def exp_iZ(angle: float) -> npt.NDArray[np.complex128]:
        mat = np.zeros((num_z, 2, 2), dtype=complex)
        mat[:, 0, 0] = np.exp(1j * angle)
        mat[:, 1, 1] = np.exp(-1j * angle)
        return mat

    # A_0 = exp(i * lambd * Z) * exp(i * phi_0 * X) * exp(i * theta_0 * Z)
    U = np.matmul(
        exp_iZ(lambd), np.matmul(exp_iX(phi_angles[0]), exp_iZ(theta_angles[0]))
    )

    # Iteratively apply W(z) and A_k
    for k in range(1, d + 1):
        W = np.zeros((num_z, 2, 2), dtype=complex)
        W[:, 0, 0] = z
        W[:, 1, 1] = 1.0

        A_k = np.matmul(exp_iX(phi_angles[k]), exp_iZ(theta_angles[k]))
        U = np.matmul(U, np.matmul(W, A_k))

    return U[:, 0, 0]


def assert_gqsp_angles_match_target(
    theta_angles: npt.NDArray[np.float64],
    phi_angles: npt.NDArray[np.float64],
    lambd: float,
    alpha: float,
    target_coeffs: npt.NDArray[np.float64],
    num_test_points: int = 500,
    atol: float = 1e-6,
    check_magnitude_only: bool = False,
):
    """
    Tests if a set of GQSP angles accurately reconstructs the standard target polynomial.

    Parameters
    ----------
    theta_angles : numpy.ndarray
        The Z-rotation angles.
    phi_angles : numpy.ndarray
        The X-rotation angles.
    lambd : float
        The initial global Z-rotation phase.
    alpha : float
        The GQSP scaling factor.
    target_coeffs : numpy.ndarray
        The standard polynomial coefficients (e.g., [a_0, a_1, a_2] for a_0 + a_1*z + a_2*z^2).
    num_test_points : int
        Number of points to evaluate on the unit circle [0, 2pi).
    atol : float
        Absolute tolerance for the maximum error.
    check_magnitude_only : bool
        If True, only checks if the absolute magnitudes match |U_00(z)| == |P(z)|.
        If False, checks for exact equality in the complex plane (default).

    Raises
    ------
    AssertionError
        If the maximum absolute error between the target and the reconstructed
        response exceeds `atol`.
    """
    # Evaluate over the full unit circle
    omegas = np.linspace(0, 2 * np.pi, num_test_points)
    z_circle = np.exp(1j * omegas)

    # 1. Evaluate target analytically using standard polynomial basis
    expected_values = poly.polyval(z_circle, target_coeffs)

    # 2. Evaluate classical matrix reconstruction
    u00_response = evaluate_gqsp_polynomial(theta_angles, phi_angles, lambd, z_circle)

    # 3. Scale back by alpha
    reconstructed_complex = u00_response * alpha

    # 4. Calculate error
    if check_magnitude_only:
        # Just compare |U_00| to |P(z)|
        diff = np.abs(reconstructed_complex) - np.abs(expected_values)
        max_error = np.max(np.abs(diff))
        error_type = "Magnitude Error"
    else:
        # Compare distance in the complex plane: |U_00 - P(z)|
        diff = reconstructed_complex - expected_values
        max_error = np.max(np.abs(diff))
        error_type = "Complex Plane Error"

    # 5. Assert with detailed error message
    error_msg = (
        f"\nGQSP Angle Verification Failed!\n"
        f"Maximum {error_type}: {max_error:.2e} (Tolerance: {atol})\n"
        f"Polynomial Degree: {len(target_coeffs) - 1}\n"
        f"Ensure alpha is applied and angles are passed in the correct order."
    )

    assert max_error <= atol, error_msg


def evaluate_qsp_polynomial(
    angles: npt.NDArray[np.float64],
    x_values: Union[npt.NDArray[np.float64], npt.NDArray[np.complex128]],
    signal_basis: str = "X",
    phase_basis: str = "Z",
) -> npt.NDArray[np.complex128]:
    """
    Evaluates the classical Quantum Signal Processing (QSP) / QSVT matrix response
    with customizable bases for the signal and phase operators.

    This function simulates the QSP unitary sequence and returns the
    top-left matrix element U_{00}(x), which encodes the target polynomial.

    The sequence is defined as:
    U(x) = S(phi_0) * W(x) * S(phi_1) * ... * W(x) * S(phi_d)

    Parameters
    ----------
    angles : numpy.ndarray
        A 1D array of float64 representing the sequence of phase angles.
        Expected length is d + 1, where d is the polynomial degree.
    x_values : numpy.ndarray
        A 1D array of floats or complex numbers representing the evaluation points.
        For standard QSP, these should lie in the domain [-1, 1].
    signal_basis : str, optional
        The Pauli basis for the signal operator W(x).
        - 'X' (default): W_x(x) = exp(i * arccos(x) * X) = [[x, i*sqrt(1-x^2)], [i*sqrt(1-x^2), x]]
        - 'Z': W_z(x) = exp(i * arccos(x) * Z) = [[x + i*sqrt(1-x^2), 0], [0, x - i*sqrt(1-x^2)]]
    phase_basis : str, optional
        The Pauli basis for the phase operator S(phi).
        - 'Z' (default): S_z(phi) = exp(i * phi * Z) = [[exp(i*phi), 0], [0, exp(-i*phi)]]
        - 'X': S_x(phi) = exp(i * phi * X) = [[cos(phi), i*sin(phi)], [i*sin(phi), cos(phi)]]

    Returns
    -------
    numpy.ndarray
        A 1D complex128 array containing the evaluated polynomial response U_{00}(x)
        for each input x in `x_values`.

    Raises
    ------
    ValueError
        If an invalid `signal_basis` or `phase_basis` is provided.
    """
    x = np.asarray(x_values, dtype=complex)
    num_x = len(x)

    # Pre-compute the shared imaginary root for efficiency
    sqrt_term = 1j * np.sqrt(1 - x**2 + 0j)

    # Helper to generate the Phase Operator S(phi)
    def get_phase_operator(phi: float) -> npt.NDArray[np.complex128]:
        S = np.zeros((num_x, 2, 2), dtype=complex)
        if phase_basis.upper() == "Z":
            S[:, 0, 0] = np.exp(1j * phi)
            S[:, 1, 1] = np.exp(-1j * phi)
        elif phase_basis.upper() == "X":
            cos_phi = np.cos(phi)
            sin_phi = 1j * np.sin(phi)
            S[:, 0, 0] = cos_phi
            S[:, 1, 1] = cos_phi
            S[:, 0, 1] = sin_phi
            S[:, 1, 0] = sin_phi
        else:
            raise ValueError("phase_basis must be 'X' or 'Z'")
        return S

    # Helper to generate the Signal Operator W(x)
    def get_signal_operator() -> npt.NDArray[np.complex128]:
        W = np.zeros((num_x, 2, 2), dtype=complex)
        if signal_basis.upper() == "X":
            W[:, 0, 0] = x
            W[:, 1, 1] = x
            W[:, 0, 1] = sqrt_term
            W[:, 1, 0] = sqrt_term
        elif signal_basis.upper() == "Z":
            W[:, 0, 0] = x + sqrt_term
            W[:, 1, 1] = x - sqrt_term
            # Off-diagonals remain 0
        else:
            raise ValueError("signal_basis must be 'X' or 'Z'")
        return W

    # 1. Initialize U = S(phi_0)
    U = get_phase_operator(angles[0])

    # Only calculate W(x) once since it does not depend on phi
    if len(angles) > 1:
        W_x = get_signal_operator()

    # 2. Iteratively apply W(x) and S(phi_k)
    for phi in angles[1:]:
        S_k = get_phase_operator(phi)
        # Multiply: U = U @ W @ S
        U = np.matmul(U, np.matmul(W_x, S_k))

    return U[:, 0, 0]


def assert_qsp_angles_match_target(
    angles: npt.NDArray[np.float64],
    alpha: float,
    target_cheb_coeffs: npt.NDArray[np.float64],
    num_test_points: int = 500,
    atol: float = 1e-6,
    signal_basis: str = "X",
    phase_basis: str = "Z",
) -> None:
    """
    Tests if a set of QSP angles accurately reconstructs the target Chebyshev polynomial.

    Automatically searches the 4 standard complex axes (Re, Im, -Re, -Im) of U_{00}
    to account for the arbitrary global phase rotations introduced by switching
    between different signal (Wx vs Wz) and phase (Sx vs Sz) operator bases.

    Parameters
    ----------
    angles : numpy.ndarray
        The generated phase angles from the QSP algorithm.
    alpha : float
        The QSP scaling factor (usually <= 1.0).
    target_cheb_coeffs : numpy.ndarray
        The Chebyshev coefficients of the target polynomial in domain [-1, 1].
    num_test_points : int
        Number of points to evaluate in the domain [-1, 1]. Default is 500.
    atol : float
        Absolute tolerance for the maximum error. Default is 1e-6.
    signal_basis : str, optional
        The Pauli basis for the signal operator W(x). ('X' or 'Z').
    phase_basis : str, optional
        The Pauli basis for the phase operator S(phi). ('X' or 'Z').

    Raises
    ------
    AssertionError
        If the maximum absolute error between the target and the closest QSP axis
        mapping exceeds `atol`.
    """
    # 1. Generate Domain & Evaluate Target
    x_range = np.linspace(-1, 1, num_test_points)
    expected_values = cheb.chebval(x_range, target_cheb_coeffs)

    # 2. Evaluate classical matrix reconstruction (using the universal helper)
    u00_response = evaluate_qsp_polynomial(
        angles, x_range, signal_basis=signal_basis, phase_basis=phase_basis
    )

    # 3. Scale back by alpha
    reconstructed_real = u00_response.real * alpha
    reconstructed_imag = u00_response.imag * alpha

    # 4. Determine which axis houses the encoded polynomial
    # We calculate the max absolute error against all 4 possible axis alignments
    err_real = np.max(np.abs(reconstructed_real - expected_values))
    err_imag = np.max(np.abs(reconstructed_imag - expected_values))
    err_real_neg = np.max(np.abs(-reconstructed_real - expected_values))
    err_imag_neg = np.max(np.abs(-reconstructed_imag - expected_values))

    errors = {
        "Real (+1)": err_real,
        "Imaginary (+i)": err_imag,
        "Negative Real (-1)": err_real_neg,
        "Negative Imaginary (-i)": err_imag_neg,
    }

    best_match_component = min(errors, key=errors.get)
    min_error = errors[best_match_component]

    # 5. Assert with detailed debugging message
    error_msg = (
        f"\nQSP Angle Verification Failed!\n"
        f"Bases Used: Signal=W_{signal_basis.lower()}, Phase=S_{phase_basis.lower()}\n"
        f"Maximum Absolute Error: {min_error:.2e} (Tolerance: {atol})\n"
        f"Closest Component Axis: {best_match_component}\n"
        f"Angle Array Length: {len(angles)}\n"
        f"Check sequence endianness (angles[::-1]) or alpha scaling."
    )

    assert min_error <= atol, error_msg


def evaluate_qsvt_polynomial(
    angles: npt.NDArray[np.float64],
    x_values: Union[npt.NDArray[np.float64], npt.NDArray[np.complex128]],
) -> npt.NDArray[np.complex128]:
    """
    Evaluates the classical Quantum Singular Value Transformation (QSVT) matrix response.

    This function simulates the standard QSVT unitary sequence in the 2x2 subspace
    and returns the top-left matrix element U_{00}(x).

    Parameters
    ----------
    angles : numpy.ndarray
        A 1D array of QSVT phase angles of length d + 1.
    x_values : numpy.ndarray
        A 1D array of evaluation points in the domain [-1, 1].

    Returns
    -------
    numpy.ndarray
        A 1D complex128 array containing the evaluated response U_{00}(x).
    """
    x = np.asarray(x_values, dtype=complex)
    num_x = len(x)

    # Pre-compute the real square root for the off-diagonals
    sqrt_term = np.sqrt(1 - x**2 + 0j)

    # Helper to generate the QSVT Phase Operator S(phi)
    def get_phase_operator(phi: float) -> npt.NDArray[np.complex128]:
        S = np.zeros((num_x, 2, 2), dtype=complex)
        S[:, 0, 0] = np.exp(1j * phi)
        S[:, 1, 1] = np.exp(-1j * phi)
        return S

    # Helper to generate the QSVT Reflection Operator R(x)
    R = np.zeros((num_x, 2, 2), dtype=complex)
    R[:, 0, 0] = x
    R[:, 1, 1] = -x
    R[:, 0, 1] = sqrt_term
    R[:, 1, 0] = sqrt_term

    # 1. Initialize U = S(phi_0)
    U = get_phase_operator(angles[0])

    # 2. Iteratively apply R(x) and S(phi_k)
    for phi in angles[1:]:
        S_k = get_phase_operator(phi)
        # Multiply: U = U @ R @ S
        U = np.matmul(U, np.matmul(R, S_k))

    return U[:, 0, 0]


def assert_qsvt_angles_match_target(
    angles: npt.NDArray[np.float64],
    alpha: float,
    target_cheb_coeffs: npt.NDArray[np.float64],
    num_test_points: int = 500,
    atol: float = 1e-6,
) -> None:
    """
    Tests if a set of QSVT angles accurately reconstructs the target Chebyshev polynomial.

    Automatically searches the 4 standard complex axes (Re, Im, -Re, -Im) of U_{00}
    to account for the arbitrary global phase rotations introduced by switching
    between different signal (Wx vs Wz) and phase (Sx vs Sz) operator bases.

    Parameters
    ----------
    angles : numpy.ndarray
        The generated phase angles from the QSP algorithm.
    alpha : float
        The QSP scaling factor (usually <= 1.0).
    target_cheb_coeffs : numpy.ndarray
        The Chebyshev coefficients of the target polynomial in domain [-1, 1].
    num_test_points : int
        Number of points to evaluate in the domain [-1, 1]. Default is 500.
    atol : float
        Absolute tolerance for the maximum error. Default is 1e-6.

    Raises
    ------
    AssertionError
        If the maximum absolute error between the target and the closest QSP axis
        mapping exceeds `atol`.
    """
    x_range = np.linspace(-1, 1, num_test_points)
    expected_values = cheb.chebval(x_range, target_cheb_coeffs)

    # Evaluate classical QSVT matrix reconstruction
    u00_response = evaluate_qsvt_polynomial(angles, x_range)

    # Scale back by alpha
    reconstructed_real = u00_response.real * alpha
    reconstructed_imag = u00_response.imag * alpha

    # Determine which axis houses the encoded polynomial
    errors = {
        "Real (+1)": np.max(np.abs(reconstructed_real - expected_values)),
        "Imaginary (+i)": np.max(np.abs(reconstructed_imag - expected_values)),
        "Negative Real (-1)": np.max(np.abs(-reconstructed_real - expected_values)),
        "Negative Imaginary (-i)": np.max(
            np.abs(-reconstructed_imag - expected_values)
        ),
    }

    best_match_component = min(errors, key=errors.get)
    min_error = errors[best_match_component]

    error_msg = (
        f"\nQSVT Angle Verification Failed!\n"
        f"Maximum Absolute Error: {min_error:.2e} (Tolerance: {atol})\n"
        f"Closest Component Axis: {best_match_component}\n"
        f"Angle Array Length: {len(angles)}\n"
        f"Check sequence endianness or alpha scaling."
    )

    assert min_error <= atol, error_msg
