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
import jax.numpy as jnp
import jax
import numpy as np
from qrisp import QuantumVariable, QuantumFloat, h, control, conjugate
from qrisp.alg_primitives.reflection import reflection
from qrisp.jasp import jrange, q_cond, check_for_tracing_mode, expectation_value
import scipy


def inner_lanczos(H, k, state_prep_func):
    r"""
    Perform the quantum subroutine of the exact and efficient Lanczos method to estimate expectation values of Chebyshev polynomials of a Hamiltonian.

    This function implements the Krylov space construction via block-encodings 
    of Chebyshev polynomials $T_k(H)$, as described in 
    `"Exact and efficient Lanczos method on a quantum computer" <https://quantum-journal.org/papers/q-2023-05-23-1018/>`_.
    
    For each polynomial order $k = 0, \dots, 2D-1$, it prepares circuits corresponding 
    either to $\bra{\psi\lfloor k/2\rfloor}R\ket{\psi\lfloor k/2\rfloor}$ for even $k$, or
    $\bra{\psi\lfloor k/2\rfloor}U\ket{\psi\lfloor k/2\rfloor}$ for odd $k$. 
    The measured statistics of the prepared quantum states encode the expectation values $\langle T_k(H)\rangle$. 

    Parameters
    ----------
    H : QubitOperator
        Hamiltonian for which to estimate the ground-state energy.
    D : int
        Krylov space dimension. Determines maximum Chebyshev order $(2D-1)$.
    state_prep_func : callable 
        Function preparing the initial system state $\ket{\psi_0}$ on a (operand) QuantumVariable.

    Returns
    -------
    QuantumVariable
        A QuantumVariable for which the measured statistics encodes the expectation values $\langle T_k(H)\rangle$. 

    """

    # Extract unitaries and size of the case_indicator QuantumFloat.
    U, state_prep, n = H.pauli_block_encoding()

    def UR(case_indicator, operand):
        U(case_indicator, operand) # applies $U = \sum_i\ket{i}\bra{i}\otimes P_i$.
        reflection(case_indicator, state_function=state_prep) # reflection operator R about $\ket{G}$.

    case_indicator = QuantumFloat(n)
    operand = QuantumVariable(H.find_minimal_qubit_amount())
    state_prep_func(operand) # prepare operand $\ket{\psi_0}$

    def even(case_indicator, operand, k):
        # EVEN k: Figure 1 top
        with conjugate(state_prep)(case_indicator):
            for _ in jrange(k//2):
                UR(case_indicator, operand)
        return case_indicator

    def odd(case_indicator, operand, k):
        # ODD k: Figure 1 bottom
        state_prep(case_indicator)
        for _ in jrange(k//2):
            UR(case_indicator, operand)
        qv = QuantumFloat(1)
        h(qv) # Hadamard test for <U>
        with control(qv[0]):
            U(case_indicator, operand) # control-U on the case_indicator QuantumFloat
        h(qv) # Hadamard test for <U>
        return qv
    
    if check_for_tracing_mode():
        x_cond = q_cond
    else:
        def x_cond(pred, true_fun, false_fun, *operands):
            if pred:
                return true_fun(*operands)
            else:
                return false_fun(*operands)
            
    return x_cond(k%2==0, even, odd, case_indicator, operand, k)


def compute_expectation(meas_res):
    r"""
    Convert measurement results into an expectation value.

    Assumes measurement outcomes correspond to $\pm 1$ eigenvalues of observables 
    (reflection $R$ or block-encoding unitary $U$). 

    For even $k$ we measure the auxilary case_indicator QuantumFloat in the computational basis. We then map:
    $+1$ if all-zeros outcome (projector $2\ket{0}\bra{0} - \mathbb{I}$), else $-1$ (paper step 2).

    For odd $k$ we perform the Hadamard test for the block-encoding unitary $U$ and then measure the 
    auxilary QuantumVariable in the $Z$ basis.

    Parameters
    ----------
    meas_res : dict
        A dictionary of values and their corresponding measurement probabilities.

    Returns
    -------
    expval : float
        Expectation value of the measured observable.

    """
    expval = 0.0
    for outcome, prob in meas_res.items():

        if int(outcome) == 0:
            expval += prob * 1
        else:
            expval += prob * (-1)
    return expval


def lanczos_expvals(H, D, state_prep_func, mes_kwargs={}):
    r"""
    Perform the quantum subroutine of the exact and efficient Lanczos method to estimate expectation values of Chebyshev polynomials of a Hamiltonian.

    This function implements the Krylov space construction via block-encodings 
    of Chebyshev polynomials $T_k(H)$, as described in 
    `"Exact and efficient Lanczos method on a quantum computer" <https://quantum-journal.org/papers/q-2023-05-23-1018/>`_.
    
    For each polynomial order $k = 0, \dotsc, 2D-1$, it prepares and measures circuits corresponding 
    either to $\bra{\psi\lfloor k/2\rfloor}R\ket{\psi\lfloor k/2\rfloor}$ for even $k$, or
    $\bra{\psi\lfloor k/2\rfloor}U\ket{\psi\lfloor k/2\rfloor}$ for odd $k$. 
    The measured statistics encode the expectation values $\langle T_k(H)\rangle$. 

    Parameters
    ----------
    H : QubitOperator
        Hamiltonian for which to estimate the ground-state energy.
    D : int
        Krylov space dimension. Determines maximum Chebyshev order $(2D-1)$.
    state_prep_func : callable 
        Function preparing the initial system state $\ket{\psi_0}$ on a (operand) QuantumVariable.
    mes_kwargs : dict, optional
        The keyword arguments for the measurement function.

    Returns
    -------
    expvals : ndarray
        The expectation values $\langle T_k(H)\rangle$ for $k=0, \dotsc, 2D-1$.

    """

    # Set default options
    if not "shots" in mes_kwargs:
        mes_kwargs["shots"] = 100000

    if check_for_tracing_mode():

        @jax.jit
        def post_processor(x):
            """
            Returns 1 if the input integer x is 0, and -1 otherwise, using jax.numpy.where.
            """
            return jnp.where(x == 0, 1, -1)
        
        ev_function = expectation_value(inner_lanczos, shots = mes_kwargs["shots"], post_processor = post_processor)
        expvals = jnp.zeros(2*D)

        for k in range(0, 2*D):
            expval = ev_function(H, k, state_prep_func)
            expvals = expvals.at[k].set(expval)

    else:
        expvals = np.zeros(2*D)
        for k in range(0, 2*D):
            qarg = inner_lanczos(H, k, state_prep_func)
            meas = qarg.get_measurement(**mes_kwargs)
            expvals[k] = compute_expectation(meas)
    
    return expvals


def build_S_H_from_Tk(Tk_expvals, D):
    r"""
    Construct the overlap matrix $\mathbf{S}$ and the Krylov Hamiltonian matrix $\mathbf{H}$ from Chebyshev polynomial expectation values.

    Using Chebyshev recurrence relations, this function generates the matrix elements for
    both the overlap matrix ($\mathbf{S}$) and the Hamiltonian matrix ($\mathbf{H}$) in the Krylov subspace.
    The approach follows Equations (17) and (19) in 
    `"Exact and efficient Lanczos method on a quantum computer" <https://quantum-journal.org/papers/q-2023-05-23-1018/>`_.

    Parameters
    ----------
    Tk_expectation : ndarray
        Dictionary of expectations $⟨T_k(H)⟩$ for each Chebyshev polynomial order $k$.
    D : int
        Krylov space dimension.

    Returns
    -------
    S : ndarray
        Overlap (Gram) matrix $\mathbf{S}$ for Krylov states.
    H_mat : ndarray
        Hamiltonian matrix $\mathbf{H}$ in Krylov subspace.

    """
    def Tk_vec(k):
        k = np.abs(k)
        return Tk_expvals[k]

    # Create 2D arrays of indices i and j
    i_indices = np.arange(D, dtype=np.int32)[:, None] # Column vector (D, 1)
    j_indices = np.arange(D, dtype=np.int32)[None, :] # Row vector (1, D)
    # The combination of these two will broadcast operations across a (D, D) grid

    # Calculate S matrix using vectorized operations
    # i+j and abs(i-j) are performed element-wise across the (D, D) grid
    S = 0.5 * (Tk_vec(i_indices + j_indices) + Tk_vec(np.abs(i_indices - j_indices)))

    # Calculate H_mat matrix using vectorized operations
    H_mat = 0.25 * (
        Tk_vec(i_indices + j_indices + 1)
        + Tk_vec(np.abs(i_indices + j_indices - 1))
        + Tk_vec(np.abs(i_indices - j_indices + 1))
        + Tk_vec(np.abs(i_indices - j_indices - 1))
    )

    return S, H_mat


@partial(jax.jit, static_argnums=(1,))
def build_S_H_from_Tk_jax(Tk_expvals, D):

    def Tk_vec(k):
        k = jnp.abs(k)
        return Tk_expvals[k]

    # Create 2D arrays of indices i and j
    i_indices = jnp.arange(D, dtype=jnp.int32)[:, None] # Column vector (D, 1)
    j_indices = jnp.arange(D, dtype=jnp.int32)[None, :] # Row vector (1, D)
    # The combination of these two will broadcast operations across a (D, D) grid

    # Calculate S matrix using vectorized operations
    # i+j and abs(i-j) are performed element-wise across the (D, D) grid
    S = 0.5 * (Tk_vec(i_indices + j_indices) + Tk_vec(jnp.abs(i_indices - j_indices)))

    # Calculate H_mat matrix using vectorized operations
    H_mat = 0.25 * (
        Tk_vec(i_indices + j_indices + 1)
        + Tk_vec(jnp.abs(i_indices + j_indices - 1))
        + Tk_vec(jnp.abs(i_indices - j_indices + 1))
        + Tk_vec(jnp.abs(i_indices - j_indices - 1))
    )

    return S, H_mat


def regularize_S_H(S, H_mat, cutoff=1e-3):
    r"""
    Regularize the overlap matrix $\mathbf{S}$ by retaining only eigenvectors with sufficiently large eigenvalues and project the Hamiltonian matrix $\mathbf{H}$ (``H_mat``) accordingly.

    This function applies a spectral cutoff: only directions in the Krylov subspace with eigenvalues
    above ``cutoff * max_eigenvalue`` are kept. Both the overlap matrix ($\mathbf{S}$) and the Hamiltonian matrix ($\mathbf{H}$)
    are projected onto this reduced subspace, ensuring numerical stability for subsequent
    generalized eigenvalue calculations.
    
    Parameters
    ----------
    S : ndarray
        Overlap matrix.
    H_mat : ndarray
        Hamiltonian matrix.
    cutoff : float
        Eigenvalue threshold for regularizing $\mathbf{S}$.

    Returns
    -------
    S_reg : ndarray
        Regularized overlap matrix.
    H_reg : ndarray
        Regularized Hamiltonian matrix in Krylov subspace.

    """
    eigvals, eigvecs = np.linalg.eigh(S)
    max_eigval = eigvals.max()
    threshold = cutoff * max_eigval
    mask = eigvals > threshold
    U = eigvecs[:, mask]
    S_reg = U.T @ S @ U
    H_reg = U.T @ H_mat @ U
    return S_reg, H_reg


@partial(jax.jit, static_argnums=(2,)) # max_D_out must be static
def regularize_S_H_jax(S, H_mat, max_D_out, cutoff=1e-3):
    eigvals, eigvecs = jnp.linalg.eigh(S)
    
    max_eigval = eigvals.max()
    threshold = cutoff * max_eigval
    
    mask = eigvals > threshold
    
    # Use nonzero with a fixed size and fill_value to handle static shapes
    kept_indices = jnp.nonzero(mask, size=max_D_out, fill_value=0)[0]
    
    # Use jnp.take to select eigenvectors. The resulting U has a static shape.
    # We might need to ensure the indices array has the correct shape for broadcasting
    U = jnp.take(eigvecs, kept_indices, axis=1)

    S_reg = U.T @ S @ U
    H_reg = U.T @ H_mat @ U
    
    # Note: S_reg and H_reg will be of shape (max_D_out, max_D_out)
    return S_reg, H_reg


def lanczos_alg(H, D, state_prep_func, mes_kwargs={}, cutoff=1e-2, show_info=False):
    r"""
    Exact and efficient Lanczos method on a quantum computer for ground state energy estimation.

    This function implements the Lanczos method on a quantum computer using block-encodings of Chebyshev
    polynomials $T_k(H)$, closely following the algorithm proposed in
    `"Exact and efficient Lanczos method on a quantum computer" <https://quantum-journal.org/papers/q-2023-05-23-1018/>`_.

    The quantum Lanczos algorithm efficiently constructs a Krylov subspace by applying Chebyshev polynomials
    of the Hamiltonian to an initial state. The Krylov space dimension $D$ determines the accuracy of ground
    state energy estimation, with convergence guaranteed when the initial state has a sufficiently large overlap 
    ($|\gamma_0|=\Omega(1/\text{poly}(n))$ for $n$ qubits) with the true ground state. The Chebyshev approach allows exact
    Krylov space construction (up to sample noise) without real or imaginary time evolution.

    This algorithm is motivated by the rapid convergence of the Lanczos method for estimating extremal eigenvalues,
    and its quantum version avoids the classical barrier of exponential cost in representing Krylov vectors.

    Implements the following steps:
      1. Run quantum Lanczos subroutine to obtain Chebyshev expectation values $\langle T_k(H)\rangle$.
      2. Build overlap and Hamiltonian subspace matrices $(\mathbf{S}, \mathbf{H})$.
      3. Regularize overlap matrix $\mathbf{S}$ and $\mathbf{H}$ by projecting onto the subspace with well conditioned eigenvalues.
      4. Solve generalized eigenvalue problem $\mathbf{H}\vec{v}=\epsilon\mathbf{S}\vec{v}$.
      5. Return lowest eigenvalue $\epsilon_{\text{min}}$ as ground state energy estimate.

    Parameters
    ----------
        H : QubitOperator
            Hamiltonian for which to estimate the ground-state energy.
        D : int
            Krylov space dimension.
        state_prep_func : callable 
            Function preparing the initial system state $\ket{\psi_0}$ on a (operand) QuantumVariable.
        mes_kwargs : dict
            The keyword arguments for the measurement function.
        cutoff : float
            Regularization cutoff threshold for overlap matrix $\mathbf{S}$. The default is 1e-2.
        show_info : bool
            If True, a dictionary with detailed information is returned. The default is False.

    Returns
    -------
    energy : float
        Estimated ground state energy of the Hamiltonian H.
    info : dict, optional
        Full details including:
            - 'Tk_expvals': dictionary of Chebyshev expectation values
            - 'energy': ground-state energy estimate
            - 'eigvals': eigenvalues of regularized problem
            - 'eigvecs': eigenvectors of regularized problem
            - 'S_reg': regularized overlap matrix
            - 'H_reg': regularized Hamiltonian matrix
    
    Examples
    --------

    ::

        from qrisp.lanczos import lanczos_alg
        from qrisp.operators import X, Y, Z
        from qrisp.vqe.problems.heisenberg import create_heisenberg_init_function
        import networkx as nx

        L = 6
        G = nx.Graph()
        G.add_edges_from([(k, (k+1) % L) for k in range(L - 1)])

        # Define Hamiltonian e.g. Heisenberg with custom couplings
        H = (1/4)*sum((X(i)*X(j) + Y(i)*Y(j) + 0.5*Z(i)*Z(j)) for i,j in G.edges())

        print(f"Ground state energy: {H.ground_state_energy()}")

        # Prepare initial state function (tensor product of singlets)
        M = nx.maximal_matching(G)
        U_singlet = create_heisenberg_init_function(M)

        D = 6  # Krylov dimension
        energy, info = lanczos_alg(H, D, U_singlet, show_info=True)

        print(f"Ground state energy estimate: {energy}")


    """

    if check_for_tracing_mode():
        raise NotImplementedError("Solving the generalized eigenvalue problem not implemented in JAX 0.6")
    
    unitaries, coeffs = H.unitaries()
    
    # Step 1: Quantum Lanczos: Get expectation values of Chebyshev polynomials
    Tk_expvals = lanczos_expvals(H, D, state_prep_func, mes_kwargs)

    #if check_for_tracing_mode():

        # Step 2: Build matrices S and H
        #S, H_mat = build_S_H_from_Tk_jax(Tk_expvals, D)

        # Step 3: Regularize matrices via thresholding
        #S_reg, H_reg = regularize_S_H_jax(S, H_mat, D, cutoff=cutoff)  

        # Step 4: Solve generalized eigenvalue problem $\mathbf{H}\vec{v}=\epsilon\mathbf{S}\vec{v}$
        #evals, evecs = solve_generalized_eigenproblem_jax(H_reg, S_reg)
        #evals, evecs = jax.scipy.linalg.eigh(H_reg, S_reg) # Solving the generalized eigenvalue problem not implemented in JAX 0.6

        #ground_state_energy = jnp.min(evals) * jnp.sum(coeffs)


    # Step 2: Build matrices S and H
    S, H_mat = build_S_H_from_Tk(Tk_expvals, D)

    # Step 3: Regularize matrices via thresholding
    S_reg, H_reg = regularize_S_H(S, H_mat, cutoff=cutoff)  

    # Step 4: Solve generalized eigenvalue problem $\mathbf{H}\vec{v}=\epsilon\mathbf{S}\vec{v}$
    evals, evecs = scipy.linalg.eigh(H_reg, S_reg) 

    ground_state_energy = np.min(evals) * np.sum(coeffs)

    
    results = {
        'Tk_expvals': Tk_expvals,
        'energy': ground_state_energy,
        'eigvals': evals,
        'eigvecs': evecs,
        'S_reg': S_reg,
        'H_reg': H_reg,
    }
    
    if show_info:
        return ground_state_energy, results
    else:
        return ground_state_energy