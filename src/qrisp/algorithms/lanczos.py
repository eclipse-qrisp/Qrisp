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

from __future__ import annotations
from functools import partial
import jax.numpy as jnp
import jax
import numpy as np
from qrisp import QuantumVariable, QuantumBool, h, control, multi_measurement
from qrisp.operators import QubitOperator
from qrisp.block_encodings import BlockEncoding
from qrisp.jasp import jrange, check_for_tracing_mode, expectation_value
from typing import Any, TYPE_CHECKING, Callable, Dict, Tuple

if TYPE_CHECKING:
    from jax.typing import ArrayLike

def lanczos_even(BE: BlockEncoding, k: int, operand_prep: Callable[..., Any]) -> Tuple[QuantumVariable, ...]: 
    r"""
    This function implements the Krylov space construction via block-encodings 
    of Chebyshev polynomials $T_k(H)$, following the layout in Figure 1(a) of `Kirby et al. <https://quantum-journal.org/papers/q-2023-05-23-1018>`__.
    
    For even $k$, the subroutine prepares a state by applying $k/2$ qubitization 
    steps $(RU)$. The expectation value $\langle T_k(H)\rangle_0$ is then obtained 
    by measuring the reflection operator $R = 2|0\rangle_a\langle 0|_a - I$ on 
    the ancillas. 
    
    The "all-zeros" measurement outcome (representing $|0\rangle_a$) corresponds 
    to the $+1$ eigenvalue of $R$, while any other outcome corresponds to $-1$.

    Parameters
    ----------
    BE : BlockEncoding
        The block-encoding of the Hamiltonian $H$ for which we want to estimate the ground-state energy.
    k : int
        Even integer representing the even Chebyshev polynomial order.
    operand_prep : Callable 
        Function returning the (operand) QuantumVariables in the initial system state $\ket{\psi_0}$, i.e.,
        ``operands=operand_prep()``. Must return a QuantumVariable or a tuple of QuantumVariables.

    Returns
    -------
    tuple of QuantumVariable
        The ancilla QuantumVariables. Measurement outcomes of these variables encode the expectation value.
    """
    BE_qubitized = BE.qubitization()

    ancillas = BE_qubitized.create_ancillas()
    
    operands = operand_prep()
    if not isinstance(operands, tuple):
        operands = (operands,)

    for _ in jrange(k // 2):
        BE_qubitized.unitary(*ancillas, *operands)
    
    return tuple(ancillas)

def lanczos_odd(BE: BlockEncoding, k: int, operand_prep: Callable[..., Any]) -> QuantumBool:
    r"""
    This function implements the Krylov space construction via block-encodings 
    of Chebyshev polynomials $T_k(H)$, following the layout in Figure 1(b) of `Kirby et al. <https://quantum-journal.org/papers/q-2023-05-23-1018/>`__.
    
    For odd $k$, the subroutine applies $\lfloor k/2 \rfloor$ qubitization steps 
    followed by a Hadamard test for the block-encoding unitary $U$. This 
    effectively estimates $\langle \psi_{\lfloor k/2 \rfloor} | U | \psi_{\lfloor k/2 \rfloor} \rangle$, 
    which encodes the odd Chebyshev expectation value $\langle T_k(H)\rangle_0$.

    Parameters
    ----------
    BE : BlockEncoding
        The block-encoding of the Hamiltonian $H$ for which we want to estimate the ground-state energy.
    k : int
        Odd integer representing the odd Chebyshev polynomial order.
    operand_prep : Callable 
        Function returning the (operand) QuantumVariables in the initial system state $\ket{\psi_0}$, i.e.,
        ``operands=operand_prep()``. Must return a QuantumVariable or a tuple of QuantumVariables.

    Returns
    -------
    QuantumBool
        QuantumBool used as the Hadamard test ancilla. The expectation value is derived 
        from the Z-basis measurement statistics ($P(0) - P(1)$).
    """
    BE_qubitized = BE.qubitization()

    ancillas = BE_qubitized.create_ancillas()
    
    operands = operand_prep()
    if not isinstance(operands, tuple):
        operands = (operands,)
    
    for _ in jrange(k // 2):
        BE_qubitized.unitary(*ancillas, *operands)
    
    qv = QuantumBool()
    h(qv) 
    with control(qv[0]):
        BE.unitary(*ancillas, *operands) 
    h(qv)
    
    return qv

def compute_expectation(meas_res: Dict[object, float]) -> float:
    r"""
    Convert measurement results from Lanczos subroutines into the expectation 
    value of a Chebyshev polynomial $\langle T_k(H) \rangle_0$.

    This function processes the output of `:ref: lanczos_even` and `lanczos_odd`` constructing the circuits described in 
    `Kirby et al. <https://quantum-journal.org/papers/q-2023-05-23-1018/>`__ to extract the physical 
    expectation value from the ancilla measurement statistics.

    Assumes measurement outcomes correspond to $\pm 1$ eigenvalues of observables 
    (reflection $R$ or block-encoding unitary $U$). 

    For even $k$,  he subroutine returns a tuple of outcomes for the auxiliary 
    ``case_indicator`` QuantumVariable. Since the observable is the reflection 
    operator $R = 2|0\rangle_a\langle 0|_a - I$, the outcome is mapped to 
    $+1$ if the ancilla is in the ground state $|0\rangle_a$ (all-zeros), 
    and $-1$ otherwise.
    
    For odd $k$, the subroutine returns the outcome of a single Hadamard test ancilla. 
    The expectation value of the block-encoding unitary $U$ is given by 
    the $Z$-basis contrast: $P(0) - P(1)$.

    Parameters
    ----------
    meas_res : dict
        A dictionary where keys are measurement outcomes (integers for scalar 
        variables or tuples for multiple variables) and values are their 
        respective probabilities.

    Returns
    -------
    expval : float
        The estimated expectation value $\langle T_k(H) \rangle_0$. 
        In the absence of noise, this value is exact.

    """
    expval = 0.0
    for outcome, prob in meas_res.items():
        if isinstance(outcome, tuple):
            is_zero = all(v == 0 for v in outcome)
        else:
            is_zero = (int(outcome) == 0)

        if is_zero:
            expval += prob
        else:
            expval -= prob
    return expval


def lanczos_expvals(H: BlockEncoding | QubitOperator, D: int, operand_prep: Callable[..., Any], mes_kwargs: Dict[str, object] = {}) -> "ArrayLike":

    r"""
    Estimate the expectation values of Chebyshev polynomials $\langle T_k(H) \rangle_0$ 
    for the exact and efficient Quantum Lanczos method.

    This function constructs the Krylov space basis by evaluating the expectation 
    values of Chebyshev polynomials up to order $2D-1$. It dispatches tasks to 
    :func:`lanczos_even` and :func:`lanczos_odd`, which implement the circuit 
    layouts described in `Figure 1 in Kirby et al. <https://quantum-journal.org/papers/q-2023-05-23-1018/>`__.
    
    For each polynomial order $k = 0, \dotsc, 2D-1$, it prepares and measures circuits corresponding 
    either to $\bra{\psi\lfloor k/2\rfloor}R\ket{\psi\lfloor k/2\rfloor}$ for even $k$, or
    $\bra{\psi\lfloor k/2\rfloor}U\ket{\psi\lfloor k/2\rfloor}$ for odd $k$. 
    The measured statistics encode the expectation values $\langle T_k(H)\rangle_0$. 

    The function supports two execution modes:

    - Tracing Mode (JAX): 
       Uses :func:`expectation_value` with a JIT-compiled 
       post-processor for high-performance execution.
    - Standard Mode: 
       Uses :func:`multi_measurement` for NISQ hardware execution.
    
    Parameters
    ----------
    H : QubitOperator or BlockEncoding
        Hamiltonian for which to estimate the ground-state energy. If a 
        QubitOperator is provided, it is automatically converted to a 
        Pauli block-encoding.
    D : int
        Krylov space dimension. Determines maximum Chebyshev order $(2D-1)$.
    operand_prep : callable 
        Function returning the (operand) QuantumVariables in the initial system state $\ket{\psi_0}$, i.e.,
        ``operands=operand_prep()``. Must return a QuantumVariable or a tuple of QuantumVariables.
    mes_kwargs : dict, optional
        The keyword arguments for the measurement function.
        By default, 100_000 ``shots`` are executed for measuring each expectation value.

    Returns
    -------
    expvals : ArrayLike, shape (2D,)
        The expectation values $\langle T_k(H) \rangle_0$ for $k=0, \dots, 2D-1$.

    """

    # Set default options
    if not "shots" in mes_kwargs:
        mes_kwargs["shots"] = 100000

    BE = H if isinstance(H, BlockEncoding) else BlockEncoding.from_operator(H)

    if check_for_tracing_mode():

        @jax.jit
        def post_processor(*args):
            """
            Maps the 'all-zeros' outcome to 1 and any other outcome to -1.
            """
            return jnp.where(jnp.all(jnp.array(args)) == 0, 1, -1)
        
        ev_even = expectation_value(lanczos_even, shots=mes_kwargs["shots"], post_processor=post_processor)
        ev_odd = expectation_value(lanczos_odd, shots=mes_kwargs["shots"], post_processor=post_processor)
        expvals = jnp.zeros(2*D)

        for k in range(0, 2*D):
            if k % 2 == 0:
                val = ev_even(BE, k, operand_prep)
            else:
                val = ev_odd(BE, k, operand_prep)
            expvals = expvals.at[k].set(val)

    else:
        expvals = np.zeros(2*D)
        for k in range(0, 2*D):
            if k % 2 == 0:
                qargs = lanczos_even(BE, k, operand_prep)
                meas = multi_measurement(list(qargs), **mes_kwargs)
            else:
                qarg = lanczos_odd(BE, k, operand_prep)
                meas = qarg.get_measurement(**mes_kwargs)
            expvals[k] = compute_expectation(meas)
    
    return expvals

# Postprocessing

@jax.jit
def build_S_H_from_Tk(expvals: "ArrayLike") -> Tuple["ArrayLike", "ArrayLike"]:
    r"""
    Construct the overlap matrix $\mathbf{S}$ and the Krylov Hamiltonian matrix $\mathbf{H}$ from Chebyshev polynomial expectation values.

    Using Chebyshev recurrence relations, this function generates the matrix elements for
    both the overlap matrix ($\mathbf{S}$) and the Hamiltonian matrix ($\mathbf{H}$) in the Krylov subspace.
    The approach follows `Equations (17) and (19) in Exact and efficient Lanczos method on a quantum computer <https://quantum-journal.org/papers/q-2023-05-23-1018/>`__.

    Parameters
    ----------
    expvals : ArrayLike, shape (2D,)
        The expectation values $\langle T_k(H)\rangle_0$ for each Chebyshev polynomial order $k$.

    Returns
    -------
    S : ArrayLike, shape (D, D)
        The (Gram) matrix $\mathbf{S}$ for Krylov states.
    H_mat : ArrayLike, shape (D, D)
        The Hamiltonian matrix $\mathbf{H}$ in Krylov subspace.

    """
    def Tk_vec(k):
        k = jnp.abs(k)
        return expvals[k]

    D = expvals.shape[0] // 2
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


@jax.jit
def regularize_S_H(S: "ArrayLike", H_mat: "ArrayLike", cutoff: float = 1e-2) -> Tuple["ArrayLike", "ArrayLike", "ArrayLike"]:
    r"""
    Regularize the overlap matrix $\mathbf{S}$ by retaining only eigenvectors with sufficiently large eigenvalues and project the Hamiltonian matrix $\mathbf{H}$ accordingly.

    This function applies a spectral cutoff: only directions in the Krylov subspace with eigenvalues
    above ``cutoff * max_eigenvalue`` are kept. Both the overlap matrix ($\mathbf{S}$) and the Hamiltonian matrix ($\mathbf{H}$)
    are projected onto this reduced subspace, ensuring numerical stability for subsequent
    generalized eigenvalue calculations. The regularized matrices are caculated as $\tilde{S} = V^TSV$ and $\tilde{H}=V^THV$ for a projection matrix $V$.
    
    Parameters
    ----------
    S : ArrayLike, shape (D, D)
        The overlap matrix.
    H_mat : ArrayLike, shape (D, D)
        The Hamiltonian matrix.
    cutoff : float
        Eigenvalue threshold for regularizing $\mathbf{S}$.

    Returns
    -------
    S_reg : ArrayLike, shape (D, D)
        The regularized overlap matrix.
    H_reg : ArrayLike, shape (D, D)
        The regularized Hamiltonian matrix in Krylov subspace.
    V : ArrayLike, shape (D, D)
        The projection matrix.
    """
    D = S.shape[0]

    eigvals, eigvecs = jnp.linalg.eigh(S)
    
    max_eigval = eigvals.max()
    threshold = cutoff * max_eigval
    
    mask = eigvals > threshold
    
    # Use nonzero with a fixed size and fill_value to handle static shapes
    kept_indices = jnp.nonzero(mask, size=D, fill_value=0)[0]
    
    # Use jnp.take to select eigenvectors. The resulting U has a static shape.
    # We might need to ensure the indices array has the correct shape for broadcasting
    U = jnp.take(eigvecs, kept_indices, axis=1)

    # Replace the vectors for the remaining indices by all zeros vectors
    new_mask = jnp.sort(mask)[::-1]
    zeros = jnp.zeros(D)
    V = jnp.zeros((D, D), dtype=U.dtype)
    V = V.at[jnp.arange(D), :].set(jnp.where(new_mask, U, zeros))

    S_reg = V.T @ S @ V
    H_reg = V.T @ H_mat @ V

    I = jnp.eye(D)
    S_reg = jnp.where(new_mask, S_reg, I)
    H_reg =jnp.where(new_mask, H_reg, I)
    
    # S_reg and H_reg will be a block matrix of shape (D, D):
    # upper left: k x k block of regularized matrices S_reg, H_reg, respectively; 
    # lower right: (D-k) x (D-k) identity; 
    # upper right and lower left are zeros
    return S_reg, H_reg, V


# jax.scipy.linalg.eigh does currently not support the generalized eigenvalue problem
@jax.jit
def generalized_eigh(A: "ArrayLike", B: "ArrayLike") -> Tuple["ArrayLike", "ArrayLike"]:
    r"""
    Solves the generalized eigenvalue problem $A v = \lambda B v$
    for a complex Hermitian or real symmetric matrix $A$ and a real symmetric positive-definite matrix $B$.

    Parameters
    ----------
    A : ArrayLike, shape (D, D)
        complex Hermitian or real symmetrix matrix.
    B : ArrayLike, shape (D, D)
        A real symmetric positive-definite matrix.

    Returns
    -------
    eigvals : ArrayLike, shape (D,)
        The generalized eigenvalues.
    eigvecs : ArrayLike, shape (D, D)
        The generalized eigenvectors.
    """
    # Compute Cholesky decomposition of B (L*L.T)
    # The 'lower=True' parameter is typically the default but good to be explicit.
    L = jax.scipy.linalg.cholesky(B, lower=True)
    
    # Compute L_inv * A * L_inv_T. 
    # Use solve_triangular for efficiency and numerical stability instead of explicit inverse.
    A_prime = jax.scipy.linalg.solve_triangular(L, A, lower=True)
    A_prime = jax.scipy.linalg.solve_triangular(L.T, A_prime.T, lower=False).T

    # Solve the standard eigenvalue problem for A'
    eigvals, eigvecs_prime = jax.scipy.linalg.eigh(A_prime)
    
    # Transform eigenvectors back to the original basis (v = L_inv_T * v_prime)
    eigvecs = jax.scipy.linalg.solve_triangular(L.T, eigvecs_prime, lower=False)
    
    # Normalize eigenvectors if needed (optional)
    eigvecs = eigvals / jnp.linalg.norm(eigvecs, axis=0)

    return eigvals, eigvecs


def lanczos_alg(H: BlockEncoding | QubitOperator, D: int, operand_prep: Callable[..., Any], mes_kwargs: Dict[str, object] = {}, cutoff: float = 1e-2, show_info: bool = False):
    r"""
    Estimate the ground state energy of a Hamiltonian using the `Exact and efficient Lanczos method on a quantum computer <https://quantum-journal.org/papers/q-2023-05-23-1018/>`__.

    This function implements the algorithm proposed in Kirby et al. by constructing a Krylov subspace using block-encodings of Chebyshev polynomials $T_k(H)$, 
    bypassing the need for real or imaginary time evolution.

    The quantum Lanczos algorithm efficiently constructs a Krylov subspace by applying Chebyshev polynomials
    of the Hamiltonian $T_k(H)$ to an initial state $\ket{\psi_0}$. The Krylov space dimension $D$ determines the accuracy of ground
    state energy estimation, with convergence guaranteed when the initial state has a sufficiently large overlap 
    ($|\gamma_0|=\Omega(1/\text{poly}(n))$ for $n$ qubits) with the true ground state. The Chebyshev approach allows exact
    Krylov space construction (up to sample noise) without real or imaginary time evolution. This algorithm is motivated by the rapid 
    convergence of the Lanczos method for estimating extremal eigenvalues, and its quantum version avoids the classical barrier of 
    exponential cost in representing Krylov vectors.

    The Krylov basis vectors are generated by applying Chebyshev polynomials of the Hamiltonian to $\ket{\psi_0}$:

    .. math::

        \ket{\psi_k}=T_k(H)\ket{\psi_0}, k=0,\dots,D-1,

    which span the same Krylov space as $\{H^k\ket{\psi_0}\}$. The generalized eigenvalue problem $\mathbf{H}\vec{v}=\epsilon \mathbf{S}\vec{v}$
    arises as a standard quantum subspace diagonalization task, with the projected Hamiltonian matrix $\mathbf{H}$ and overlap matrix $\mathbf{S}$ defined 
    in this Chebyshev basis.

    We can obtain $\mathbf{S}$ by evaluating its matrix elements as

    .. math::

        S_{ij}=\bra{\psi_0}T_i(H)T_j(H)\ket{\psi_0}=\langle T_i(H)T_j(H)\rangle_0,

    where $\langle\cdot\rangle_0$ denotes expectation value with respect to the initial state $\ket{\psi_0}$.

    Using the identity $T_i(x)T_j(x)=\frac{1}{2}(T_{i+j}(x)+T_{|i-j|}(x))$, the overlap elements become

    .. math::

        S_{ij}=\frac{1}{2}\langle T_{i+j}(H)+T_{|i-j|}(H)\rangle_0=\frac{1}{2}\bigg(\langle T_{i+j}(H)\rangle_0+\langle T_{|i-j|}(H)\rangle_0\bigg).

    For $\mathbf{H}$, we have $H_{ij}=\langle T_i(H) H T_j(H)\rangle_0$. By using the fact that $H=T_1(H)$ and applying the Chebyshev identity twice, we get

    .. math::

        H_{ij}=\frac{1}{4}\bigg(\langle T_{i+j+1}(H)\rangle_0+\langle T_{|i+j-1|}(H)\rangle_0 + \langle T_{|i-j+1|}(H)\rangle_0 + \langle T_{|i-j-1|}(H)\rangle_0\bigg).

    Because $i,j=0,1,2,\dots, D-1$, all matrix elements $S_{ij}$ and $H_{ij}$ are expressed as linear combinations of expectation values of Chebyshev polynomials
    with respect to the initial state $\langle T_k(H)\rangle_0$, where $k=0,1,2,\dots, 2D-1$. The highest value $2D-1$ comes from the first term
    for $H_{ij}$ when $i=j=D-1$. To construct $\mathbf{H}$ and $\mathbf{S}$ it is enough to estimate all of the expectation values $\langle T_k(H)\rangle_0$.
   
    The final step in this implementation is solving the generalized eigenvalue problem $\mathbf{H}\vec{v}=\epsilon \mathbf{S}\vec{v}$.
    In practice, the overlap matrix $\mathbf{S}$ can become ill-conditioned due to the linear dependencies in the Krylov basis
    or sampling noise in the expectation values $T_k(H)$. To prevent numerical instability, the matrices are regularized via thresholding,
    a process involving discarding small eigenvalues of $\mathbf{S}$ below a specified cutoff. The choice of this cutoff is vital;
    a value too small may fail to suppress noise, while a value too large may discard physically relevant information.

    The entire approach can be summarized by the following steps:
      1. Run quantum Lanczos subroutine :func:`lanczos_expvals` to obtain Chebyshev expectation values $\langle T_k(H)\rangle_0$.
      2. Build overlap and Hamiltonian subspace matrices $(\mathbf{S}, \mathbf{H})$.
      3. Regularize overlap matrix $\mathbf{S}$ and $\mathbf{H}$ by projecting onto the subspace with well conditioned eigenvalues.
      4. Solve generalized eigenvalue problem $\mathbf{H}\vec{v}=\epsilon \mathbf{S}\vec{v}$.
      5. Return lowest eigenvalue $\epsilon_{\text{min}}$ as ground state energy estimate.

    Parameters
    ----------
        H : QubitOperator or BlockEncoding
            Hamiltonian for which to estimate the ground-state energy. If a 
            QubitOperator is provided, it is automatically converted to a 
            Pauli block-encoding.
        D : int
            Krylov space dimension.
        operand_prep : callable 
            Function returning the (operand) QuantumVariables in the initial system state $\ket{\psi_0}$, i.e.,
            ``operands=operand_prep()``. Must return a QuantumVariable or a tuple of QuantumVariables.
        mes_kwargs : dict
            The keyword arguments for the measurement function. 
            By default, 100_000 ``shots`` are executed for measuring each expectation value.
        cutoff : float
            Regularization cutoff threshold for overlap matrix $\mathbf{S}$. The default is 1e-2.
        show_info : bool, optional
            If True, a dictionary with detailed information is returned. The default is False.

    Returns
    -------
    energy : float
        Estimated ground state energy of the Hamiltonian H.
    info : dict, optional
        Full details including:
            - 'energy' : float 
                Ground-state energy estimate
            - 'eigvals' : ArrayLike, shape (D,)
                Eigenvalues of regularized problem
            - 'eigvecs' : ArrayLike, shape (D, D)
                Eigenvectors of regularized problem
            - 'H' : ArrayLike, shape (D, D)
                The Hamiltonian matrix
            - 'H_reg' : ArrayLike, shape (D, D)
                Regularized Hamiltonian matrix
            - 'S' : ArrayLike, shape (D, D)
                The overlap matrix
            - 'S_reg' : ArrayLike, shape (D, D)
                Regularized overlap matrix
            - 'Tk_expvals' : ArrayLike, shape (2D,)
                Chebyshev expectation values
    
    Examples
    --------

    **Example 1: Jasp Mode (Dynamic Execution)**

    This mode uses Qrisp's :ref:`Jasp <jasp>` framework for JIT-compilation and 
    tracing, ideal for high-performance execution.

    ::

        from qrisp import QuantumVariable
        from qrisp.algorithms.lanczos import lanczos_alg
        from qrisp.operators import X, Y, Z
        from qrisp.vqe.problems.heisenberg import create_heisenberg_init_function
        from qrisp.jasp import jaspify
        import networkx as nx

        # Define a 1D Heisenberg model
        L = 6
        G = nx.cycle_graph(L)
        H = (1/4)*sum((X(i)*X(j) + Y(i)*Y(j) + 0.5*Z(i)*Z(j)) for i,j in G.edges())

        # Prepare initial state function (tensor product of singlets)
        M = nx.maximal_matching(G)
        U_singlet = create_heisenberg_init_function(M)

        def operand_prep():
            qv = QuantumVariable(H.find_minimal_qubit_amount())
            U_singlet(qv)
            return qv

        D = 6  # Krylov dimension

        @jaspify(terminal_sampling=True)
        def main():
            return lanczos_alg(H, D, operand_prep, show_info=True)

        energy, info = main()
        print(f"Ground state energy estimate: {energy}")

    We can compare the results obtained by classical calculation:

    ::

        print(f"Ground state energy: {H.ground_state_energy()}")

    **Example 2: Standard Mode (Static Execution)**

    The standard mode is useful for simple scripts and direct hardware 
    interface without JAX overhead.

    ::

        # Using the same Hamiltonian and prep function from the previous example
        energy = lanczos_alg(H, D=6, operand_prep=operand_prep)
        print(f"Ground state energy: {energy}")
        print(f"Ground state energy: {H.ground_state_energy()}")

    """
    
    BE = H if isinstance(H, BlockEncoding) else BlockEncoding.from_operator(H)

    # Step 1: Quantum Lanczos: Find expectation values of Chebyshev polynomials
    Tk_expvals = lanczos_expvals(BE, D, operand_prep, mes_kwargs)

    # Step 2: Build matrices S and H
    S, H_mat = build_S_H_from_Tk(Tk_expvals)

    # Step 3: Regularize matrices via thresholding
    S_reg, H_reg, _ = regularize_S_H(S, H_mat, cutoff=cutoff)  

    # Step 4: Solve generalized eigenvalue problem $\mathbf{H}\vec{v}=\epsilon\mathbf{S}\vec{v}$
    #eigvals, eigvecs = jax.scipy.linalg.eigh(H_reg, S_reg) # Solving the generalized eigenvalue problem not implemented in JAX 0.6
    eigvals, eigvecs = generalized_eigh(H_reg, S_reg)

    # Step 5: Find ground state energy
    # Rescale by block-encoding normalization factor
    ground_state_energy = jnp.min(eigvals) * BE.alpha
    
    if show_info:

        results = {
            'energy': ground_state_energy,
            'eigvals': eigvals,
            'eigvecs': eigvecs,
            'H': H_mat,
            'H_reg': H_reg,
            'S': S,
            'S_reg': S_reg,
            'Tk_expvals': Tk_expvals,
        }
        return ground_state_energy, results
    
    else:

        return ground_state_energy
    