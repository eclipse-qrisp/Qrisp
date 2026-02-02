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
from qrisp import QuantumBool
from qrisp.algorithms.gqsp.gqsp import GQSP
from qrisp.algorithms.gqsp.gqsp_angles import gqsp_angles
from qrisp.jasp import qache
from qrisp.block_encodings import BlockEncoding
from qrisp.operators import QubitOperator
from scipy.special import jv
import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax.typing import ArrayLike


# https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368
def hamiltonian_simulation(H: QubitOperator | BlockEncoding, t: "ArrayLike" = 1, N: int = 1) -> BlockEncoding:
    r"""
    Performs Hamiltonian simulation using Generalised Quantum Signal Processing.

    This function approximates the unitary evolution operator $U = e^{-itH}$
    by expanding the time-evolution function into a series of Bessel functions 
    via the Jacobi-Anger identity

    .. math ::

        e^{-it\cos(\theta)} \approx \sum_{n=-N}^{N}(-i)^nJ_n(t)e^{in\theta}

    where $J_n(t)$ are Bessel functions of the first kind.

    Parameters
    ----------
    H : BlockEncoding | QubitOperator
        The Hermitian operator to be simulated.
    t : ArrayLike
        The scalar evolution time $t$. The default is 1.0.
    N : int
        The truncation order of the expansion. The default is 1.

    Returns
    -------
    BlockEncoding
        A block encoding approximating the unitary $e^{-itH}$.

    Notes
    -----
    - **Precision**: The truncation error scales (decreases) super-exponentially with $N$. 
      For a fixed $t$, choosing $N > |t|$ ensures rapid convergence.
    - **Normalization**: The resulting operator is nearly unitary, meaning its 
      block-encoding normalization factor $\alpha$ will be close to 1.

    References
    ----------
    - Motlagh & Wiebe (2024) `Generalized Quantum Signal Processing <https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368>`_.

    Examples
    --------

    Below is an example of using the :func:`hamiltonian_simulation` function to simulate a quantum system governed by an Ising Hamiltonian on a 1D chain graph. 
    In this example, we construct a chain graph, define an Ising Hamiltonian with specific coupling and magnetic field strengths, 
    and compute the system's energy and magnetization over various evolution times using the QSP-based simulation algorithm. 
    Finally, the results are compared against a classical simulation.

    ::

        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np
        from qrisp import *
        from qrisp.gqsp import hamiltonian_simulation
        from qrisp.operators import X, Y, Z
        import scipy as sp

        def generate_chain_graph(N):
            coupling_list = [[k,k+1] for k in range(N-1)]
            G = nx.Graph()
            G.add_edges_from(coupling_list)
            return G

        def create_ising_hamiltonian(G, J, B):
            H = sum(-J * Z(i) * Z(j) for (i,j) in G.edges()) + sum(B * X(i) for i in G.nodes())
            return H

        def create_magnetization(G):
            H = (1 / G.number_of_nodes()) * sum(Z(i) for i in G.nodes())
            return H

        # Evaluate observables classically
        def sim_classical(T_values, H, M):
            M_values = []
            E_values = []
            H_mat = H.to_array()
            M_mat = M.to_array()

            def _psi(t):
                psi0 = np.zeros(2**H.find_minimal_qubit_amount())
                psi0[0] = 1
                psi = sp.linalg.expm(-1.0j * t * H_mat) @ psi0
                psi = psi / np.linalg.norm(psi)
                return psi

            for t in T_values:
                psi = _psi(t)
                magnetization = (np.conj(psi) @ M_mat @ psi).real
                energy = (np.conj(psi) @ H_mat @ psi).real
                M_values.append(magnetization)
                E_values.append(energy)
                
            return np.array(M_values), np.array(E_values)

            
        # Evaluate observables using QSP-based simulation
        def sim_qsp(T_values, H, M):
            M_values = []
            E_values = []

            def operand_prep():
                return QuantumFloat(H.find_minimal_qubit_amount())

            def psi(t):
                BE = hamiltonian_simulation(H, t=t, N=10)
                operand = BE.apply_rus(operand_prep)()
                return operand

            @jaspify(terminal_sampling=True)
            def magnetization(t):
                magnetization = M.expectation_value(psi, precision=0.005)(t)
                return magnetization

            @jaspify(terminal_sampling=True)
            def energy(t):
                energy = H.expectation_value(psi, precision=0.005)(t)
                return energy

            for t in T_values:
                M_values.append(magnetization(t))
                E_values.append(energy(t))
            
            return np.array(M_values), np.array(E_values)

        
        G = generate_chain_graph(6)
        H = create_ising_hamiltonian(G, 0.25, 0.5)
        M = create_magnetization(G)

        T_values = np.arange(0.1, 3.0, 0.1)
        M_classical, E_classical = sim_classical(T_values, H, M)
        M_qsp, E_qsp = sim_qsp(T_values, H, M)

        # Plot the results
        plt.scatter(T_values, E_classical, color='#20306f', marker="d", label="E target")
        plt.scatter(T_values, M_classical, color='#6929C4', marker="d", label="M target")
        plt.plot(T_values, E_qsp, color='#20306f', marker="o", linestyle="solid", alpha=0.5, label="E qsp")
        plt.plot(T_values, M_qsp, color='#6929C4', marker="o", linestyle="solid", alpha=0.5, label="M qsp")
        plt.xlabel("Evolution time T", fontsize=15, color="#444444")
        plt.ylabel("Energy and Magnetization", fontsize=15, color="#444444")
        plt.legend(fontsize=15, labelcolor='linecolor')
        plt.tick_params(axis='both', labelsize=12)
        plt.grid()
        plt.show()

    .. image:: /_static/qsp_simulation.png
        :alt: QSP Hamiltonian simulation
        :align: center
        :width: 600px

    The plots illustrate the time evolution of energy $E$ and magnetization $M$ for an Ising model, 
    comparing classical simulation results with those from a quantum simulation employing the Quantum Signal Processing (QSP) based Hamiltonian simulation algorithm.

    Analysis of Results

    - Agreement Phase $T\le 2.0$:  There is excellent agreement between the classical and QSP simulation results for both energy and magnetization during the initial evolution phase, up to an evolution time of $T=2.0$. The $M_{\text{qsp}}$ curve closely follows $M_{\text{classical}}$, while the energy values $E_{\text{classical}}$ and $E_{\text{qsp}}$ remain constant as expected for an isolated system.
    
    - Divergence Phase $T>2.0$: Beyond $T=2.0$, the quantum simulation results diverge noticeably from the classical benchmark. Both $M_{\text{qsp}}$ and $E_{\text{qsp}}$ drift away from the classical trajectories, with the energy showing a significant downward trend.
    
    - Mitigation: This divergence is attributed to an insufficient truncation order $N$ used in the QSP polynomial expansion. The simulation error accumulates over time when the truncation order is too low for the required evolution time $T$. Increasing the truncation order $N$ can mitigate this effect and maintain accuracy at larger $T$ values, but this comes at the expense of a higher computational runtime or circuit depth.

    """

    if isinstance(H, QubitOperator):
        H = H.pauli_block_encoding()

    # Rescaling the time to account for scaling factor alpha of pauli block-encoding
    alpha = H.alpha
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

    BE_walk = H.qubitization()

    angles, new_alpha = gqsp_angles(coeffs)

    def new_unitary(*args):
        GQSP(args[0], *args[1:], unitary = BE_walk.unitary, angles=angles, k=N)

    new_anc_templates = [QuantumBool().template()] + BE_walk.anc_templates
    return BlockEncoding(new_alpha, new_anc_templates, new_unitary, is_hermitian=False)


# Apply the wache decorator with the workaround in order to show in documentation
#temp_docstring = hamiltonian_simulation.__doc__
#hamiltonian_simulation = qache(static_argnames=["H", "N"])(hamiltonian_simulation)
#hamiltonian_simulation.__doc__ = temp_docstring


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