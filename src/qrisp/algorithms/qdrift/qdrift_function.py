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

import numpy as npSi
from qrisp import *


def qdrift(H, qv, time_simulation, N, use_arctan= False ):

    r"""
    This algorithm simulates the time evolution of a quantum state under a Hamiltonian using the **QDRIFT** (Quantum Stochastic Drift Protocol) algorithm.

    QDRIFT approximates the exact time-evolution operator

    .. math::
        U(t) = e^{-i H t}, \qquad 
        H = \sum_j h_j P_j,

    by replacing it with a stochastic product of simpler exponentials

    .. math::
        \tilde{U}(t) = \prod_{k=1}^N e^{-i \, \tau \, \mathrm{sgn}(h_{j_k}) P_{j_k}},

    where each term :math:`P_j` is sampled independently with probability

    .. math::
        p_j = \frac{|h_j|}{\lambda}, \qquad 
        \lambda = \sum_j |h_j|.

    The number of samples :math:`N` controls the overall simulation accuracy. The expected diamond-norm error of the QDRIFT channel satisfies : 

    .. math::
        \epsilon \le \frac{2 \lambda^2 t^2}{N},

    so that achieving a target precision :math:`\epsilon` requires

    .. math::
        N = \mathcal{O}\!\left( \frac{\lambda^2 t^2}{\epsilon} \right).

    Each sampled exponential uses a fixed time-step parameter

    .. math::
        \tau = \frac{\lambda t}{N},

    or, if :code:`use_arctan=True`, a numerically stable variant

    .. math::
        \tau = \arctan\!\left( \frac{\lambda t}{N} \right).

    QDRIFT is particularly suited for large quantum systems whose Hamiltonians can be decomposed into a sum of local Pauli terms. 

    Parameters
    ----------
    H : :class:`QubitOperator`
        Hamiltonian to simulate, written as 
        :math:`H = \sum_j h_j P_j`, where each :math:`P_j`
        is a tensor product of Pauli operators.
    qv : :class:`QuantumVariable`
        Quantum state to which the simulated evolution is applied.
    time_simulation : float
        Total simulation time :math:`t`.
    N : int
        Number of random samples (the number of exponentials in the product).
        Larger values yield higher accuracy at the cost of more gates.
    use_arctan : bool, optional
        If ``True``, uses :math:`\tau = \arctan(\lambda t / N)`
        instead of linear scaling. This can improve numerical stability
        for large :math:`\lambda t`. Default is ``False``.

    Returns
    -------
    qv : :class:`QuantumVariable`
        The evolved quantum state after applying the QDRIFT approximation.
    """
    
    coeffs = []
    terms = []
    # signs = []

    coeffs = []
    terms = []
    for term, coeff in  H.terms_dict.items():
        terms.append(term)
        coeffs.append(coeff)

    # Step 1
    normalisation_factor = sum(abs(c) for c in coeffs)
    if normalisation_factor == 0 or time_simulation == 0:
        return []  

    # Step 3
    tau = normalisation_factor * time_simulation / N
    angle = np.arctan(tau) if use_arctan else tau

    # Step 4
    probs = [abs(c) / normalisation_factor for c in coeffs]

    # Step 5
    i = 0
    def sample(probs):
        return np.random.choice(range(len(probs)), p=probs)

    while i < N:
        i += 1  # Step 5a
        j = sample(probs)  # Step 5b
        terms[j].simulate(angle, qv)  # Step 5c
        
    #Step 6
    return qv

r"""
Example usage of the qdrift function to simulate a quantum system governed by an Ising Hamiltonian on a chain graph.


Below is an example usage of the :func:`qdrift` function to simulate a quantum system governed
by an Ising Hamiltonian on a one-dimensional chain graph.

In this example, we build a chain graph, define an Ising Hamiltonian with given coupling and
magnetic field strengths, and compute the magnetization of the system over a range of evolution
times using the QDRIFT algorithm. The results are compared to a first-order Trotterization
simulation to visualize their qualitative agreement.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    from qrisp import QuantumVariable
    from qrisp.operators import X, Z, QubitOperator

    # Helper functions
    def generate_chain_graph(N):
        G = nx.Graph()
        G.add_edges_from((k, k+1) for k in range(N-1))
        return G

    def create_ising_hamiltonian(G, J, B):
        # H = -J ∑ Z_i Z_{i+1} - B ∑ X_i
        H = sum(-J * Z(i)*Z(j) for (i,j) in G.edges()) + sum(B * X(i) for i in G.nodes())
        return H

    def create_magnetization(G):
        return (1.0 / G.number_of_nodes()) * sum(Z(i) for i in G.nodes())

    # Simulation setup
    G = generate_chain_graph(6)
    H = create_ising_hamiltonian(G, J=1.0, B=1.0)
    M = create_magnetization(G)
    T_values = np.arange(0, 2, 0.05)
    precision_expectation_value = 0.001

    # Choose N according to the theoretical scaling 
    # The QDRIFT bound suggests:  N ≈ ceil(2 λ² t² / ε), where λ = ∑|h_j|. Here we use this expression directly, although for many models it leads to very large circuits.
    #
    # The user is free to choose any alternative formula for N (e.g. N ∝ (λ t)²), depending on desired accuracy and runtime.

    lam = sum(abs(coeff) for coeff in H.terms_dict.values())
    epsilon = 0.001
    N = int(np.ceil(2 * lam**2 * (max(T_values)**2) / epsilon))

    def psi(t, use_arctan=True):
        qv = QuantumVariable(G.number_of_nodes())
        qdrift(H, qv, time_simulation=t, N=N, use_arctan=use_arctan)
        return qv

    # Compute magnetization expectation 
    M_values = []
    for t in T_values:
        ev_M = M.expectation_value(psi, precision_expectation_value)
        M_values.append(float(ev_M(t)))



.. image:: imNth.png
   :alt: QDRIFT Ising magnetization simulation
   :align: center
   :width: 600px

Discussion
----------
For the sake of demonstration, this example uses the **theoretical bound**

.. math::
    N = \left\lceil \frac{2 \lambda^2 t^2}{\epsilon} \right\rceil,

where :math:`\lambda = \sum_j |h_j|` and :math:`\epsilon` is the target diamond-norm precision.

While this choice guarantees the formal error bound from Random Compiler for Fast Hamiltonian Simulation, Physical Review Letters 123, 070503 (2019), it can produce
extremely large circuit depths (here on the order of hundreds of thousands of Pauli rotations),
making the simulation slow for dense Hamiltonians such as the Ising model.

QDRIFT is **not optimal** for Hamiltonians with many large coefficients (like the Ising chain),
because the total weight :math:`\lambda` is high, leading to large :math:`N`.
However, it is **highly efficient for sparse or weakly weighted Hamiltonians**
where :math:`\lambda` is small—common in chemistry or local lattice models—
making it a powerful tool for large-scale quantum simulations with bounded resources.

    """
