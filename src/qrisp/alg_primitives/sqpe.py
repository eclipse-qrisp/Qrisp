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

import math
import numpy as np
import random
from scipy.special import iv
from qrisp.operators import QubitOperator
from qrisp import QuantumVariable, QuantumBool, h, control, rz

def SQPE(H, ansatz_state,
        precision_energy_ground_state, tau, energy_threshold,
        energy_resolution_cdf, overlap_H_ansatz, total_error, failure_probability = 0.05, n_max=20, shots_override=None):
        
    r"""
        Implements a **Stochastic Quantum Phase Estimation (SQPE)**–type decision
        algorithm based on the randomized statistical phase estimation method of

        X. Wan, M. Berta, and E. T. Campbell, 
        *"A randomized quantum algorithm for statistical phase estimation."*

        The goal of this routine is **not** to output a precise estimate of the
        ground state energy, but to solve an **eigenvalue-threshold decision problem**
        of the form:

        .. math::

            \text{Decide whether } E_0 \le X \quad \text{or} \quad E_0 > X,

        where :math:`E_0` is the ground state energy of a Hamiltonian

        .. math::

            H = \sum_{\ell} h_{\ell} P_{\ell},

        expressed as a weighted sum of Pauli strings :math:`P_{\ell}`, and :math:`X`
        is a user-chosen energy threshold.

        The algorithm assumes access to a state :math:`\rho` (here encoded via
        the quantum variable ``ansatz_state``) that approximates the ground state
        with **overlap** :math:`\eta`:

        .. math::

            \eta = \langle \psi_0 | \rho | \psi_0 \rangle,

        where :math:`|\psi_0\rangle` is the ground state of :math:`H`.

        The method estimates the **cumulative distribution function (CDF)**

        .. math::

            C(x) = \mathrm{tr}\!\left[\rho \, \Pi_{(-\infty, x/\tau]}(H)\right],
            \qquad x \in \mathbb{R},

        where :math:`\Pi_{(-\infty, x/\tau]}(H)` is the projector onto the
        eigenspaces of :math:`H` with eigenvalues :math:`E \le x/\tau`, and
        :math:`\tau > 0` is a fixed time parameter.

        In particular, for a chosen threshold :math:`X`, we evaluate :math:`C(\tau X)`,
        which equals the probability that a measurement of the energy on :math:`\rho`
        yields a value :math:`E \le X`:

        .. math::

            C(\tau X) = \Pr(E \le X).

        The algorithm constructs a **Fourier series approximation** of :math:`C(x)`
        using a finite set of time points :math:`\{t_j\}`, and then estimates the
        corresponding Fourier coefficients via **randomized time evolution** and a
        **Hadamard test**. Specifically, it approximates integrals of the form

        .. math::

            C(x) \approx \sum_{j} F_j \, \mathrm{tr}\!\left(\rho \, e^{-i t_j H}\right),

        where the complex weights :math:`F_j` and times :math:`t_j` are chosen
        according to the construction in the paper, using modified Bessel functions
        :math:`I_j(\beta)`.

        Since :math:`e^{-i t_j H}` cannot be implemented directly, the algorithm
        uses a **randomized product formula**, analogous in spirit to QDRIFT:
        it samples Pauli terms :math:`P_{\ell}` according to a distribution
        proportional to :math:`|h_{\ell}|` and builds products of exponentials

        .. math::

            e^{-i t_j H} \approx \prod_k e^{-i \theta_{k} P_{\ell_k}},

        with angles :math:`\theta_{k}` drawn from a **truncated Taylor-like**
        distribution. The expectation value of the resulting random unitary
        approximates the exact time evolution.

        The overlap between the ansatz and the ground state enters through a
        **decision threshold** :math:`\eta/2`. If the estimated CDF value
        :math:`\widehat{C}(\tau X)` lies significantly above :math:`\eta/2`, the
        algorithm concludes that the ground state energy is **below or equal** to
        :math:`X`; if it lies significantly below :math:`\eta/2`, it concludes the
        ground state energy is **strictly above** :math:`X`. The total error
        (statistical + Fourier approximation) is controlled by the parameters
        ``total_error`` and ``failure_probability``.

        Parameters : 
        ----------
        H : :class:`QubitOperator`
                Hamiltonian to simulate, written as 
                :math:`H = \sum_j h_j P_j`, where each :math:`P_j`
                is a tensor product of Pauli operators.
        ansatz_state : QuantumVariable
            Quantum state approximating the ground state of H.
        precision_energy_ground_state : float
            Desired precision Δ of the ground state energy estimate.
        tau : float
            Time parameter for the phase estimation.
        energy_threshold : float
            Energy threshold X for the decision problem.
        energy_resolution_cdf : float
            Energy resolution δ for the cumulative distribution function.
        overlap_H_ansatz : float
            Overlap η between the Hamiltonian and the ansatz state.
        total_error : float
            Total allowable error ε in the estimation.
        failure_probability : float, optional
            Acceptable probability of failure v. The default is 0.05.
        n_max : int, optional
            Maximum Taylor series order. The default is 20.
        shots_override : int, optional
            Override for the number of shots in the Hadamard test. The default is None
        
        Returns :
        -------
        prob_estimate : float
            The estimated probability

            .. math::
                \widehat{C}(\tau X) \approx \Pr(E \le X),

            i.e. the approximate value of the CDF at the chosen threshold.

        threshold : float
            The decision threshold

            .. math::
                \frac{\eta}{2} = \frac{\texttt{overlap\_H\_ansatz}}{2},

            against which the estimated probability is compared.

        is_below_threshold : bool
            Boolean flag indicating the conclusion of the decision procedure:

            * ``True`` if :math:`\widehat{C}(\tau X) \ge \eta/2`, i.e. the ground
            state energy is **estimated to be less than or equal to** :math:`X`.
            * ``False`` otherwise, i.e. the ground state energy is **estimated to be
            strictly greater than** :math:`X`.

        conclusion : str
            Sentence summarizing the result, of the form

            .. code-block:: text

                The ground state energy is estimated to be
                (less than or equal to / strictly greater than)
                the chosen energy threshold X = ...
                with estimated probability Pr(E ≤ X) ≈ ...,
                compared to the decision threshold η/2 = ...,
                and an overall failure probability at most v = ... 

        Her Below is a minimal example using a single-qubit Hamiltonian
    :math:`H = Z_0`, together with a simple ansatz state prepared by
    flipping qubit 0 to :math:`|1\rangle`:

    .. code-block:: python

        import numpy as np
        from qrisp import QuantumVariable
        from qrisp.operators import X, Z

        # Define a simple 1-qubit Hamiltonian
        H_1q = Z(0)

        # Prepare an ansatz state |100⟩ on 3 qubits, with qubit 0 in |1⟩
        def prepare_ansatz():
            qv = QuantumVariable(3)   # starts in |000⟩
            X(qv[0])                 # prepare |100⟩
            return qv

        ansatz_state_1q = prepare_ansatz()

        # Parameters
        precision_energy_ground_state = 0.5    # Δ
        tau = 1.0                              # τ
        energy_threshold = 0.0                 # X
        energy_resolution_cdf = 0.2            # δ (≤ τΔ)
        overlap_H_ansatz = 0.9                 # η
        total_error = 0.2                      # ε (< η/2)
        failure_probability = 0.1              # v
        n_max = 10

        # Run the SQPE decision algorithm
        prob_estimate, threshold, is_below_threshold, conclusion = SQPE(
            H=H_1q,
            ansatz_state=ansatz_state_1q,
            precision_energy_ground_state=precision_energy_ground_state,
            tau=tau,
            energy_threshold=energy_threshold,
            energy_resolution_cdf=energy_resolution_cdf,
            overlap_H_ansatz=overlap_H_ansatz,
            total_error=total_error,
            failure_probability=failure_probability,
            n_max=n_max
        )

        print("Pr(E ≤ X) ≈", prob_estimate)
        print("Decision threshold η/2 =", threshold)
        print("Is estimated energy ≤ X? :", is_below_threshold)
        print(conclusion)

    This example solves the decision problem for the simple Hamiltonian
    :math:`H = Z_0` and the given ansatz, estimating whether the ground state
    energy is less than or equal to the chosen threshold :math:`X`.
        """
    # Step 0: Hamiltonian preprocessing and parameter checks

    H_coeffs = []
    H_terms = []

    for term, coeff in  H.terms_dict.items():
        H_terms.append(term)
        H_coeffs.append(coeff)

    normalisation_factor = sum(abs(h) for h in H_coeffs)
    if normalisation_factor == 0:
        raise ValueError("Hamiltonian has zero norm.")
    
    if precision_energy_ground_state <= 0:
        raise ValueError("The precision of the ground state must be positive.")
    
    if energy_resolution_cdf <= 0 or energy_resolution_cdf > tau * precision_energy_ground_state:
        raise ValueError("The energy resolution of the Cumulative Distribution Function value must be in (0, tau * precision_energy_ground_state].")
    
    if overlap_H_ansatz <= 0 or overlap_H_ansatz > 1:
        raise ValueError("The overlap between the hamiltonian and the ansatz state must be in (0, 1].")
    
    if total_error <= 0 or total_error > overlap_H_ansatz / 2:
        raise ValueError("The total error must be in (0, overlap_H_ansatz / 2].")
    
    if failure_probability <= 0 or failure_probability >= 1:
        raise ValueError("The probability of failure must be in the range (0, 1).")

    # normalize H: sum |h_l_hat| = 1
    H_coeffs_normalised = (np.abs(H_coeffs) / normalisation_factor).tolist()

    # Helper functions

    def compute_fourier_coeffs(energy_resolution_cdf, error_fourier, normalisation_factor, tau):

        """
        Compute Fourier coefficients F_j and corresponding times t_j.
        """
    
        beta = max(1 / (4 * np.sin(energy_resolution_cdf)**2) * np.log(3 / (np.pi * error_fourier**2)), 1)
        d = int(np.ceil(2 * beta))
        S0 = range(d + 1)

        F = {}

        for j in S0:
            if j == 0:
                F[0] = 0.5
            elif j == d:
                # kept exactly as in your code
                F[d] = -1j * np.sqrt(beta / (2 * np.pi)) * np.exp(-beta) * iv(d, beta) / (2*d + 1)
            else:
                coeff = -1j * np.sqrt(beta / (2 * np.pi)) * np.exp(-beta)
                bessel_sum = (iv(j, beta) + iv(j + 1, beta)) / (2 * j + 1)
                F[j] = coeff * bessel_sum if j > 0 else np.conj(coeff * bessel_sum)

        # times: t_j = - j * tau * normalisation_factor
        t = {j: -j * tau * normalisation_factor for j in S0}
        return F, S0, t

    def sample_unitary(H_coeffs_normalised, H_terms, time_evolution, decomposition_depth, n_max ):

        """
        Sample a product of unitaries according to the truncated Taylor series scheme.

        Returns
        -------
        A0 : list[float]
            Rotation angles θ.
        A1 : list[PauliTerm-like]
            First Pauli term P0 in each segment.
        B : list[QubitOperator]
            Product of Pauli unitaries B_i for each segment.
        """

        indexes_sampled = []

        t = float(time_evolution)
        r = int(decomposition_depth)

        if r <= 0:
            raise ValueError("Decomposition_depth must be a positive integer.")
        
        tau = abs(t) / r

        probs_n =[]
        A0 = []  # angles θ
        A1 = []  # first Pauli term P0 in each product
        B = []   # product of Pauli matrices for each segment
        j = 0

        while j < r:

            n_list = list(range(0, 2*(n_max + 1), 2))
            probs_n = [ (tau**i / math.factorial(i)) * math.sqrt(1.0 + (tau / (i + 1))**2) for i in n_list ]
            probs_n = np.asarray(probs_n, dtype=float); probs_n /= probs_n.sum()

            n = n_list[int(np.random.choice(len(probs_n), p=probs_n))]
            
            p_arr = np.asarray(H_coeffs_normalised, dtype=float); p_arr /= p_arr.sum()
            indexes_sampled = list(np.random.choice(len(p_arr), size=n+1, p=p_arr))

            theta = math.acos(1.0 / math.sqrt(1.0 + (tau / (n + 1))**2))

            I = QubitOperator(())                              # identity
            P0 = H_terms[indexes_sampled[0]]
            A0.append(theta)
            A1.append(P0)

            U=QubitOperator(())   # start from I

            for i in range(1, n + 1):
                Ui = QubitOperator({ H_terms[indexes_sampled[i]]: 1.0 })
                U = Ui * U
            U = (1j * np.sign(time_evolution)) ** n * I * U
            B.append(U)

            j += 1

        return A0, A1, B
    
    def hadamard_test(U, psi, shots):
    
        """
        Hadamard test to estimate Re⟨ψ|U|ψ⟩ and Im⟨ψ|U|ψ⟩.
        """

        # Ancilla qubit
        anc = QuantumBool()          # single qubit, starts in |0>

        # Real part 
        h(anc)

        with control(anc):
            U(psi)

        # 3) Hadamard again on ancilla
        h(anc)

        # 4) Measure ancilla
        counts_re = anc.get_measurement(shots=shots)  
        p1_re = counts_re.get(True, 0.0)

        # Obtention of the trace
        X_trace = 2 * p1_re - 1

        # Imaginary part 
        anc = QuantumBool()   # fresh ancilla

        h(anc)

        with control(anc):
            U(psi)

        rz(-np.pi/2, anc)

        h(anc)

        counts_im = anc.get_measurement(shots=shots)
        p1_im = counts_im.get(True, 0.0)
        Y_trace = 2 * p1_im - 1

        return X_trace, Y_trace
    
    # Step 1–4: Fourier coefficients, sampling parameters

    error_fourier = total_error / 4
    Fj, S1, tj = compute_fourier_coeffs(
        energy_resolution_cdf, error_fourier, normalisation_factor, tau
    )

    error_stat = overlap_H_ansatz / 4


    # Non-zero times (j = 0 has t_j = 0 and is treated separately via F_j[0])    
    S10 = [j for j in S1 if tj[j] != 0]
    
    rj = {}
    mu = {}

    for j in S1:
        tj_abs2 = float(abs(tj[j])**2)

        rj_j = int(np.ceil(2 * tj_abs2))

        if rj_j < 1:
            rj_j = 1

        rj[j] = rj_j

        mu[j] = math.exp(tj_abs2 / rj_j) 
            

    total_weight = sum(abs(Fj[j]) * mu[j] for j in S10)  # upper bound

    C_sample = int(np.ceil(((2 * total_weight) / (overlap_H_ansatz / 2 - total_error))**2 * np.log(1 / failure_probability)))

    shots = int(np.ceil(((total_weight ** 2) / (error_stat ** 2)) * np.log(1 / failure_probability)))

    if shots_override is not None:       
        shots = int(shots_override)

    # Step 5–7: Sampling of the Cumulative Distribution Function via unitaries

    pj = np.array([abs(Fj[j]) for j in S10], dtype=float)
    pj /= pj.sum()
    S1_list = list(S10)

    cdf_samples = []
    x = tau* energy_threshold

    for _ in range(C_sample):
        # Step 5a : sample j according to |F_j| / sum |F_j|
        j = random.choices(S1_list, weights=pj, k=1)[0]

        # Step 5b : sample unitary corresponding to time t_j[j]
        A0, A1, B = sample_unitary(H_coeffs_normalised, H_terms, tj[j], rj[j], n_max)


        def U_k(qv) : 
            for i in range(rj[j]):
                theta = A0[i]
                P = A1[i]
                P.simulate(theta, qv)

                Bi = B[i]

                unitaries, coeffs = Bi.unitaries()
                unitaries[0](qv)
                #print("[debug] Applied unitary:", unitaries[0], "with coeff:", coeffs[0])

        # Step 6 : Hadamard test -> complex estimate of ⟨ψ|U_k|ψ⟩
        X_trace, Y_trace = hadamard_test(U_k, ansatz_state, shots)
        mi = X_trace + 1j * Y_trace

        # Step 7 : contribution to the CDF estimator
        z_i = total_weight * np.exp(1j * (np.angle(Fj[j]) + j * x)) * mi
        cdf_samples.append(z_i)

    # Step 8 : Final Cumulative distribution Function Estimate and Conclusion
    z_bar = sum(cdf_samples) / C_sample + Fj[0]

    # Estimated probability and threshold in the CDF picture
    prob_estimate = float(np.real(z_bar))        # ≈ Pr(E ≤ X)
    threshold = overlap_H_ansatz / 2.0           # decision threshold η/2
    is_below_threshold = prob_estimate >= threshold

    #Conclusion
    conclusion = (
    f"The ground state energy is estimated to be "
    f"{'less than or equal to' if is_below_threshold else 'strictly greater than'} "
    f"the chosen energy threshold X = {energy_threshold:.6f}, "
    f"with estimated probability Pr(E ≤ X) ≈ {prob_estimate:.4f}, "
    f"compared to the decision threshold η/2 = {threshold:.4f}, "
    f"and an overall failure probability at most v = {failure_probability:.3f}."
    )

    print(conclusion)
    print(f"Estimated probability Pr(E ≤ X) = {prob_estimate:.4f}")
    print(f"Decision threshold = {threshold:.4f}")
    print(f"Is estimated energy ≤ X? : {is_below_threshold}")
    
