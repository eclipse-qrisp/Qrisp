# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Defines the OracleDCQOProblem class for counterdiabatic driving of diagonal costs given as phase oracles."""

import numpy as np
import sympy as sp

from qrisp import h, rx, s, s_dg, x, z


def _default_lam_func():
    """Default scheduling function, the same as in :func:`create_COLD_instance`."""
    t, T = sp.symbols("t T", real=True)
    return sp.sin(sp.pi / 2 * sp.sin(sp.pi * t / (2 * T)) ** 2) ** 2


class OracleDCQOProblem:
    r"""Counterdiabatic driving (LCD) for a diagonal cost function given as a phase oracle.

    Minimizes an arbitrary diagonal cost $C(b)$ over bitstrings $b \in \{0,1\}^n$. The cost
    does not need to be a QUBO or any other polynomial, and no Pauli decomposition of $C$ is
    ever built. It enters only through

    * ``phase_oracle(qarg, gamma)``, applying $e^{-i\gamma C}$ to ``qarg``, and
    * ``cost_fn(B)``, evaluating $C$ classically for a batch of bitstrings.

    The system is evolved along

    .. math::
        H(\lambda) = (1-\lambda)\, s \sum_i X_i + \lambda C, \qquad s = \text{mixer\_strength},

    starting in the ground state $\ket{-}^{\otimes n}$ of the mixer. The adiabatic gauge
    potential (AGP) uses the first-order nested-commutator ansatz

    .. math::
        A_\lambda = \alpha(\lambda)\, i[H, \partial_\lambda H] = \alpha(\lambda)\, s\, i[H_X, C]
                  = -\alpha(\lambda)\, s \sum_i Y_i\, g_i,
        \qquad g_i(b) = C(b|_{b_i=1}) - C(b|_{b_i=0}).

    $g_i$ is the local energy difference of bit $i$ and does not depend on $b_i$ itself. Each
    factor of the Trotterized AGP is built from two oracle calls,

    .. math::
        e^{i\theta Y_i g_i} = V_i\, X_i U_C(-\theta) X_i\, U_C(\theta)\, V_i^\dagger,
        \qquad V = S H, \quad U_C(\gamma) = e^{-i\gamma C},

    because $V Z V^\dagger = Y$ and $Z_i g_i = C(b \oplus e_i) - C(b)$.

    $\alpha(\lambda)$ minimizes the action $S = \mathrm{Tr}[G^2]$, $G = \partial_\lambda H + i[A_\lambda, H]$,
    which gives $\alpha = \mathrm{Tr}[O_1^2] / \mathrm{Tr}[O_2^2]$ with $O_1 = [H, \partial_\lambda H]$ and
    $O_2 = [H, O_1]$. Both traces reduce to uniform averages of finite differences of $C$,
    with $\Delta_i(b) = C(b \oplus e_i) - C(b)$ and $D_{ij}(b) = \Delta_i(b \oplus e_j) - \Delta_i(b)$:

    .. math::
        \alpha(\lambda) = -\frac{\sum_i E[\Delta_i^2]}
            {4 s^2 (1-\lambda)^2 \left(E[(\sum_i \Delta_i)^2] + \sum_{i<j} E[D_{ij}^2]\right)
             + \lambda^2 \sum_i E[\Delta_i^4]}.

    With ``uniform=False`` every qubit gets its own coefficient, $A_\lambda = \sum_i \alpha_i(\lambda)\, s\, i[X_i, C]$,
    which uses the same circuit with a separate angle per qubit. Minimizing the same action gives
    the linear system $T(\lambda)\, \alpha = -d$ with $d_k = E[\Delta_k^2]$ and

    .. math::
        T_{kl}(\lambda) = s^2 (1-\lambda)^2 \left(4 E[\Delta_k \Delta_l] + Q_{kl}\right)
                         + \delta_{kl}\, \lambda^2 E[\Delta_k^4],
        \qquad Q_{kl} = E[D_{kl}^2]\ (k \neq l), \quad Q_{kk} = \sum_{j \neq k} E[D_{kj}^2].

    The averages are computed once, by exact enumeration of all $2^n$ bitstrings (default) or
    by Monte Carlo sampling (``n_samples``).

    Parameters
    ----------
    cost_fn : callable
        Vectorized classical cost. Receives an integer array ``B`` of shape ``(m, n)``, where
        ``B[:, i]`` is the bit of qubit ``i``, and returns an array of ``m`` floats. This is
        the same function the oracle implements, including all penalty terms.
    phase_oracle : callable
        ``phase_oracle(qarg, gamma)`` applies $e^{-i\gamma C}$ to the :ref:`QuantumVariable`
        ``qarg``. Must accept negative ``gamma``. Ancillas must be uncomputed.
    num_qubits : int
        Number of problem qubits $n$.
    lam_func : callable, optional
        Scheduling function $\lambda(t, T)$, returning a sympy expression in the symbols
        ``t`` and ``T`` as in :class:`DCQOProblem`. Defaults to the schedule of
        :func:`create_COLD_instance`.
    mixer_strength : float, optional
        Prefactor $s$ of the transverse-field mixer. Sets the energy scale of the mixer
        relative to $C$. The default is 1.
    uniform : bool, optional
        If ``True``, one AGP coefficient $\alpha(\lambda)$ for all qubits. If ``False``, one
        coefficient $\alpha_i(\lambda)$ per qubit. The circuit is the same in both cases.
        The default is ``True``.
    local_phase_oracle : callable, optional
        ``local_phase_oracle(qarg, gamma, i)`` applies $e^{-i\gamma C_i}$, where $C_i$ keeps
        only the terms of $C$ that depend on bit ``i``. Used inside the AGP instead of the full
        oracle, which gives the same unitary because $g_i$ only sees these terms.
    feasible_fn : callable, optional
        Vectorized feasibility check with the same input as ``cost_fn``, returning a boolean
        array. Only used by :meth:`evaluate`.
    n_samples : int, optional
        If given, the AGP averages are estimated from this many uniformly random bitstrings
        instead of enumerating all $2^n$. Use this when $2^n$ is too large.
    seed : int, optional
        Seed for the Monte Carlo sampling.

    Examples
    --------
    A small cost with a ReLU penalty, applied through a stand-in oracle. In practice
    ``phase_oracle`` is the arithmetic oracle of the problem.

    ::

        import numpy as np
        from qrisp import QuantumVariable, as_hamiltonian
        from qrisp.algorithms.cold.oracle_dcqo_problem import OracleDCQOProblem

        n = 4
        v = np.array([1.0, 0.8, 0.6, 0.5])
        a = np.array([2, 1, 2, 1])

        def cost_fn(B):
            return -B @ v + 2 * np.maximum(0, B @ a - 3)

        @as_hamiltonian
        def C(label):
            return cost_fn(np.array([[int(c) for c in label]]))[0]

        def phase_oracle(qarg, gamma):
            C(qarg, t=-gamma)

        problem = OracleDCQOProblem(cost_fn, phase_oracle, n)
        res = problem.run(QuantumVariable(n), N_steps=10, T=1.0)
        print(problem.evaluate(res))

    """

    def __init__(  # noqa: PLR0913 -- public, keyword-callable API shape
        self,
        cost_fn,
        phase_oracle,
        num_qubits,
        lam_func=None,
        mixer_strength=1.0,
        uniform=True,
        local_phase_oracle=None,
        feasible_fn=None,
        n_samples=None,
        seed=None,
    ):
        """Construct an OracleDCQOProblem instance. See class docstring for parameter details."""
        self.cost_fn = cost_fn
        self.phase_oracle = phase_oracle
        self.num_qubits = num_qubits
        self.lam_func = lam_func if lam_func is not None else _default_lam_func
        self.mixer_strength = mixer_strength
        self.uniform = uniform
        self.local_phase_oracle = local_phase_oracle
        self.feasible_fn = feasible_fn
        self.n_samples = n_samples
        self.seed = seed

        self.lam = None
        self.lamdot = None
        self._moments = None

    # ----- classical part -----

    def _all_bitstrings(self):
        """All 2^n bitstrings as rows, with bit i of the row index on column i."""
        idx = np.arange(1 << self.num_qubits)
        return ((idx[:, None] >> np.arange(self.num_qubits)) & 1).astype(np.int8)

    def nc_moments(self):
        r"""Uniform averages of finite differences of $C$ that determine $\alpha(\lambda)$.

        Computed on first use and cached.

        Returns
        -------
        dict
            ``d`` with $d_k = E[\Delta_k^2]$, ``P`` with $P_{kl} = E[\Delta_k \Delta_l]$,
            ``Dij`` with $E[D_{kl}^2]$ for $k \neq l$ (zero diagonal) and ``q4`` with
            $E[\Delta_k^4]$.

        """
        if self._moments is not None:
            return self._moments

        n = self.num_qubits
        if self.n_samples is None:
            # Exact enumeration: a bit flip is an XOR on the row index, so C is evaluated only once.
            B = self._all_bitstrings()
            idx = np.arange(1 << n)
            C = np.asarray(self.cost_fn(B), dtype=float)

            def flipped(*bits):
                mask = sum(1 << i for i in bits)
                return C[idx ^ mask]

        else:
            rng = np.random.default_rng(self.seed)
            B = rng.integers(0, 2, size=(self.n_samples, n), dtype=np.int8)
            C = np.asarray(self.cost_fn(B), dtype=float)

            def flipped(*bits):
                Bf = B.copy()
                Bf[:, list(bits)] ^= 1
                return np.asarray(self.cost_fn(Bf), dtype=float)

        C_flip = [flipped(i) for i in range(n)]
        Delta = np.array([Cf - C for Cf in C_flip])
        m = Delta.shape[1]

        Dij = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                # D_ij = Delta_i(b xor e_j) - Delta_i(b)
                D = (flipped(i, j) - C_flip[j]) - Delta[i]
                Dij[i, j] = Dij[j, i] = np.mean(D**2)

        self._moments = {
            "d": np.mean(Delta**2, axis=1),
            "P": Delta @ Delta.T / m,
            "Dij": Dij,
            "q4": np.mean(Delta**4, axis=1),
        }
        return self._moments

    def agp_coeff(self, lam):
        r"""First-order nested-commutator AGP coefficient(s) at $\lambda$.

        Parameters
        ----------
        lam : float
            Value of the scheduling function.

        Returns
        -------
        float or numpy.ndarray
            $\alpha(\lambda)$ if ``uniform``, else the array of $\alpha_i(\lambda)$, see the
            class docstring.

        """
        m = self.nc_moments()
        s2 = self.mixer_strength**2
        if self.uniform:
            # Sum of the per-qubit system over all k, l
            off = m["Dij"].sum() / 2
            return -m["d"].sum() / (4 * s2 * (1 - lam) ** 2 * (m["P"].sum() + off) + lam**2 * m["q4"].sum())
        Q = m["Dij"] + np.diag(m["Dij"].sum(axis=1))
        T = s2 * (1 - lam) ** 2 * (4 * m["P"] + Q) + lam**2 * np.diag(m["q4"])
        # lstsq: a qubit that C does not depend on gives a zero row
        return -np.linalg.lstsq(T, m["d"], rcond=None)[0]

    def _precompute_timegrid(self, N_steps, T):
        """Compute lambda(t, T) and its time derivative on the midpoints of the N_steps intervals."""
        t_sym, T_sym = sp.symbols("t T", real=True)
        # Midpoint rule, as in DCQOProblem: avoids t=0/T exactly
        t_list = (np.arange(N_steps) + 0.5) * (T / N_steps)

        lam_func = sp.lambdify((t_sym, T_sym), self.lam_func(), "numpy")
        lamdot_func = sp.lambdify((t_sym, T_sym), sp.diff(self.lam_func(), t_sym), "numpy")

        self.lam = np.broadcast_to(np.asarray(lam_func(t_list, T), dtype=float), (N_steps,))
        self.lamdot = np.broadcast_to(np.asarray(lamdot_func(t_list, T), dtype=float), (N_steps,))

    # ----- quantum part -----

    def _apply_flip_term(self, qarg, i, theta):
        r"""Apply $e^{i\theta Y_i g_i}$ with two oracle calls."""
        if self.local_phase_oracle is None:
            oracle = self.phase_oracle
        else:

            def oracle(q, gamma):
                self.local_phase_oracle(q, gamma, i)

        # V_i^dagger, with V = S H mapping Z to Y
        s_dg(qarg[i])
        h(qarg[i])
        # exp(i theta Delta_i) = exp(i theta C(b xor e_i)) exp(-i theta C(b))
        oracle(qarg, theta)
        x(qarg[i])
        oracle(qarg, -theta)
        x(qarg[i])
        # V_i
        h(qarg[i])
        s(qarg[i])

    def apply_nc_agp(self, qarg, theta, agp_order=1, reverse=False):
        r"""Apply $\prod_i e^{i\theta_i Y_i g_i}$, the Trotterized $e^{-i \sum_i \theta_i\, i[X_i, C]}$.

        The factors for different qubits do not commute.

        Parameters
        ----------
        qarg : :ref:`QuantumVariable`
            The problem register.
        theta : float or array
            Rotation parameter $dt\, \dot\lambda\, \alpha\, s$ for one LCD step, either one value
            for all qubits or one per qubit.
        agp_order : int, optional
            1 sweeps the qubits once. 2 uses the symmetric sweep (forward and backward with
            $\theta_i/2$ each), which doubles the oracle calls. The default is 1.
        reverse : bool, optional
            Sweep the qubits in descending order. The default is ``False``.

        """
        n = self.num_qubits
        theta = np.broadcast_to(theta, (n,))
        order = list(reversed(range(n))) if reverse else list(range(n))
        if agp_order == 1:
            sweep = [(i, theta[i]) for i in order]
        elif agp_order == 2:  # noqa: PLR2004 -- Trotter order
            sweep = [(i, theta[i] / 2) for i in order] + [(i, theta[i] / 2) for i in reversed(order)]
        else:
            raise ValueError(f"agp_order must be 1 or 2 (received {agp_order!r}).")
        for i, th in sweep:
            self._apply_flip_term(qarg, i, th)

    def apply_lcd_hamiltonian(self, qarg, N_steps, T, agp=True, agp_order=1, alternate=False):  # noqa: PLR0913
        r"""Prepare the initial state and apply the Trotterized LCD evolution.

        Each step applies the mixer, the cost oracle and the AGP, in that order.

        Parameters
        ----------
        qarg : :ref:`QuantumVariable`
            The problem register, in the state $|0\rangle^{\otimes n}$.
        N_steps : int
            Number of Trotter steps.
        T : float
            Evolution time.
        agp : bool, optional
            If ``False``, the AGP is left out (plain digitized annealing). The default is ``True``.
        agp_order : int, optional
            Trotter order of the AGP sweep over the qubits, see :meth:`apply_nc_agp`.
        alternate : bool, optional
            Reverse the qubit order of the AGP sweep on every second step. Reduces the
            Trotter error of the sweep at no extra cost. The default is ``False``.

        """
        if len(qarg) != self.num_qubits:
            raise ValueError(f"qarg has {len(qarg)} qubits, expected {self.num_qubits}.")

        # Ground state of +s * sum X
        h(qarg)
        z(qarg)

        self._precompute_timegrid(N_steps, T)
        dt = T / N_steps
        s_mix = self.mixer_strength

        for k in range(N_steps):
            lam, lamdot = self.lam[k], self.lamdot[k]
            # exp(-i dt (1-lam) s sum X), rx(phi) = exp(-i phi X / 2)
            rx(2 * dt * (1 - lam) * s_mix, qarg)
            self.phase_oracle(qarg, dt * lam)
            if agp:
                theta = dt * lamdot * self.agp_coeff(lam) * s_mix
                self.apply_nc_agp(qarg, theta, agp_order=agp_order, reverse=alternate and k % 2 == 1)

    def run(  # noqa: PLR0913 -- public, keyword-callable API shape
        self,
        qarg,
        N_steps,
        T,
        method="LCD",
        agp=True,
        agp_order=1,
        alternate=False,
        mes_kwargs=None,
    ):
        """Run the counterdiabatic evolution and measure.

        Parameters
        ----------
        qarg : :ref:`QuantumVariable`
            The problem register. Its measurement labels must be bitstrings, with character
            ``i`` belonging to qubit ``i`` (the default for :ref:`QuantumVariable`).
        N_steps : int
            Number of Trotter steps.
        T : float
            Evolution time.
        method : str, optional
            Only ``"LCD"`` is implemented.
        agp : bool, optional
            If ``False``, run plain digitized annealing for comparison. The default is ``True``.
        agp_order : int, optional
            Trotter order of the AGP sweep over the qubits, see :meth:`apply_nc_agp`.
        alternate : bool, optional
            Reverse the AGP sweep on every second step, see :meth:`apply_lcd_hamiltonian`.
        mes_kwargs : dict, optional
            Keyword arguments for :meth:`get_measurement <qrisp.QuantumVariable.get_measurement>`,
            for example ``{"shots": 5000}``.

        Returns
        -------
        res_dict : dict
            ``{bitstring: [probability, cost]}``, with ``cost`` from ``cost_fn``.

        """
        if method != "LCD":
            raise ValueError(f'"{method}" is not an option for method. Only "LCD" is implemented.')

        self.apply_lcd_hamiltonian(qarg, N_steps, T, agp=agp, agp_order=agp_order, alternate=alternate)
        meas = qarg.get_measurement(**(mes_kwargs or {}))

        labels = list(meas.keys())
        if not all(isinstance(label, str) for label in labels):
            raise TypeError("Measurement labels must be bitstrings. Use a plain QuantumVariable as qarg.")
        B = np.array([[int(c) for c in label] for label in labels], dtype=np.int8)
        costs = np.asarray(self.cost_fn(B), dtype=float)
        return {label: [meas[label], float(c)] for label, c in zip(labels, costs)}

    # ----- evaluation -----

    def exact_minimum(self):
        """Minimum of the cost and all bitstrings attaining it, by enumeration of all 2^n bitstrings.

        Returns
        -------
        tuple
            ``(min_cost, [bitstrings])``.

        """
        B = self._all_bitstrings()
        C = np.asarray(self.cost_fn(B), dtype=float)
        c_min = C.min()
        opt = ["".join(map(str, b)) for b in B[np.isclose(C, c_min)]]
        return c_min, opt

    def evaluate(self, res, reference_cost=None, tol=1e-9):
        """Summarize a result of :meth:`run` with respect to the original cost.

        Parameters
        ----------
        res : dict
            ``{bitstring: [probability, cost]}`` as returned by :meth:`run`.
        reference_cost : float, optional
            Optimal cost, for ``p_optimal``. If not given and the averages were computed by
            enumeration, it is computed with :meth:`exact_minimum`.
        tol : float, optional
            Tolerance for counting a bitstring as optimal.

        Returns
        -------
        dict
            ``expected_cost``, ``best`` (lowest-cost measured bitstring and its cost), and if
            available ``p_optimal`` and ``reference_cost``, plus ``p_feasible`` and
            ``best_feasible`` if ``feasible_fn`` is set.

        """
        labels = list(res.keys())
        probs = np.array([res[label][0] for label in labels], dtype=float)
        costs = np.array([res[label][1] for label in labels], dtype=float)

        out = {"expected_cost": float(probs @ costs)}
        k = int(np.argmin(costs))
        out["best"] = (labels[k], float(costs[k]))

        if reference_cost is None and self.n_samples is None:
            reference_cost = self.exact_minimum()[0]
        if reference_cost is not None:
            out["reference_cost"] = float(reference_cost)
            out["p_optimal"] = float(probs[costs <= reference_cost + tol].sum())

        if self.feasible_fn is not None:
            B = np.array([[int(c) for c in label] for label in labels], dtype=np.int8)
            feas = np.asarray(self.feasible_fn(B), dtype=bool)
            out["p_feasible"] = float(probs[feas].sum())
            if feas.any():
                kf = int(np.flatnonzero(feas)[np.argmin(costs[feas])])
                out["best_feasible"] = (labels[kf], float(costs[kf]))
            else:
                out["best_feasible"] = None

        return out
