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
from scipy.optimize import Bounds, minimize

from qrisp import h, rx, ry, rz, s, s_dg, x, z


def _default_lam_func():
    """Default scheduling function, the same as in :func:`create_COLD_instance`."""
    t, T = sp.symbols("t T", real=True)
    return sp.sin(sp.pi / 2 * sp.sin(sp.pi * t / (2 * T)) ** 2) ** 2


class OracleDCQOProblem:
    r"""Counterdiabatic driving (LCD and COLD) for a diagonal cost function given as a phase oracle.

    Minimizes an arbitrary diagonal cost $C(b)$ over bitstrings $b \in \{0,1\}^n$. The cost
    does not need to be a QUBO or any other polynomial, and no Pauli decomposition of $C$ is
    ever built. It enters only through

    * ``phase_oracle(qarg, gamma)``, applying $e^{-i\gamma C}$ to ``qarg``, and
    * ``cost_fn(B)``, evaluating $C$ classically for a batch of bitstrings.

    The system is evolved along

    .. math::
        H(\lambda) = (1-\lambda)\, s \sum_i X_i + \lambda C + f(\lambda) \sum_i Z_i, \qquad s = \text{mixer\_strength},

    starting in the ground state $\ket{-}^{\otimes n}$ of the mixer. The control pulse $f$ is only
    present for COLD (``method="COLD"``); LCD has $f = 0$. Finite differences of $C$ are written as
    $\Delta_i(b) = C(b \oplus e_i) - C(b)$, $D_{ij}(b) = \Delta_i(b \oplus e_j) - \Delta_i(b)$ and
    $g_i(b) = C(b|_{b_i=1}) - C(b|_{b_i=0})$, so that $\Delta_i = z_i g_i$ with $z_i = 1 - 2 b_i$.
    $g_i$ is the local energy difference of bit $i$ and does not depend on $b_i$ itself.

    **Nested-commutator AGP** (``agp_type="nc"``). The adiabatic gauge potential (AGP) uses the
    first-order ansatz $A_\lambda = \alpha\, i[H, \partial_\lambda H]$. With
    $\mu = f + (1-\lambda) f'$ and $f' = df/d\lambda$,

    .. math::
        A_\lambda = \alpha\, s\, i[H_X, C + \mu H_Z] = -\alpha\, s \sum_i Y_i\, (g_i - 2\mu).

    Each factor of the Trotterized AGP is built from two oracle calls and one ``ry`` gate,

    .. math::
        e^{i\theta Y_i (g_i - 2\mu)} = V_i\, X_i U_C(-\theta) X_i\, U_C(\theta)\, V_i^\dagger\; e^{-2i\theta\mu Y_i},
        \qquad V = S H, \quad U_C(\gamma) = e^{-i\gamma C},

    because $V Z V^\dagger = Y$, $Z_i g_i = C(b \oplus e_i) - C(b)$ and $Y_i$ commutes with $g_i$.
    $\alpha(\lambda)$ minimizes the action $S = \mathrm{Tr}[G^2]$, $G = \partial_\lambda H + i[A_\lambda, H]$.
    With $\Delta^\mu_i = \Delta_i - 2\mu z_i$ (differences of $C + \mu H_Z$) and
    $\Delta^H_i = \lambda \Delta_i - 2 f z_i$ (differences of $\lambda C + f H_Z$):

    .. math::
        \alpha = -\frac{\sum_i E[(\Delta^\mu_i)^2]}
            {4 s^2 (1-\lambda)^2 \left(E[(\sum_i \Delta^\mu_i)^2] + \sum_{i<j} E[D_{ij}^2]\right)
             + \sum_i E[(\Delta^\mu_i)^2 (\Delta^H_i)^2]}.

    With ``uniform=False`` every qubit gets its own coefficient,
    $A_\lambda = \sum_i \alpha_i\, s\, i[X_i, C + \mu Z_i]$, which uses the same circuit with a separate
    angle per qubit. Minimizing the same action gives the linear system $T\alpha = -d$ with
    $d_k = E[(\Delta^\mu_k)^2]$ and

    .. math::
        T_{kl} = s^2 (1-\lambda)^2 \left(4 E[\Delta^\mu_k \Delta^\mu_l] + Q_{kl}\right)
                 + \delta_{kl}\, E[(\Delta^\mu_k)^2 (\Delta^H_k)^2],
        \qquad Q_{kl} = E[D_{kl}^2]\ (k \neq l), \quad Q_{kk} = \sum_{j \neq k} E[D_{kj}^2].

    **Local AGP** (``agp_type="local"``). The ansatz $A_\lambda = \sum_i \alpha_i Y_i$ is one ``ry``
    gate per qubit and needs no oracle calls. Minimizing the action gives

    .. math::
        \alpha_i = -\frac{s\,(\mu - E[g_i]/2)}
                         {2\left(s^2(1-\lambda)^2 + f^2 - \lambda f E[g_i] + \lambda^2 E[g_i^2]/4\right)},

    and the uniform coefficient sums numerator and denominator over $i$. For a QUBO this is the
    coefficient of :func:`create_COLD_instance` with ``agp_type="local"``.

    All expectation values are uniform averages over bitstrings. They are polynomials in $f$ and
    $\mu$ whose coefficients are computed once, by exact enumeration of all $2^n$ bitstrings
    (default) or by Monte Carlo sampling (``n_samples``).

    **COLD** adds the control pulse $f = \sum_k \beta_k \sin(\omega_k t/T)$ with
    $\omega_k = 2\pi k$ (COLD) or $\omega_k = 2\pi k (1 + r_k)$, $r_k$ uniform in $[-1/2, 1/2]$
    (COLD-CRAB), applied as ``rz`` gates. The parameters $\beta_k$ minimize the expected cost of
    the final state. The circuit is a fixed product formula, so the optimization can use
    :meth:`model_state`, an exact numpy simulation of that product formula from the cost table
    (``objective_backend="model"``), or simulate the Qrisp circuit at every evaluation
    (``objective_backend="circuit"``).

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
    agp_type : str, optional
        ``"nc"`` for the first-order nested commutator (two oracle calls per qubit and step) or
        ``"local"`` for $\sum_i \alpha_i Y_i$ (no oracle calls). The default is ``"nc"``.
    local_phase_oracle : callable, optional
        ``local_phase_oracle(qarg, gamma, i)`` applies $e^{-i\gamma C_i}$, where $C_i$ keeps
        only the terms of $C$ that depend on bit ``i``. Used inside the nested-commutator AGP
        instead of the full oracle, which gives the same unitary because $g_i$ only sees these terms.
    feasible_fn : callable, optional
        Vectorized feasibility check with the same input as ``cost_fn``, returning a boolean
        array. Only used by :meth:`evaluate`.
    n_samples : int, optional
        If given, the AGP averages are estimated from this many uniformly random bitstrings
        instead of enumerating all $2^n$. Use this when $2^n$ is too large.
    seed : int, optional
        Seed for the Monte Carlo sampling, the CRAB frequencies and the initial COLD parameters.

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

        # COLD with two pulse parameters, optimized on the exact model of the circuit
        res = problem.run(QuantumVariable(n), N_steps=10, T=1.0, method="COLD", N_opt=2)
        print(problem.opt_params, problem.evaluate(res))

    """

    def __init__(  # noqa: PLR0913, PLR0917 -- public, keyword-callable API shape
        self,
        cost_fn,
        phase_oracle,
        num_qubits,
        lam_func=None,
        mixer_strength=1.0,
        uniform=True,
        agp_type="nc",
        local_phase_oracle=None,
        feasible_fn=None,
        n_samples=None,
        seed=None,
    ):
        """Construct an OracleDCQOProblem instance. See class docstring for parameter details."""
        if agp_type not in ("nc", "local"):
            raise ValueError(f"{agp_type!r} is not a valid option as agp_type. Valid options are 'nc' and 'local'.")
        self.cost_fn = cost_fn
        self.phase_oracle = phase_oracle
        self.num_qubits = num_qubits
        self.lam_func = lam_func if lam_func is not None else _default_lam_func
        self.mixer_strength = mixer_strength
        self.uniform = uniform
        self.agp_type = agp_type
        self.local_phase_oracle = local_phase_oracle
        self.feasible_fn = feasible_fn
        self.n_samples = n_samples
        self.seed = seed

        self.lam = None
        self.lamdot = None
        self._moments = None
        self._cost_table = None
        self._crab_r = None
        self.opt_params = None
        self.opt_value = None

    # ----- classical part -----

    def _all_bitstrings(self):
        """All 2^n bitstrings as rows, with bit i of the row index on column i."""
        idx = np.arange(1 << self.num_qubits)
        return ((idx[:, None] >> np.arange(self.num_qubits)) & 1).astype(np.int8)

    def cost_table(self):
        r"""$C(b)$ for every basis index $b = \sum_i b_i 2^i$, computed on first use and cached."""
        if self._cost_table is None:
            self._cost_table = np.asarray(self.cost_fn(self._all_bitstrings()), dtype=float)
        return self._cost_table

    def nc_moments(self):
        r"""Uniform averages of finite differences of $C$ that determine the AGP coefficients.

        Computed on first use and cached.

        Returns
        -------
        dict
            ``d`` with $E[\Delta_k^2]$, ``P`` with $P_{kl} = E[\Delta_k \Delta_l]$, ``Dij`` with
            $E[D_{kl}^2]$ for $k \neq l$ (zero diagonal), ``q4`` with $E[\Delta_k^4]$, ``zD`` with
            $E[z_k \Delta_k] = E[g_k]$, ``Pz`` with $E[\Delta_k z_l]$ and ``zD3`` with $E[z_k \Delta_k^3]$.

        """
        if self._moments is not None:
            return self._moments

        n = self.num_qubits
        if self.n_samples is None:
            # Exact enumeration: a bit flip is an XOR on the row index, so C is evaluated only once.
            B = self._all_bitstrings()
            idx = np.arange(1 << n)
            C = self.cost_table()

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
        Zb = (1 - 2 * B.astype(float)).T  # z_k = 1 - 2 b_k, shape (n, m)
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
            "zD": np.mean(Zb * Delta, axis=1),
            "Pz": Delta @ Zb.T / m,
            "zD3": np.mean(Zb * Delta**3, axis=1),
        }
        return self._moments

    def agp_coeff(self, lam, f=0.0, f_deriv=0.0):
        r"""AGP coefficient(s) at $\lambda$ for the chosen ``agp_type``, see the class docstring.

        Parameters
        ----------
        lam : float
            Value of the scheduling function.
        f : float, optional
            Control pulse amplitude (COLD). The default is 0 (LCD).
        f_deriv : float, optional
            Derivative $df/d\lambda$ of the control pulse. The default is 0 (LCD).

        Returns
        -------
        float or numpy.ndarray
            $\alpha(\lambda)$ if ``uniform``, else the array of $\alpha_i(\lambda)$.

        """
        m = self.nc_moments()
        s_mix = self.mixer_strength
        mu = f + (1 - lam) * f_deriv

        if self.agp_type == "local":
            # D^(1)_i = -g_i/2 + f is the part of the diagonal of H that flips sign with Z_i
            a = (1 - lam) * s_mix
            Eg, Eg2 = m["zD"], m["d"]
            num = s_mix * (mu - Eg / 2)
            den = 2 * (a**2 + f**2 - lam * f * Eg + lam**2 * Eg2 / 4)
            if self.uniform:
                return -num.sum() / den.sum()
            return -num / den

        n = self.num_qubits
        s2 = s_mix**2
        # Moments of Delta^mu = Delta - 2 mu z and Delta^H = lam Delta - 2 f z, expanded with z^2 = 1
        d = m["d"] - 4 * mu * m["zD"] + 4 * mu**2
        P = m["P"] - 2 * mu * (m["Pz"] + m["Pz"].T) + 4 * mu**2 * np.eye(n)
        u2, u4, zu, zu3 = m["d"], m["q4"], m["zD"], m["zD3"]
        q = (
            lam**2 * u4
            - 4 * lam * (f + mu * lam) * zu3
            + 4 * (f**2 + 4 * mu * lam * f + mu**2 * lam**2) * u2
            - 16 * mu * f * (f + mu * lam) * zu
            + 16 * mu**2 * f**2
        )
        if self.uniform:
            # Sum of the per-qubit system over all k, l
            off = m["Dij"].sum() / 2
            return -d.sum() / (4 * s2 * (1 - lam) ** 2 * (P.sum() + off) + q.sum())
        Q = m["Dij"] + np.diag(m["Dij"].sum(axis=1))
        T = s2 * (1 - lam) ** 2 * (4 * P + Q) + np.diag(q)
        # lstsq: a qubit that C does not depend on gives a zero row
        return -np.linalg.lstsq(T, d, rcond=None)[0]

    def _precompute_timegrid(self, N_steps, T):
        """Compute lambda(t, T) and its time derivative on the midpoints of the N_steps intervals."""
        t_sym, T_sym = sp.symbols("t T", real=True)
        # Midpoint rule, as in DCQOProblem: avoids t=0/T exactly
        t_list = (np.arange(N_steps) + 0.5) * (T / N_steps)

        lam_func = sp.lambdify((t_sym, T_sym), self.lam_func(), "numpy")
        lamdot_func = sp.lambdify((t_sym, T_sym), sp.diff(self.lam_func(), t_sym), "numpy")

        self.lam = np.broadcast_to(np.asarray(lam_func(t_list, T), dtype=float), (N_steps,))
        self.lamdot = np.broadcast_to(np.asarray(lamdot_func(t_list, T), dtype=float), (N_steps,))

    def _pulses(self, N_steps, T, beta, CRAB=False):
        r"""Control pulse $f$ and $df/d\lambda$ on the time grid, for the parameters ``beta``.

        $f = \sum_k \beta_k \sin(\omega_k g)$ with $g = t/T$ and $\omega_k = 2\pi (k+1)(1 + r_k)$, as in
        :class:`DCQOProblem`. The CRAB offsets $r_k$ are drawn once and reused until the next
        :meth:`run`, so that optimized parameters fit the basis they were optimized for.
        """
        self._precompute_timegrid(N_steps, T)
        beta = np.asarray(beta, dtype=float)
        if beta.size == 0:
            return np.zeros(N_steps), np.zeros(N_steps)
        if CRAB:
            if self._crab_r is None or len(self._crab_r) != beta.size:
                self._crab_r = np.random.default_rng(self.seed).uniform(-0.5, 0.5, beta.size)
            r = self._crab_r
        else:
            r = np.zeros(beta.size)
        g = (np.arange(N_steps) + 0.5) / N_steps
        # dg/dlam = (1/T) dt/dlam
        g_deriv = 1.0 / (T * self.lamdot)
        omega = 2 * np.pi * np.arange(1, beta.size + 1) * (1 + r)
        f = np.sin(np.outer(g, omega)) @ beta
        f_deriv = (omega * np.cos(np.outer(g, omega))) @ beta * g_deriv
        return f, f_deriv

    def _agp_sweep(self, theta, agp_order, reverse):
        """Qubit order and angle of the nested-commutator factors of one AGP step."""
        n = self.num_qubits
        theta = np.broadcast_to(theta, (n,))
        order = list(reversed(range(n))) if reverse else list(range(n))
        if agp_order == 1:
            return [(i, theta[i]) for i in order]
        if agp_order == 2:  # noqa: PLR2004 -- Trotter order
            return [(i, theta[i] / 2) for i in order] + [(i, theta[i] / 2) for i in reversed(order)]
        raise ValueError(f"agp_order must be 1 or 2 (received {agp_order!r}).")

    def _step_angles(self, k, dt, f, f_deriv):
        """AGP angles of step k: theta per qubit and mu (``nc``), or the ry half-angles (``local``)."""
        lam, lamdot = self.lam[k], self.lamdot[k]
        alpha = self.agp_coeff(lam, f, f_deriv)
        if self.agp_type == "local":
            return dt * lamdot * np.broadcast_to(alpha, (self.num_qubits,)), None
        return dt * lamdot * alpha * self.mixer_strength, f + (1 - lam) * f_deriv

    # ----- quantum part -----

    def _apply_flip_term(self, qarg, i, theta, mu=0.0):
        r"""Apply $e^{i\theta Y_i (g_i - 2\mu)}$ with two oracle calls and one ``ry``."""
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
        if mu != 0:
            # exp(-2i theta mu Y_i), commutes with Y_i g_i; ry(phi) = exp(-i phi Y / 2)
            ry(4 * theta * mu, qarg[i])

    def apply_nc_agp(self, qarg, theta, agp_order=1, reverse=False, mu=0.0):
        r"""Apply $\prod_i e^{i\theta_i Y_i (g_i - 2\mu)}$, the Trotterized nested-commutator AGP.

        The factors for different qubits do not commute.

        Parameters
        ----------
        qarg : :ref:`QuantumVariable`
            The problem register.
        theta : float or array
            Rotation parameter $dt\, \dot\lambda\, \alpha\, s$ for one step, either one value
            for all qubits or one per qubit.
        agp_order : int, optional
            1 sweeps the qubits once. 2 uses the symmetric sweep (forward and backward with
            $\theta_i/2$ each), which doubles the oracle calls. The default is 1.
        reverse : bool, optional
            Sweep the qubits in descending order. The default is ``False``.
        mu : float, optional
            Control part $\mu = f + (1-\lambda) f'$ of the nested commutator (COLD). The default is 0.

        """
        for i, th in self._agp_sweep(theta, agp_order, reverse):
            self._apply_flip_term(qarg, i, th, mu)

    def _apply_evolution(self, qarg, N_steps, T, beta, CRAB, agp, agp_order, alternate):  # noqa: PLR0913, PLR0917
        """Prepare the initial state and apply all Trotter steps (LCD for ``beta=None``, else COLD)."""
        if len(qarg) != self.num_qubits:
            raise ValueError(f"qarg has {len(qarg)} qubits, expected {self.num_qubits}.")

        # Ground state of +s * sum X
        h(qarg)
        z(qarg)

        f_list, fd_list = self._pulses(N_steps, T, [] if beta is None else beta, CRAB)
        dt = T / N_steps
        s_mix = self.mixer_strength

        for k in range(N_steps):
            lam = self.lam[k]
            # exp(-i dt (1-lam) s sum X), rx(phi) = exp(-i phi X / 2)
            rx(2 * dt * (1 - lam) * s_mix, qarg)
            self.phase_oracle(qarg, dt * lam)
            if agp:
                theta, mu = self._step_angles(k, dt, f_list[k], fd_list[k])
                if self.agp_type == "local":
                    # exp(-i dt lamdot alpha_i Y_i)
                    for i in range(self.num_qubits):
                        ry(2 * theta[i], qarg[i])
                else:
                    self.apply_nc_agp(qarg, theta, agp_order=agp_order, reverse=alternate and k % 2 == 1, mu=mu)
            if beta is not None:
                # exp(-i dt f sum Z), rz(phi) = exp(-i phi Z / 2)
                rz(2 * dt * f_list[k], qarg)

    def apply_lcd_hamiltonian(self, qarg, N_steps, T, agp=True, agp_order=1, alternate=False):  # noqa: PLR0913, PLR0917
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
            Trotter order of the nested-commutator sweep over the qubits, see :meth:`apply_nc_agp`.
        alternate : bool, optional
            Reverse the qubit order of the nested-commutator sweep on every second step. Reduces
            the Trotter error of the sweep at no extra cost. The default is ``False``.

        """
        self._apply_evolution(qarg, N_steps, T, None, False, agp, agp_order, alternate)

    def apply_cold_hamiltonian(  # noqa: PLR0913, PLR0917 -- public, keyword-callable API shape
        self, qarg, N_steps, T, beta, CRAB=False, agp_order=1, alternate=False
    ):
        r"""Prepare the initial state and apply the Trotterized COLD evolution for the pulse parameters ``beta``.

        Each step applies the mixer, the cost oracle, the AGP (with coefficients that depend on
        the pulse) and the control pulse $e^{-i\, dt\, f \sum_i Z_i}$, in that order.

        Parameters
        ----------
        qarg : :ref:`QuantumVariable`
            The problem register, in the state $|0\rangle^{\otimes n}$.
        N_steps : int
            Number of Trotter steps.
        T : float
            Evolution time.
        beta : array
            Pulse parameters $\beta_k$.
        CRAB : bool, optional
            Use the randomized CRAB frequencies. The default is ``False``.
        agp_order, alternate
            See :meth:`apply_lcd_hamiltonian`.

        """
        self._apply_evolution(qarg, N_steps, T, beta, CRAB, True, agp_order, alternate)

    # ----- exact model of the circuit -----

    def model_state(  # noqa: PLR0913, PLR0917 -- mirrors the circuit options
        self, N_steps, T, beta=None, CRAB=False, agp=True, agp_order=1, alternate=False
    ):
        r"""Statevector of the LCD (``beta=None``) or COLD circuit, computed with numpy from the cost table.

        Applies exactly the product formula of :meth:`apply_lcd_hamiltonian` and
        :meth:`apply_cold_hamiltonian`, factor by factor, on a vector of $2^n$ amplitudes. The
        cost is diagonal, so every factor is a phase or a rotation on pairs of basis states.
        Needs $C$ on all $2^n$ bitstrings (:meth:`cost_table`).

        Returns
        -------
        numpy.ndarray
            Amplitudes over the basis index $b = \sum_i b_i 2^i$.

        """
        n = self.num_qubits
        idx = np.arange(1 << n)
        Zb = 1 - 2 * ((idx[:, None] >> np.arange(n)) & 1)
        C = self.cost_table()
        g = [C[idx | (1 << i)] - C[idx & ~(1 << i)] for i in range(n)]

        def rot_y(psi, i, a):
            # exp(i a Y_i), a constant or a function of the other bits: Y|0> = i|1>, Y|1> = -i|0>
            return np.cos(a) * psi + np.sin(a) * Zb[:, i] * psi[idx ^ (1 << i)]

        psi = np.prod(Zb, axis=1) / np.sqrt(1 << n) + 0j
        f_list, fd_list = self._pulses(N_steps, T, [] if beta is None else beta, CRAB)
        dt = T / N_steps
        for k in range(N_steps):
            lam = self.lam[k]
            phi = dt * (1 - lam) * self.mixer_strength
            for i in range(n):
                psi = np.cos(phi) * psi - 1j * np.sin(phi) * psi[idx ^ (1 << i)]
            psi = psi * np.exp(-1j * dt * lam * C)
            if agp:
                theta, mu = self._step_angles(k, dt, f_list[k], fd_list[k])
                if self.agp_type == "local":
                    for i in range(n):
                        psi = rot_y(psi, i, -theta[i])
                else:
                    for i, th in self._agp_sweep(theta, agp_order, alternate and k % 2 == 1):
                        psi = rot_y(psi, i, th * (g[i] - 2 * mu))
            if beta is not None:
                psi = psi * np.exp(-1j * dt * f_list[k] * Zb.sum(axis=1))
        return psi

    def model_expectation(self, N_steps, T, beta=None, **kwargs):
        """Expected cost of the final state of :meth:`model_state`."""
        return float(np.abs(self.model_state(N_steps, T, beta, **kwargs)) ** 2 @ self.cost_table())

    # ----- COLD optimization -----

    def _circuit_expectation(self, N_steps, T, beta, CRAB, agp_order, alternate, mes_kwargs):  # noqa: PLR0913, PLR0917
        """Expected cost from a simulation of the Qrisp circuit."""
        from qrisp import QuantumVariable

        qarg = QuantumVariable(self.num_qubits)
        self.apply_cold_hamiltonian(qarg, N_steps, T, beta, CRAB=CRAB, agp_order=agp_order, alternate=alternate)
        meas = qarg.get_measurement(**(mes_kwargs or {}))
        qarg.delete()
        labels = list(meas.keys())
        B = np.array([[int(c) for c in label] for label in labels], dtype=np.int8)
        return float(np.array([meas[label] for label in labels]) @ np.asarray(self.cost_fn(B), dtype=float))

    def optimize_pulses(  # noqa: PLR0913, PLR0917 -- public, keyword-callable API shape
        self,
        N_steps,
        T,
        N_opt,
        CRAB=False,
        optimizer="COBYQA",
        options=None,
        bounds=None,
        n_restarts=3,
        objective_backend="model",
        agp_order=1,
        alternate=False,
        mes_kwargs=None,
    ):
        r"""Optimize the COLD pulse parameters against the expected cost of the final state.

        Parameters
        ----------
        N_steps : int
            Number of Trotter steps.
        T : float
            Evolution time.
        N_opt : int
            Number of pulse parameters $\beta_k$.
        CRAB : bool, optional
            Use the randomized CRAB frequencies. The default is ``False``.
        optimizer : str, optional
            `SciPy optimization routine <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.
            The default is ``"COBYQA"``, as in :class:`DCQOProblem`.
        options : dict, optional
            Options for the optimizer.
        bounds : tuple, optional
            Common lower and upper bound of the parameters. The default is no bounds.
        n_restarts : int, optional
            Number of optimizations from random initial points; the best is kept. The default is 3.
        objective_backend : str, optional
            ``"model"`` evaluates the expected cost with :meth:`model_state` (exact, needs $C$ on
            all $2^n$ bitstrings), ``"circuit"`` simulates the Qrisp circuit at every evaluation.
            The default is ``"model"``.
        agp_order, alternate
            See :meth:`apply_lcd_hamiltonian`.
        mes_kwargs : dict, optional
            Keyword arguments for the measurements of the ``"circuit"`` backend.

        Returns
        -------
        tuple
            ``(beta, expected_cost)`` of the best run. Also stored as ``opt_params`` and ``opt_value``.

        """
        if isinstance(N_opt, (bool, np.bool_)) or not isinstance(N_opt, (int, np.integer)) or N_opt < 1:
            raise ValueError(f"N_opt must be a positive integer for the COLD method (received {N_opt!r}).")
        if objective_backend == "model":

            def objective(beta):
                return self.model_expectation(N_steps, T, beta, CRAB=CRAB, agp_order=agp_order, alternate=alternate)

        elif objective_backend == "circuit":

            def objective(beta):
                return self._circuit_expectation(N_steps, T, beta, CRAB, agp_order, alternate, mes_kwargs)

        else:
            raise ValueError(f"{objective_backend!r} is not a valid objective_backend. Use 'model' or 'circuit'.")

        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(n_restarts):
            res = minimize(
                objective,
                rng.random(N_opt) * np.pi / 2,
                method=optimizer,
                options=options or {},
                bounds=None if bounds is None else Bounds(*bounds),
            )
            if best is None or res.fun < best.fun:
                best = res
        self.opt_params, self.opt_value = np.asarray(best.x), float(best.fun)
        return self.opt_params, self.opt_value

    def run(  # noqa: PLR0913, PLR0917 -- public, keyword-callable API shape
        self,
        qarg,
        N_steps,
        T,
        method="LCD",
        agp=True,
        agp_order=1,
        alternate=False,
        mes_kwargs=None,
        N_opt=None,
        CRAB=False,
        optimizer="COBYQA",
        options=None,
        bounds=None,
        n_restarts=3,
        objective_backend="model",
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
            ``"LCD"`` (default) or ``"COLD"``. COLD first optimizes ``N_opt`` pulse parameters with
            :meth:`optimize_pulses` and then runs the circuit with the best ones.
        agp : bool, optional
            If ``False``, run plain digitized annealing for comparison (LCD only). The default is ``True``.
        agp_order : int, optional
            Trotter order of the nested-commutator sweep over the qubits, see :meth:`apply_nc_agp`.
        alternate : bool, optional
            Reverse the nested-commutator sweep on every second step, see :meth:`apply_lcd_hamiltonian`.
        mes_kwargs : dict, optional
            Keyword arguments for :meth:`get_measurement <qrisp.QuantumVariable.get_measurement>`,
            for example ``{"shots": 5000}``.
        N_opt, CRAB, optimizer, options, bounds, n_restarts, objective_backend
            COLD only, see :meth:`optimize_pulses`. A new run draws new CRAB frequencies.

        Returns
        -------
        res_dict : dict
            ``{bitstring: [probability, cost]}``, with ``cost`` from ``cost_fn``.

        """
        if method == "LCD":
            self.apply_lcd_hamiltonian(qarg, N_steps, T, agp=agp, agp_order=agp_order, alternate=alternate)
        elif method == "COLD":
            self._crab_r = None
            beta, _ = self.optimize_pulses(
                N_steps,
                T,
                N_opt,
                CRAB=CRAB,
                optimizer=optimizer,
                options=options,
                bounds=bounds,
                n_restarts=n_restarts,
                objective_backend=objective_backend,
                agp_order=agp_order,
                alternate=alternate,
                mes_kwargs=mes_kwargs,
            )
            self.apply_cold_hamiltonian(qarg, N_steps, T, beta, CRAB=CRAB, agp_order=agp_order, alternate=alternate)
        else:
            raise ValueError(f'"{method}" is not an option for method. Choose "LCD" or "COLD".')

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
        C = self.cost_table()
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
