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

"""Provides helper functions to build and solve QUBO problems using COLD/LCD counterdiabatic driving."""

import itertools

import numpy as np
import sympy as sp

from qrisp.algorithms.cold import DCQOProblem, solve_alpha
from qrisp.core import QuantumVariable
from qrisp.operators.qubit import QubitOperator, X, Y, Z


def _order1_agp_coeffs(h, J, lam, f=0.0, f_deriv=0.0, *, uniform=True):  # noqa: PLR0913 -- one coefficient formula, four call sites
    r"""First-order AGP coefficients for the ansatz $A_\lambda = \sum_i \alpha_i \sigma^y_i$.

    Minimises the action $S = \mathrm{Tr}[G_\lambda^2]$ with
    $G_\lambda = \partial_\lambda H + i[A_\lambda, H]$ for the Hamiltonian the circuit evolves,

    .. math::
        H(\lambda) = (1-\lambda)\sum_i \sigma^x_i
                    + \lambda\Big(\sum_{i<j} J_{ij}\sigma^z_i\sigma^z_j + \sum_i h_i\sigma^z_i\Big)
                    + f\sum_i \sigma^z_i .

    Writing $b = 1-\lambda$ and $c_i = \lambda h_i + f$, the Pauli strings in $G_\lambda$ are
    trace-orthogonal, so the action is quadratic in the coefficients and

    .. math::
        \alpha_i = -\frac{b\dot c_i - c_i\dot b}
                        {2\left(b^2 + c_i^2 + \lambda^2\sum_{j\neq i} J_{ij}^2\right)},
        \qquad b\dot c_i - c_i\dot b = h_i + f + (1-\lambda)f'.

    The uniform case shares one coefficient across all sites, which sums numerator and
    denominator over $i$. Both forms reproduce an exact matrix minimisation of $S$ to machine
    precision, and the same derivation applied to Eq. (23) of `COLD
    <https://doi.org/10.1103/PRXQuantum.4.010312>`_ reproduces its Eq. (30).

    Parameters
    ----------
    h : np.array
        Onsite energies of the problem Hamiltonian.
    J : np.array
        Coupling energies of the problem Hamiltonian.
    lam : float
        Value of the scheduling function at this timestep.
    f : float, optional
        Control pulse amplitude (COLD). Zero for LCD, which has no control Hamiltonian.
    f_deriv : float, optional
        Derivative of the control pulse with respect to ``lam``. Zero for LCD.
    uniform : bool, optional
        Whether to share a single coefficient across all sites. The default is ``True``.

    Returns
    -------
    alph : list[float]
        The AGP coefficient per site, of length ``len(h)``.

    """
    N = len(h)
    c = lam * h + f
    b = 1 - lam
    nom = h + f + b * f_deriv
    # sum_{j != i} J_ij**2, per site
    J_sq = np.sum(J**2, axis=1) - np.diag(J) ** 2

    if uniform:
        alph = -np.sum(nom) / (2 * (N * b**2 + np.sum(c**2) + lam**2 * np.sum(J_sq)))
        return [alph] * N

    return [-nom[i] / (2 * (b**2 + c[i] ** 2 + lam**2 * J_sq[i])) for i in range(N)]


def _nc_uniform_agp_coeffs(h, J):
    r"""Build the shared coefficient of the nested-commutator AGP, as a function of the schedule.

    Minimises the action $S = \mathrm{Tr}[G_\lambda^2]$, $G_\lambda = \partial_\lambda H + i[A_\lambda, H]$,
    for the single merged ansatz operator $A_\lambda = \sum_i A_i$ with

    .. math::
        A_i = -2\Big(h_i\sigma^y_i
              + \sum_{j<i} J_{ij}(\sigma^y_i\sigma^z_j + \sigma^z_i\sigma^y_j)\Big),

    against the Hamiltonian the circuit evolves,

    .. math::
        H(\lambda) = (1-\lambda)\sum_i \sigma^x_i
                   + \lambda\Big(\sum_{i<j} J_{ij}\sigma^z_i\sigma^z_j + \sum_i h_i\sigma^z_i\Big)
                   + f\sum_i \sigma^z_i .

    The action is quadratic in the coefficient and every trace evaluates in closed form, giving

    .. math::
        \alpha = \frac{S_{h^2} + 2S_2 + \big(f + (1-\lambda)f'\big)S_h}
                      {4\big[(S_{h^2} + 8S_2)(1-\lambda)^2
                       + \lambda^2 (S_{h^4} + 2S_4 + 6S_{h^2R} + 6S_{adj})
                       + 2\lambda (S_{h^3} + 3S_{hR})f
                       + (S_{h^2} + 2S_2)f^2\big]} .

    The numerator at $f = f' = 0$ reproduces Eq. (S11) of `BF-DCQO
    <https://arxiv.org/abs/2405.13898>`_ for a uniform transverse field. Every aggregate is
    $O(N^2)$ to build and the coefficient is then $O(1)$ per timestep, where an explicit
    minimal-action solve would cost $O(N^4)$.

    Parameters
    ----------
    h : np.array
        Onsite energies of the problem Hamiltonian.
    J : np.array
        Coupling energies of the problem Hamiltonian.

    Returns
    -------
    alpha : callable
        A function ``alpha(lam, f=0.0, f_deriv=0.0)`` returning the coefficient for every site.
        LCD has no control Hamiltonian and calls it with ``f = f_deriv = 0``.

    """
    N = len(h)
    h = np.asarray(h, dtype=float)
    J_sq = np.asarray(J, dtype=float) ** 2
    np.fill_diagonal(J_sq, 0.0)
    h_sq = h**2

    S_h = np.sum(h)
    S_h2 = np.sum(h_sq)
    S_h3 = np.sum(h**3)
    S_h4 = np.sum(h_sq**2)
    # sums over unordered pairs i < j
    S_2 = np.sum(J_sq) / 2
    S_4 = np.sum(J_sq**2) / 2
    # sum_{i<j} J_ij**2 (h_i + h_j) and sum_{i<j} J_ij**2 (h_i**2 + h_j**2)
    S_hR = np.sum(J_sq * h[None, :])
    S_hsqR = np.sum(J_sq * h_sq[None, :])
    # sum over unordered pairs of distinct edges sharing a site
    R_i = np.sum(J_sq, axis=1)
    S_adj = (np.sum(R_i**2) - 2 * S_4) / 2

    def alpha(lam, f=0.0, f_deriv=0.0):
        nom = S_h2 + 2 * S_2 + (f + (1 - lam) * f_deriv) * S_h
        denom = 4 * (
            (S_h2 + 8 * S_2) * (1 - lam) ** 2
            + lam**2 * (S_h4 + 2 * S_4 + 6 * S_hsqR + 6 * S_adj)
            + 2 * lam * (S_h3 + 3 * S_hR) * f
            + (S_h2 + 2 * S_2) * f**2
        )
        return [nom / denom] * N

    return alpha


def _nested_commutator_operators(h, J):
    """A_i = -2 (h_i Y_i + sum_{j<i} J_ij (Y_i Z_j + Z_i Y_j)), the first-order NC ansatz.

    This is the ansatz of Eq. (1) of `BF-DCQO <https://arxiv.org/abs/2405.13898>`_, split into one
    operator per site so that non-uniform coefficients can be attached to it.
    """
    N = len(h)
    return [
        -2 * (h[i] * Y(i) + QubitOperator.sum(J[i][j] * (Y(i) * Z(j) + Z(i) * Y(j)) for j in range(i)))
        for i in range(N)
    ]


def create_COLD_instance(Q, uniform_AGP_coeffs, agp_type="order1"):
    """Create the necessary parameters and operators to initialize a DCQO problem instance for COLD.

    Parameters
    ----------
    Q : np.array
        The QUBO Matrix to be encoded in the Hamiltonian.
    uniform_AGP_coeffs : bool
        Whether to approximate the AGP with uniform or non-uniform coefficients.
    agp_type : str, optional
        Which approximation of the AGP to use, either ``order1`` (a sum of single-qubit Y
        operators) or ``nc`` (first-order nested commutators). The default is ``order1``.
        ``nc`` is the better approximation and is recommended for short evolution times, and
        requires ``uniform_AGP_coeffs=True``: the non-uniform coefficients have no closed form and
        their solver cannot take the symbolic control pulse COLD compiles into the circuit. Use
        :func:`create_LCD_instance` for the non-uniform nested-commutator ansatz.

    Returns
    -------
    collected operators : tuple
        Tuple containing the following functions and operators:
        Scheduling function (lam(t)), function for AGP coefficients (alpha), initial and problem
        Hamiltonian (H_init, H_prob), AGP function (A_lam), Coupling and onsite energies of
        problem Hamiltonian (J, h), inverse scheduling function (g(lam)), control Hamiltonian (H_control).

    """
    N = len(Q[0])
    h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
    J = 0.5 * Q

    def lam():
        t, T = sp.symbols("t T", real=True)
        lam_expr = sp.sin(sp.pi / 2 * sp.sin(sp.pi * t / (2 * T)) ** 2) ** 2
        return lam_expr

    # AGP coefficients
    if agp_type == "order1":

        def alpha(lam, f, f_deriv):
            return _order1_agp_coeffs(h, J, lam, f, f_deriv, uniform=uniform_AGP_coeffs)

    elif agp_type == "nc" and uniform_AGP_coeffs:
        alpha = _nc_uniform_agp_coeffs(h, J)

    elif agp_type == "nc":
        # Non-uniform nested-commutator coefficients have no closed form; they need the explicit
        # minimal-action solve in solve_alpha, which works on numpy arrays. COLD's exp_value
        # objective compiles a parametrized circuit and passes sympy Symbols as f and f_deriv,
        # which that solver cannot consume. Refuse rather than fail deep inside the compile.
        raise NotImplementedError(
            "agp_type='nc' with uniform_AGP_coeffs=False is not supported for COLD. Use "
            "uniform_AGP_coeffs=True, which has a closed-form coefficient, or run the "
            "non-uniform nested-commutator ansatz with method='LCD'."
        )

    else:
        raise ValueError(f"{agp_type} is not a valid option as agp_type. Valid options are 'order1' and 'nc'.")

    # Initial Hamiltonian
    H_init = 1 * sum([X(i) for i in range(N)])

    # Problem Hamiltonian
    H_prob = QubitOperator.sum(
        itertools.chain(
            (J[i][j] * Z(i) * Z(j) for i in range(N) for j in range(i, N)),
            (h[i] * Z(i) for i in range(N)),
        )
    )

    # AGP as function of alpha. A single QubitOperator signals uniform coefficients to
    # DCQOProblem, a list of them one coefficient per site.
    if agp_type == "order1":
        A_lam = sum([Y(i) for i in range(N)]) if uniform_AGP_coeffs else [Y(i) for i in range(N)]
    else:
        site_operators = _nested_commutator_operators(h, J)
        A_lam = QubitOperator.sum(site_operators) if uniform_AGP_coeffs else site_operators

    # Control Hamiltonian
    H_control = sum([Z(i) for i in range(N)])

    collected_operators = (Q, H_init, H_prob, A_lam, alpha, lam, H_control)

    return collected_operators


def create_LCD_instance(Q, agp_type, uniform_AGP_coeffs=True):
    """Create the necessary parameters and operators to initialize a DCQO problem instance for LCD.

    Parameters
    ----------
    Q : np.array
        The QUBO Matrix to be encoded in the Hamiltonian.
    agp_type : str
        Which approximation of the AGP to use. Can choose between ``order1``,
        ``order2``, ``nc`` (nested commutators up to first order).
    uniform_AGP_coeffs : bool
        Whether to approximate the AGP with uniform or non-uniform coefficients.

    Returns
    -------
    collected operators : tuple
        Tuple containing the following functions and operators:
        Scheduling function (lam(t)), function for AGP coefficients (alpha), initial and problem
        Hamiltonian (H_init, H_prob), AGP function (A_lam), Coupling and onsite energies of
        problem Hamiltonian (J, h), inverse scheduling function (g(lam)), control Hamiltonian (H_control).

    """

    def build_agp(agp_type, J, h):

        def order1():
            A_lam = [Y(i) for i in range(N)]
            return A_lam

        def nested_commutators(J, h):
            return _nested_commutator_operators(h, J)

        builders = {"order1": order1(), "nc": nested_commutators(J, h)}

        return builders[agp_type]

    def build_coeffs(agp_type, uniform_AGP_coeffs, J, h):

        def order1_uniform(J, h):
            # LCD has no control Hamiltonian, so f = f_deriv = 0.
            def alpha(lam):
                return _order1_agp_coeffs(h, J, lam, uniform=True)

            return alpha

        def order1_nonuniform(J, h):
            def alpha(lam):
                return _order1_agp_coeffs(h, J, lam, uniform=False)

            return alpha

        def nc_uniform(J, h):
            # LCD has no control Hamiltonian, so the coefficient is evaluated at f = f_deriv = 0.
            return _nc_uniform_agp_coeffs(h, J)

        def nc_nonuniform(J, h):
            def alpha(lam):
                alph = solve_alpha(h, J, lam)
                return alph

            return alpha

        builders = {
            ("order1", True): order1_uniform(J, h),
            ("order1", False): order1_nonuniform(J, h),
            ("nc", True): nc_uniform(J, h),
            ("nc", False): nc_nonuniform(J, h),
        }

        return builders[(agp_type, uniform_AGP_coeffs)]

    N = len(Q[0])
    h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
    J = 0.5 * Q

    def lam():
        t, T = sp.symbols("t T", real=True)
        lam_expr = sp.sin(sp.pi / 2 * sp.sin(sp.pi * t / (2 * T)) ** 2) ** 2
        return lam_expr

    # AGP coefficients
    coeff_func = build_coeffs(agp_type, uniform_AGP_coeffs, J, h)

    # Initial Hamiltonian
    H_init = 1 * sum([X(i) for i in range(N)])

    # Problem Hamiltonian
    # H_prob = sum(
    #    [sum([J[i][j] * Z(i) * Z(j) for j in range(i)]) for i in range(N)]
    # ) + sum([h[i] * Z(i) for i in range(N)])

    H_prob = QubitOperator.sum(
        itertools.chain(
            (J[i][j] * Z(i) * Z(j) for i in range(N) for j in range(i, N)),
            (h[i] * Z(i) for i in range(N)),
        )
    )

    # AGP
    A_lam = build_agp(agp_type, J, h)

    return Q, H_init, H_prob, A_lam, coeff_func, lam


def solve_QUBO(Q: np.array, problem_args: dict, run_args: dict):
    """Solves a QUBO Matrix using counterdiabatic driving.

    This method uses the pre-defined COLD/LCD operators
    (hamiltonian, scheduling function, AGP parameters) as described in the tutorial.
    To define your own operators, create a DCQO instance and use the ``run`` method.

    Parameters
    ----------
    Q : np.array
        QUBO Matrix to solve.
    problem_args : dict
        Holds arguments for DCQO problem creation (``method``: str ("COLD"/"LCD"), ``uniform``: bool,
        ``agp_type``: str ("order1"/"nc"), optional, default "order1"). ``agp_type`` applies to both
        methods. ``nc`` approximates the AGP with first-order nested commutators and is the better
        approximation, especially at short evolution times. For COLD, ``nc`` requires
        ``"uniform": True``.
    run_args : dict
        Holds arguments for running the DCQO instance
        (``N_steps``, ``T``, ``N_opt``, ``CRAB``,``objective``,
        ``precision``, ``backend``, ``exp_value_backend``).
        All optionas are also listed here: :meth:`DCQOProblem.run`.

    Returns
    -------
    result : dict
        The dictionary holding the QUBO vector results, with their probabilitites and cost.
        They are ordered from most to least likely and the dictionary entries are {"state": [prob, cost]}.


    Examples
    --------
    ::

        import numpy as np
        from qrisp.algorithms.cold import solve_QUBO

        Q = np.array([[-0.6,  0.2, -0.5, -0.4, -0.6,  0.0],
                    [ 0.2, -1.0,  0.0,  0.0,  0.0,  0.0],
                    [-0.5,  0.0, -1.2,  0.5, -0.5,  0.0],
                    [-0.4,  0.0,  0.5, -0.8,  0.0,  0.1],
                    [-0.6,  0.0, -0.5,  0.0, -1.2,  0.0],
                    [ 0.0,  0.0,  0.0,  0.1,  0.0,  0.3]])

        problem_args = {"method": "COLD", "uniform": False}

        run_args = {"N_steps": 20, "T": 1, "N_opt": 1, "CRAB": False, "bounds": (-3, 3)}

        result = solve_QUBO(Q, problem_args, run_args)

        print(result)

    ::

    {
        '111110': [0.9816, np.float64(-7.3999999999999995)],
        '111111': [0.0058, np.float64(-6.8999999999999995)],
        '111100': [0.0048, np.float64(-4.0)],
        '011110': [0.0032, np.float64(-4.2)],
        '110110': [0.0022, np.float64(-5.199999999999999)],
        '111010': [0.0012, np.float64(-6.8)],
        '001110': [0.0004, np.float64(-3.2)],
        '011100': [0.0002, np.float64(-2.0)],
        '110010': [0.0002, np.float64(-3.5999999999999996)],
        '011010': [0.0002, np.float64(-4.4)],
        '101110': [0.0002, np.float64(-6.8)]
    }

    """
    method = problem_args["method"]
    # Both methods accept the AGP type; 1st order is the default.
    agp_type = problem_args.get("agp_type", "order1")

    if method == "LCD":
        problem_operators = create_LCD_instance(Q, agp_type=agp_type, uniform_AGP_coeffs=problem_args["uniform"])

    elif method == "COLD":
        problem_operators = create_COLD_instance(Q, uniform_AGP_coeffs=problem_args["uniform"], agp_type=agp_type)

    else:
        raise ValueError(f"{method} is not a valid option as method. Valid options are 'LCD' and 'COLD'.")

    # Create qarg and problem instrance
    qarg = QuantumVariable(Q.shape[0])
    prob = DCQOProblem(*problem_operators)

    # Run problem
    result = prob.run(qarg, method=method, **run_args)

    return result
