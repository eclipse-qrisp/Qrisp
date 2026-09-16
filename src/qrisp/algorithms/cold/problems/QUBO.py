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


def create_COLD_instance(Q, uniform_AGP_coeffs):
    """Create the necessary parameters and operators to initialize a DCQO problem instance for COLD.

    Parameters
    ----------
    Q : np.array
        The QUBO Matrix to be encoded in the Hamiltonian.
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
    N = len(Q[0])
    h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
    J = 0.5 * Q

    def lam():
        t, T = sp.symbols("t T", real=True)
        lam_expr = sp.sin(sp.pi / 2 * sp.sin(sp.pi * t / (2 * T)) ** 2) ** 2
        return lam_expr

    # AGP coefficients
    def alpha(lam, f, f_deriv):
        return _order1_agp_coeffs(h, J, lam, f, f_deriv, uniform=uniform_AGP_coeffs)

    # Initial Hamiltonian
    H_init = 1 * sum([X(i) for i in range(N)])

    # Problem Hamiltonian
    H_prob = QubitOperator.sum(
        itertools.chain(
            (J[i][j] * Z(i) * Z(j) for i in range(N) for j in range(i, N)),
            (h[i] * Z(i) for i in range(N)),
        )
    )

    # AGP as function of alpha
    if uniform_AGP_coeffs:
        A_lam = sum([Y(i) for i in range(N)])
    else:
        A_lam = [Y(i) for i in range(N)]

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
            A_lam = [
                -2 * (h[i] * Y(i) + QubitOperator.sum(J[i][j] * (Y(i) * Z(j) + Z(i) * Y(j)) for j in range(i)))
                for i in range(N)
            ]
            return A_lam

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
            r"""Shared coefficient for the nested-commutator AGP.

            Minimises $S = \mathrm{Tr}[G_\lambda^2]$ for the single merged ansatz operator
            $A_\lambda = \sum_i A_i$ with
            $A_i = -2(h_i\sigma^y_i + \sum_{j<i} J_{ij}(\sigma^y_i\sigma^z_j + \sigma^z_i\sigma^y_j))$,
            against the Hamiltonian the circuit evolves (transverse term $+(1-\lambda)\sum_i\sigma^x_i$).

            The action is quadratic in the coefficient, and every trace evaluates in closed form.
            The numerator reproduces Eq. (S11) of `BF-DCQO <https://arxiv.org/abs/2405.13898>`_ for a
            uniform transverse field. All aggregates are $O(N^2)$, so this stays cheap where an
            explicit minimal-action solve would cost $O(N^4)$.
            """
            J_sq = np.asarray(J, dtype=float) ** 2
            np.fill_diagonal(J_sq, 0.0)
            h_sq = np.asarray(h, dtype=float) ** 2

            S_h2 = np.sum(h_sq)
            S_h4 = np.sum(h_sq**2)
            # sums over unordered pairs i < j
            S_2 = np.sum(J_sq) / 2
            S_4 = np.sum(J_sq**2) / 2
            # sum_{i<j} J_ij**2 (h_i**2 + h_j**2)
            S_hsqR = np.sum(J_sq * h_sq[None, :])
            # sum over unordered pairs of distinct edges sharing a site
            R_i = np.sum(J_sq, axis=1)
            S_adj = (np.sum(R_i**2) - 2 * S_4) / 2

            def alpha(lam):
                nom = S_h2 + 2 * S_2
                denom = 4 * ((S_h2 + 8 * S_2) * (1 - lam) ** 2 + lam**2 * (S_h4 + 2 * S_4 + 6 * S_hsqR + 6 * S_adj))
                return [nom / denom] * N

            return alpha

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
    """Solves a QUBO Matrix using counterdiabatic driving. This method uses the pre-defined COLD/LCD operators
    (hamiltonian, scheduling function, AGP parameters) as described in the tutorial.
    To define your own operators, create a DCQO instance and use the ``run`` method.

    Parameters
    ----------
    Q : np.array
        QUBO Matrix to solve.
    problem_args : dict
        Holds arguments for DCQO problem creation (``method``: str ("COLD"/"LCD"), ``uniform``: bool).
    run_args : dict
        Holds arguments for running the DCQO instance (``N_steps``, ``T``, ``N_opt``, ``CRAB``, ``objective``,``precision``, ``backend``, ``exp_value_backend``).
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

        Q = np.array([[-1.1, 0.6, 0.4, 0.0, 0.0, 0.0],
                    [0.6, -0.9,  0.5, 0.0, 0.0, 0.0],
                    [0.4, 0.5, -1.0, -0.6, 0.0, 0.0],
                    [0.0, 0.0, -0.6, -0.5, 0.6, 0.0],
                    [0.0, 0.0, 0.0, 0.6, -0.3, 0.5],
                    [0.0, 0.0, 0.0, 0.0, 0.5, -0.4]])

        problem_args = {"method": "COLD", "uniform": False}

        run_args = {"N_steps": 4, "T": 8, "N_opt": 1, "CRAB": False, "bounds": (-3, 3)}

        result = solve_QUBO(Q, problem_args, run_args)

        print(result)

    ::

        {'111111': [0.368, np.float64(-0.20000000000000018)], '111101': [0.1722, np.float64(-2.1000000000000005)], '111011': [0.158, np.float64(0.2999999999999999)], '110111': [0.0622, np.float64(0.19999999999999984)], '111100': [0.0448, np.float64(-1.7000000000000002)], '011111': [0.0372, np.float64(-1.1)], '110101': [0.0208, np.float64(-1.7000000000000002)], '011101': [0.0184, np.float64(-3.0)], '111110': [0.0166, np.float64(-0.8000000000000002)], '011011': [0.0164, np.float64(-0.6000000000000001)], '111001': [0.0154, np.float64(-0.40000000000000013)], '001111': [0.0078, np.float64(-1.2000000000000002)], '110011': [0.0062, np.float64(-0.5000000000000002)], '010111': [0.0062, np.float64(0.09999999999999998)], '100111': [0.006, np.float64(-0.10000000000000009)], '110100': [0.0054, np.float64(-1.3000000000000003)], '110001': [0.005, np.float64(-1.2000000000000002)], '100101': [0.0034, np.float64(-2.0)], '011100': [0.0032, np.float64(-2.6)], '001101': [0.0032, np.float64(-3.1)], '111010': [0.003, np.float64(-0.30000000000000004)], '110110': [0.0026, np.float64(-0.4000000000000002)], '001011': [0.0024, np.float64(-0.7000000000000001)], '111000': [0.0014, np.float64(-1.1102230246251565e-16)], '010001': [0.0012, np.float64(-1.3)], '010101': [0.0012, np.float64(-1.7999999999999998)], '010011': [0.0012, np.float64(-0.6000000000000001)], '000111': [0.0012, np.float64(1.0)], '011110': [0.001, np.float64(-1.7000000000000002)], '110000': [0.0008, np.float64(-0.8000000000000002)], '011001': [0.0008, np.float64(-1.3)], '101101': [0.0008, np.float64(-3.4)], '101111': [0.0008, np.float64(-1.5)], '100100': [0.0006, np.float64(-1.6)], '001100': [0.0006, np.float64(-2.7)], '000101': [0.0006, np.float64(-0.9)], '100011': [0.0006, np.float64(-0.8000000000000002)], '011000': [0.0004, np.float64(-0.9)], '000100': [0.0004, np.float64(-0.5)], '010100': [0.0004, np.float64(-1.4)], '101100': [0.0004, np.float64(-3.0)], '001110': [0.0004, np.float64(-1.8000000000000003)], '101011': [0.0004, np.float64(-1.0)], '100001': [0.0002, np.float64(-1.5)], '000011': [0.0002, np.float64(0.3)]}

    """
    method = problem_args["method"]

    if method == "LCD":
        # Check if AGP type is specified, otherwise use 1st order
        try:
            agp_type = problem_args["agp_type"]
        except KeyError:
            agp_type = "order1"

        problem_operators = create_LCD_instance(Q, agp_type=agp_type, uniform_AGP_coeffs=problem_args["uniform"])

    elif method == "COLD":
        problem_operators = create_COLD_instance(Q, uniform_AGP_coeffs=problem_args["uniform"])

    # Create qarg and problem instrance
    qarg = QuantumVariable(Q.shape[0])
    prob = DCQOProblem(*problem_operators)

    # Run problem
    result = prob.run(qarg, method=method, **run_args)

    return result
