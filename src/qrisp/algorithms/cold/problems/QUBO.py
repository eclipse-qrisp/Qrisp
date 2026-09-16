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
    if uniform_AGP_coeffs:

        def alpha(lam, f, f_deriv):
            A = lam * h + f
            B = 1 - lam
            C = h + f_deriv

            nom = np.sum(A + 4 * B * C)
            denom = 2 * (np.sum(A**2) + N * (B**2)) + 4 * (lam**2) * np.sum(np.tril(J, -1).sum(axis=1))
            alph = nom / denom
            alph = [alph] * N

            return alph

    else:

        def alpha(lam, f, f_deriv):
            nom = [h[i] + f + (1 - lam) * f_deriv for i in range(N)]
            denom = [
                2 * ((lam * h[i] + f) ** 2 + (1 - lam) ** 2 + lam**2 * sum([J[i][j] for j in range(N) if j != i]))
                for i in range(N)
            ]

            alph = [nom[i] / denom[i] for i in range(N)]
            return alph

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
            def alpha(lam):
                A = lam * h
                B = 1 - lam
                nom = np.sum(A + 4 * B * h)
                denom = 2 * (np.sum(A**2) + N * (B**2)) + 4 * (lam**2) * np.sum(np.tril(J, -1).sum(axis=1))
                alph = nom / denom
                alph = [alph] * N
                return alph

            return alpha

        def order1_nonuniform(J, h):
            def alpha(lam):
                denom = [
                    2 * ((lam * h[i]) ** 2 + (1 - lam) ** 2 + lam**2 * sum([J[i][j] for j in range(N) if j != i]))
                    for i in range(N)
                ]
                alph = [h[i] / denom[i] for i in range(N)]
                return alph

            return alpha

        def nc_uniform(J, h):
            def alpha(lam):
                S_hR = sum([sum([J[i][j] ** 2 * (h[i] + h[j]) for i in range(j)]) for j in range(N)])
                S_hsqR = sum([sum([J[i][j] ** 2 * (h[i] ** 2 + h[j] ** 2) for i in range(j)]) for j in range(N)])
                S_2 = sum([sum([J[i][j] ** 2 for i in range(j)]) for j in range(N)])
                S_4 = sum([sum([J[i][j] ** 4 for i in range(j)]) for j in range(N)])
                R_i_list = [sum([J[i][j] ** 2 if j != i else 0 for j in range(N)]) for i in range(N)]
                S_Rsq = sum(R_i**2 for R_i in R_i_list)
                S_h = sum(h)
                S_hsq = sum(i**2 for i in h)

                nom = S_h + 2 * S_2
                denom = 4 * (
                    lam**2 * (S_hsq + 2 * S_hsqR + 6 * S_hR + 2 * S_Rsq + 4 * S_2 - 2 * S_4)
                    + (1 - lam) ** 2 * (N + 8 * S_2)
                )

                alph = -nom / denom
                alph = [alph] * N
                return alph

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

        {'101101': [0.116, np.float64(-3.4)], '101110': [0.106, np.float64(-2.1)], '011101': [0.073, np.float64(-3.0)], '011110': [0.0666, np.float64(-1.7000000000000002)], '110010': [0.0634, np.float64(-1.1)], '101100': [0.0502, np.float64(-3.0)], '101010': [0.0402, np.float64(-1.6)], '011100': [0.0344, np.float64(-2.6)], '111101': [0.0334, np.float64(-2.1000000000000005)], '011010': [0.0264, np.float64(-1.2000000000000002)], '100010': [0.0254, np.float64(-1.4000000000000001)], '110101': [0.025, np.float64(-1.7000000000000002)], '101111': [0.025, np.float64(-1.5)], '100101': [0.0208, np.float64(-2.0)], '010010': [0.0202, np.float64(-1.2)], '110011': [0.0178, np.float64(-0.5000000000000002)], '111100': [0.0164, np.float64(-1.7000000000000002)], '111110': [0.0164, np.float64(-0.8000000000000002)], '110100': [0.0156, np.float64(-1.3000000000000003)], '011111': [0.0154, np.float64(-1.1)], '101011': [0.015, np.float64(-1.0)], '001110': [0.0134, np.float64(-1.8000000000000003)], '111010': [0.013, np.float64(-0.30000000000000004)], '110001': [0.011, np.float64(-1.2000000000000002)], '011011': [0.011, np.float64(-0.6000000000000001)], '010101': [0.0106, np.float64(-1.7999999999999998)], '100001': [0.0102, np.float64(-1.5)], '001101': [0.0094, np.float64(-3.1)], '100100': [0.0092, np.float64(-1.6)], '100011': [0.0074, np.float64(-0.8000000000000002)], '010001': [0.007, np.float64(-1.3)], '101001': [0.0062, np.float64(-1.7000000000000002)], '010100': [0.006, np.float64(-1.4)], '101000': [0.0056, np.float64(-1.3)], '010011': [0.0046, np.float64(-0.6000000000000001)], '111011': [0.0046, np.float64(0.2999999999999999)], '001100': [0.004, np.float64(-2.7)], '000110': [0.004, np.float64(0.39999999999999997)], '100110': [0.004, np.float64(-0.7000000000000001)], '000101': [0.0036, np.float64(-0.9)], '110000': [0.0034, np.float64(-0.8000000000000002)], '011000': [0.0032, np.float64(-0.9)], '111111': [0.0032, np.float64(-0.20000000000000018)], '011001': [0.003, np.float64(-1.3)], '001111': [0.0022, np.float64(-1.2000000000000002)], '000010': [0.002, np.float64(-0.3)], '110111': [0.002, np.float64(0.19999999999999984)], '100000': [0.0016, np.float64(-1.1)], '010000': [0.0016, np.float64(-0.9)], '001000': [0.0016, np.float64(-1.0)], '000100': [0.0016, np.float64(-0.5)], '110110': [0.0014, np.float64(-0.4000000000000002)], '001001': [0.0014, np.float64(-1.4)], '001011': [0.0012, np.float64(-0.7000000000000001)], '000111': [0.0008, np.float64(1.0)], '001010': [0.0006, np.float64(-1.3)], '010110': [0.0006, np.float64(-0.5)], '000000': [0.0004, np.float64(0.0)], '000011': [0.0004, np.float64(0.3)], '111000': [0.0002, np.float64(-1.1102230246251565e-16)], '000001': [0.0002, np.float64(-0.4)]}

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
