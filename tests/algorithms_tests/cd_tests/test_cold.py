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

import numpy as np
import sympy as sp

from qrisp import QuantumVariable
from qrisp.algorithms.cold import DCQOProblem, solve_QUBO
from qrisp.interface.provider_backends.qiskit_backend import QiskitBackend
from qrisp.operators.qubit import X, Y, Z


def test_cold_uniform_magnitude():
    """COLD with uniform AGP coefficients, magnitude objective, finds the known solution."""

    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    solution = "1011"

    problem_args = {"method": "COLD", "uniform": True}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, "objective": "agp_coeff_magnitude", "CRAB": False}

    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_cold_nonuniform_magnitude():
    """COLD with non-uniform AGP coefficients, magnitude objective, finds the known solution."""

    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    solution = "1011"

    problem_args = {"method": "COLD", "uniform": False}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, "objective": "agp_coeff_magnitude", "CRAB": False}

    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_cold_uniform_cost():
    """COLD with uniform AGP coefficients, expectation-value objective, finds the known solution."""

    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    solution = "1011"

    problem_args = {"method": "COLD", "uniform": True}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, "objective": "exp_value", "CRAB": False}

    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_cold_nonuniform_cost():
    """COLD with non-uniform AGP coefficients, expectation-value objective, finds the known solution."""

    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    solution = "1011"

    problem_args = {"method": "COLD", "uniform": False}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, "objective": "exp_value", "CRAB": False}

    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_coldcrab_uniform_cost():
    """COLD with CRAB-randomized pulses, uniform AGP, expectation-value objective, finds the known solution."""

    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    np.random.seed(42)  # Deterministic for reproducible test results
    solution = "1011"

    problem_args = {"method": "COLD", "uniform": True}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, "objective": "exp_value", "CRAB": True}

    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_coldcrab_uniform_magnitude():
    """COLD with CRAB-randomized pulses, uniform AGP, magnitude objective, finds the known solution."""

    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    solution = "1011"

    problem_args = {"method": "COLD", "uniform": True}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, "objective": "agp_coeff_magnitude", "CRAB": True}

    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_cold_expvalue_method_backend():
    """COLD's expectation-value objective runs against an explicit measurement backend, not just the default statevector path."""

    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])
    problem_args = {"method": "COLD", "uniform": True}  # , "agp_type": "order1"}
    backend = QiskitBackend()
    run_args = {
        "N_steps": 4,
        "T": 1.0,
        "objective": "exp_value",
        "N_opt": 1,
        "precision": 0.1,
        "exp_value_backend": backend,
    }
    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)
    assert isinstance(res, dict)
    assert len(res) > 0


def test_cold_full_example():
    """End-to-end COLD run built directly via DCQOProblem (not solve_QUBO's factory helpers)."""

    Q = np.array(
        [
            [-1.1, 0.6, 0.4, 0.0, 0.0, 0.0],
            [0.6, -0.9, 0.5, 0.0, 0.0, 0.0],
            [0.4, 0.5, -1.0, -0.6, 0.0, 0.0],
            [0.0, 0.0, -0.6, -0.5, 0.6, 0.0],
            [0.0, 0.0, 0.0, 0.6, -0.3, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.5, -0.4],
        ]
    )

    N = Q.shape[0]
    h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
    J = 0.5 * Q

    H_init = 1 * sum([X(i) for i in range(N)])

    H_prob = sum([sum([J[i][j] * Z(i) * Z(j) for j in range(i)]) for i in range(N)]) + sum(
        [h[i] * Z(i) for i in range(N)]
    )

    H_control = sum([Z(i) for i in range(N)])

    A_lam = [Y(i) for i in range(N)]  # non-uniform

    def alpha(lam, f, f_deriv):

        nom = [h[i] + f + (1 - lam) * f_deriv for i in range(N)]

        denom = [
            2 * ((lam * h[i] + f) ** 2 + (1 - lam) ** 2 + lam**2 * sum([J[i][j] for j in range(N) if j != i]))
            for i in range(N)
        ]

        alph = [nom[i] / denom[i] for i in range(N)]

        return alph

    def lam():
        t, T = sp.symbols("t T", real=True)
        lam_expr = t / T
        return lam_expr

    cold_problem = DCQOProblem(Q, H_init, H_prob, A_lam, alpha, lam, H_control)

    qarg = QuantumVariable(N)
    result = cold_problem.run(qarg, N_steps=4, T=8, method="COLD", N_opt=1, bounds=(-3, 3))

    assert isinstance(result, dict)
    assert len(result) > 0
