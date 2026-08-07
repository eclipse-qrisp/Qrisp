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

import time

import numpy as np
import pytest
import sympy as sp

from qrisp import QuantumVariable
from qrisp import h as had_gate
from qrisp import z as z_gate
from qrisp.algorithms.cold import DCQOProblem, solve_QUBO
from qrisp.interface.provider_backends.qiskit_backend import QiskitBackend
from qrisp.operators.qubit import QubitOperator, X, Y, Z
from qrisp.operators.qubit.qubit_term import QubitTerm


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

        alph = [nom[i] / denom[i] for i in range(N)]  # codespell:ignore

        return alph  # codespell:ignore

    def lam():
        t, T = sp.symbols("t T", real=True)
        lam_expr = t / T
        return lam_expr

    cold_problem = DCQOProblem(Q, H_init, H_prob, A_lam, alpha, lam, H_control)

    qarg = QuantumVariable(N)
    result = cold_problem.run(qarg, N_steps=4, T=8, method="COLD", N_opt=1, bounds=(-3, 3))

    assert isinstance(result, dict)
    assert len(result) > 0


def test_cold_expvalue_fast_path_matches_hprob():
    # The exp_value fast path's cost table must be derived from H_prob's own diagonal
    # terms, not hand-rolled from Q (a past regression did, computing a different
    # function). N_steps/maxiter stay minimal: only the objective's correctness matters.
    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])
    N = Q.shape[0]
    h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
    J = 0.5 * Q

    H_init = 1 * sum([X(i) for i in range(N)])
    H_prob = sum([sum([J[i][j] * Z(i) * Z(j) for j in range(i)]) for i in range(N)]) + sum(
        [h[i] * Z(i) for i in range(N)]
    )
    H_control = sum([Z(i) for i in range(N)])
    A_lam = sum([Y(i) for i in range(N)])

    def alpha(lam, f, f_deriv):
        A = lam * h + f
        B = 1 - lam
        C = h + f_deriv
        nom = np.sum(A + 4 * B * C)
        denom = 2 * (np.sum(A**2) + N * (B**2)) + 4 * (lam**2) * np.sum(np.tril(J, -1).sum(axis=1))
        return [nom / denom] * N

    def lam():
        t, T = sp.symbols("t T", real=True)
        return t / T

    def qarg_prep(q):
        had_gate(q)
        z_gate(q)
        return q

    problem = DCQOProblem(Q, H_init, H_prob, A_lam, alpha, lam, H_control=H_control, qarg_prep=qarg_prep)

    qarg1 = QuantumVariable(N)
    qc = problem.compile_U_cold(qarg1, N_opt=1, N_steps=2, T=4, CRAB=False)

    qarg2 = QuantumVariable(N)
    opt_params, cost = problem.optimization_routine(
        qarg2,
        N_opt=1,
        N_steps=2,
        T=4,
        qc=qc,
        CRAB=False,
        optimizer="Nelder-Mead",
        options={"maxiter": 1, "maxfev": 1},
        objective="exp_value",
        bounds=(-2, 2),
    )

    subs_dic = {sp.Symbol("par_0"): float(opt_params[0])}
    qarg3 = QuantumVariable(N)
    ground_truth = H_prob.expectation_value(
        qarg3, compile=False, subs_dic=subs_dic, precompiled_qc=qc, precision=0.01
    )()

    assert abs(cost - ground_truth) < 0.1


def test_cold_expvalue_fast_path_handles_projectors():
    # The fast path's per-term eigenvalue must account for each factor's actual type
    # (Z, P0, P1), not treat every factor as Z -- a P0/P1 term evaluated as Z gives a
    # different (wrong) value at every basis state.
    N = 3
    H_prob = 2.0 * Z(0) * Z(1) + 3.0 * QubitOperator({QubitTerm({1: "P0"}): 1.0})
    H_prob += 1.5 * QubitOperator({QubitTerm({2: "P1"}): 1.0})
    H_init = sum([X(i) for i in range(N)])
    A_lam = sum([Y(i) for i in range(N)])
    H_control = sum([Z(i) for i in range(N)])

    def alpha(lam, f, f_deriv):
        return [0.0] * N

    def lam():
        t, T = sp.symbols("t T", real=True)
        return t / T

    def qarg_prep(q):
        had_gate(q)
        z_gate(q)
        return q

    problem = DCQOProblem(np.eye(N), H_init, H_prob, A_lam, alpha, lam, H_control=H_control, qarg_prep=qarg_prep)

    qarg1 = QuantumVariable(N)
    qc = problem.compile_U_cold(qarg1, N_opt=1, N_steps=2, T=4, CRAB=False)

    qarg2 = QuantumVariable(N)
    opt_params, cost = problem.optimization_routine(
        qarg2,
        N_opt=1,
        N_steps=2,
        T=4,
        qc=qc,
        CRAB=False,
        optimizer="Nelder-Mead",
        options={"maxiter": 1, "maxfev": 1},
        objective="exp_value",
        bounds=(-2, 2),
    )

    subs_dic = {sp.Symbol("par_0"): float(opt_params[0])}
    qarg3 = QuantumVariable(N)
    ground_truth = H_prob.expectation_value(
        qarg3, compile=False, subs_dic=subs_dic, precompiled_qc=qc, precision=0.01
    )()

    assert abs(cost - ground_truth) < 0.1


def test_cold_expvalue_falls_back_for_nondiagonal_hprob():
    # H_prob with a non-diagonal factor (X here) has no well-defined per-basis-state
    # eigenvalue, so the fast path must disable itself and fall back to
    # expectation_value() instead of silently treating X as Z.
    N = 3
    H_prob = 2.0 * Z(0) * Z(1) + 1.5 * X(1)
    H_init = sum([X(i) for i in range(N)])
    A_lam = sum([Y(i) for i in range(N)])
    H_control = sum([Z(i) for i in range(N)])

    def alpha(lam, f, f_deriv):
        return [0.0] * N

    def lam():
        t, T = sp.symbols("t T", real=True)
        return t / T

    def qarg_prep(q):
        had_gate(q)
        z_gate(q)
        return q

    problem = DCQOProblem(np.eye(N), H_init, H_prob, A_lam, alpha, lam, H_control=H_control, qarg_prep=qarg_prep)

    qarg1 = QuantumVariable(N)
    qc = problem.compile_U_cold(qarg1, N_opt=1, N_steps=2, T=4, CRAB=False)

    qarg2 = QuantumVariable(N)
    opt_params, cost = problem.optimization_routine(
        qarg2,
        N_opt=1,
        N_steps=2,
        T=4,
        qc=qc,
        CRAB=False,
        optimizer="Nelder-Mead",
        options={"maxiter": 1, "maxfev": 1},
        objective="exp_value",
        bounds=(-2, 2),
    )

    subs_dic = {sp.Symbol("par_0"): float(opt_params[0])}
    qarg3 = QuantumVariable(N)
    ground_truth = H_prob.expectation_value(
        qarg3, compile=False, subs_dic=subs_dic, precompiled_qc=qc, precision=0.01
    )()

    assert abs(cost - ground_truth) < 0.1


def test_cold_no_exponential_precompute_for_non_expvalue_objective():
    # A past regression built the exp_value fast path's 2**n_qubits cost table
    # unconditionally. A wall-clock ceiling at n_qubits=24 catches that regression
    # without ever materializing such a table itself.
    N = 24
    Q = np.eye(N)
    h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)

    H_init = 1 * sum([X(i) for i in range(N)])
    H_prob = sum([h[i] * Z(i) for i in range(N)])
    H_control = sum([Z(i) for i in range(N)])
    A_lam = sum([Y(i) for i in range(N)])

    def alpha(lam, f, f_deriv):
        return [0.0] * N

    def lam():
        t, T = sp.symbols("t T", real=True)
        return t / T

    problem = DCQOProblem(Q, H_init, H_prob, A_lam, alpha, lam, H_control=H_control)
    problem._precompute_timegrid(N_steps=5, T=8, method="COLD")

    qarg = QuantumVariable(N)

    t0 = time.perf_counter()
    problem.optimization_routine(
        qarg,
        N_opt=1,
        N_steps=5,
        T=8,
        qc=None,
        CRAB=False,
        optimizer="Nelder-Mead",
        options={"maxiter": 3},
        objective="agp_coeff_magnitude",
        bounds=(-2, 2),
    )
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0


def test_cold_g_deriv_stays_finite_for_smooth_schedule():
    # This schedule's derivative vanishes at the domain endpoints; a past regression's
    # time grid sampled g_deriv = 1/lamdot exactly there, blowing up to ~1e32. 1e10
    # separates that signature from g_deriv's expected, benign N_steps**3 growth.
    def lam():
        t, T = sp.symbols("t T", real=True)
        return sp.sin(sp.pi / 2 * sp.sin(sp.pi * t / (2 * T)) ** 2) ** 2

    N = 2
    Q = np.eye(N)
    h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
    H_init = sum([X(i) for i in range(N)])
    H_prob = sum([h[i] * Z(i) for i in range(N)])
    A_lam = sum([Y(i) for i in range(N)])
    H_control = sum([Z(i) for i in range(N)])

    def alpha(lam, f, f_deriv):
        return [0.0] * N

    problem = DCQOProblem(Q, H_init, H_prob, A_lam, alpha, lam, H_control=H_control)
    problem._precompute_timegrid(N_steps=50, T=10, method="COLD")

    assert np.all(np.isfinite(problem.g_deriv))
    assert np.max(np.abs(problem.g_deriv)) < 1e10


@pytest.mark.parametrize("n_opt", [None, 0, -1, 1.5, True, "1"])
def test_cold_rejects_invalid_n_opt(n_opt):
    # method="COLD" needs N_opt >= 1; invalid values used to fail deep in range()/scipy
    # with confusing errors instead of explaining the constraint (issue #877).
    # Non-integer numerics and bools must be rejected too, not silently misused.
    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])
    problem_args = {"method": "COLD", "uniform": True}
    run_args = {"N_steps": 10, "T": 5, "CRAB": False, "objective": "exp_value", "bounds": (-2, 2), "N_opt": n_opt}

    with pytest.raises(ValueError, match="N_opt must be a positive integer"):
        solve_QUBO(Q, problem_args, run_args)
