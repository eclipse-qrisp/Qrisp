import time

import numpy as np
import sympy as sp

from qrisp import QuantumVariable
from qrisp import h as had_gate
from qrisp import z as z_gate
from qrisp.algorithms.cold import DCQOProblem, solve_QUBO
from qrisp.interface.provider_backends.qiskit_backend import QiskitBackend
from qrisp.operators.qubit import X, Y, Z


def test_cold_uniform_magnitude():

    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    solution = "1011"

    problem_args = {"method": "COLD", "uniform": True}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, "objective": "agp_coeff_magnitude", "CRAB": False}

    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_cold_nonuniform_magnitude():

    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    solution = "1011"

    problem_args = {"method": "COLD", "uniform": False}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, "objective": "agp_coeff_magnitude", "CRAB": False}

    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_cold_uniform_cost():

    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    solution = "1011"

    problem_args = {"method": "COLD", "uniform": True}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, "objective": "exp_value", "CRAB": False}

    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_cold_nonuniform_cost():

    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    solution = "1011"

    problem_args = {"method": "COLD", "uniform": False}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, "objective": "exp_value", "CRAB": False}

    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_coldcrab_uniform_cost():

    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    np.random.seed(42)  # Deterministic for reproducible test results
    solution = "1011"

    problem_args = {"method": "COLD", "uniform": True}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, "objective": "exp_value", "CRAB": True}

    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_coldcrab_uniform_magnitude():

    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    solution = "1011"

    problem_args = {"method": "COLD", "uniform": True}
    run_args = {"N_steps": 50, "T": 10, "N_opt": 1, "objective": "agp_coeff_magnitude", "CRAB": True}

    res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)

    assert solution in list(res.keys())[0:5]


def test_cold_expvalue_method_backend():

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


def test_cold_expvalue_fast_path_matches_hprob():
    # optimization_routine's exp_value objective has a fast statevector path that
    # estimates <H_prob> via a precomputed cost table. That table must be derived from
    # H_prob's own diagonal terms; deriving it from Q via a hand-rolled formula instead
    # (as in a past regression) computes a different function of the bitstring, since
    # H_prob is caller-supplied and not guaranteed to relate to Q via any fixed
    # convention. N_steps/maxiter are kept minimal since only the objective function's
    # correctness at one point is under test, not the quality of the optimization.
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


def test_cold_no_exponential_precompute_for_non_expvalue_objective():
    # optimization_routine's exp_value fast path needs a 2**n_qubits cost table, but a
    # past regression built it unconditionally regardless of which objective was
    # selected. This must stay gated: at n_qubits=24, an accidental 2**24-entry table
    # would take orders of magnitude longer than the classical-only work
    # "agp_coeff_magnitude" actually needs, so a generous wall-clock ceiling here
    # directly catches a regression of that gate without ever materializing such a
    # table itself (which would also cost real time/memory in a CI run).
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
    # create_COLD_instance's smooth scheduling function has zero derivative at the
    # domain endpoints. g_deriv = 1/lamdot must not be evaluated exactly at those
    # endpoints (a past regression's right-endpoint time grid did, producing a
    # ~1e32 value that made the COLD optimization landscape chaotic). Sampling closer
    # to a genuine zero-derivative point inevitably makes g_deriv larger -- with the
    # midpoint rule it grows as N_steps**3 (~1.3e9 at N_steps=1000), which is still
    # six orders of magnitude below where float64 angle precision actually breaks
    # down (~1e15); 1e10 comfortably separates that expected, benign growth from the
    # ~1e32 signature of hitting the singularity exactly. This is a pure classical
    # check on _precompute_timegrid's output, independent of n_qubits and of any
    # quantum simulation, so it stays fast regardless of scale.
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
