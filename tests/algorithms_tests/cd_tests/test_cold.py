import numpy as np
from qrisp.algorithms.cold import solve_QUBO
from qrisp.interface.provider_backends.qiskit_backend import QiskitBackend
import sympy as sp
from qrisp import QuantumVariable
from qrisp.operators.qubit import X, Y, Z
from qrisp.algorithms.cold import DCQOProblem


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
