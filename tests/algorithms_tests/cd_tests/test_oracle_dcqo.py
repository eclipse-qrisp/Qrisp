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

"""Tests for LCD with the nested-commutator AGP on diagonal costs given as phase oracles."""

import numpy as np
import pytest
import scipy.linalg as sl

from qrisp import QuantumVariable, as_hamiltonian
from qrisp.algorithms.cold.oracle_dcqo_problem import OracleDCQOProblem

N = 4
_rng = np.random.default_rng(7)
_V = _rng.uniform(0.3, 1.0, N)
_A = _rng.integers(1, 4, N)
_W = _rng.uniform(-0.3, 0.3, (N, N))


def cost_fn(B):
    """Non-polynomial toy cost: saturating value, pair terms and a ReLU penalty."""
    B = np.asarray(B, dtype=float)
    s = B @ _V
    return -(1 - 0.7**s) / 0.3 + np.einsum("mi,ij,mj->m", B, np.triu(_W, 1), B) + 0.8 * np.maximum(0, B @ _A - 4)


def make_oracle(fn):
    """Stand-in for an arithmetic phase oracle: exp(-i gamma fn) via as_hamiltonian."""

    @as_hamiltonian
    def phase(label):
        return fn(np.array([[int(c) for c in label]]))[0]

    def oracle(qarg, gamma):
        phase(qarg, t=-gamma)

    return oracle


def _dense(n, fn):
    """Dense H_X and diagonal of fn, basis index k = sum_i b_i 2^i."""
    idx = np.arange(1 << n)
    B = (idx[:, None] >> np.arange(n)) & 1
    HX = np.zeros((1 << n, 1 << n))
    for i in range(n):
        HX[idx ^ (1 << i), idx] += 1
    return B, HX, fn(B)


def _probs_from_result(res, n):
    p = np.zeros(1 << n)
    for label, (prob, _) in res.items():
        p[sum(int(c) << i for i, c in enumerate(label))] = prob
    return p


@pytest.mark.parametrize("mixer_strength", [1.0, 0.6])
def test_agp_coeff_matches_dense_traces(mixer_strength):
    """alpha from finite-difference averages equals Tr[O1^2]/Tr[O2^2] from dense matrices."""
    problem = OracleDCQOProblem(cost_fn, None, N, mixer_strength=mixer_strength)
    _, HX, C = _dense(N, cost_fn)
    Cm = np.diag(C)
    for lam in (0.1, 0.5, 0.9):
        H = (1 - lam) * mixer_strength * HX + lam * Cm
        dH = Cm - mixer_strength * HX
        O1 = H @ dH - dH @ H
        O2 = H @ O1 - O1 @ H
        expected = np.real(np.trace(O1 @ O1)) / np.real(np.trace(O2 @ O2))
        assert np.isclose(problem.agp_coeff(lam), expected)


@pytest.mark.parametrize("mixer_strength", [1.0, 0.6])
def test_per_qubit_agp_coeffs_match_dense_action_minimum(mixer_strength):
    """alpha_i from finite-difference averages minimize Tr[G^2] over A = sum_i alpha_i s i[X_i, C]."""
    problem = OracleDCQOProblem(cost_fn, None, N, mixer_strength=mixer_strength, uniform=False)
    _, HX, C = _dense(N, cost_fn)
    Cm = np.diag(C)
    idx = np.arange(1 << N)
    ops = []
    for i in range(N):
        Xi = np.zeros_like(HX)
        Xi[idx ^ (1 << i), idx] = 1
        ops.append(1j * mixer_strength * (Xi @ Cm - Cm @ Xi))
    for lam in (0.1, 0.5, 0.9):
        H = (1 - lam) * mixer_strength * HX + lam * Cm
        dH = Cm - mixer_strength * HX
        Cs = [o @ H - H @ o for o in ops]
        M = np.array([[np.real(np.trace(a @ b)) for b in Cs] for a in Cs])
        g = np.array([np.real(1j * np.trace(dH @ a)) for a in Cs])
        assert np.allclose(problem.agp_coeff(lam), np.linalg.solve(M, g))


def test_agp_coeff_monte_carlo_estimate():
    """Monte Carlo averages approximate the exact ones."""
    exact = OracleDCQOProblem(cost_fn, None, N)
    sampled = OracleDCQOProblem(cost_fn, None, N, n_samples=40000, seed=1)
    for lam in (0.2, 0.8):
        assert np.isclose(sampled.agp_coeff(lam), exact.agp_coeff(lam), rtol=0.05)


@pytest.mark.parametrize(
    ("uniform", "agp_order", "alternate"),
    [(True, 1, False), (True, 2, False), (True, 1, True), (False, 1, False), (False, 2, True)],
)
def test_lcd_circuit_matches_dense_simulation(uniform, agp_order, alternate):
    """The oracle-only circuit reproduces the dense product formula it is meant to implement."""
    N_steps, T, s_mix = 3, 1.0, 0.8
    problem = OracleDCQOProblem(cost_fn, make_oracle(cost_fn), N, mixer_strength=s_mix, uniform=uniform)
    res = problem.run(QuantumVariable(N), N_steps, T, agp_order=agp_order, alternate=alternate)
    p_circuit = _probs_from_result(res, N)

    B, HX, C = _dense(N, cost_fn)
    Z = 1 - 2 * B
    Xs = [np.zeros_like(HX) for _ in range(N)]
    idx = np.arange(1 << N)
    for i in range(N):
        Xs[i][idx ^ (1 << i), idx] = 1
    # g_i = Z_i Delta_i, Y_i = i X_i Z_i
    G = [(1j * Xs[i] @ np.diag(Z[:, i])) @ np.diag(Z[:, i] * (C[idx ^ (1 << i)] - C)) for i in range(N)]
    psi = np.prod(Z, axis=1) / np.sqrt(1 << N)
    dt = T / N_steps
    for k in range(N_steps):
        lam, lamdot = problem.lam[k], problem.lamdot[k]
        psi = sl.expm(-1j * dt * (1 - lam) * s_mix * HX) @ psi
        psi = np.exp(-1j * dt * lam * C) * psi
        theta = np.broadcast_to(dt * lamdot * problem.agp_coeff(lam) * s_mix / agp_order, (N,))
        order = list(reversed(range(N))) if alternate and k % 2 == 1 else list(range(N))
        sweep = order if agp_order == 1 else order + order[::-1]
        for i in sweep:
            psi = sl.expm(1j * theta[i] * G[i]) @ psi

    assert np.allclose(p_circuit, np.abs(psi) ** 2, atol=1e-4)


def test_local_phase_oracle_gives_same_result():
    """Restricting the AGP oracle calls to the terms that depend on bit i changes nothing."""

    def local_oracle(qarg, gamma, i):
        def C_i(B):
            B0 = np.array(B, copy=True)
            B0[:, i] = 0
            return cost_fn(B) - cost_fn(B0)

        make_oracle(C_i)(qarg, gamma)

    full = OracleDCQOProblem(cost_fn, make_oracle(cost_fn), N)
    local = OracleDCQOProblem(cost_fn, make_oracle(cost_fn), N, local_phase_oracle=local_oracle)
    p_full = _probs_from_result(full.run(QuantumVariable(N), 3, 1.0), N)
    p_local = _probs_from_result(local.run(QuantumVariable(N), 3, 1.0), N)
    assert np.allclose(p_full, p_local, atol=1e-4)


def test_agp_lowers_expected_cost_at_short_time():
    """At short evolution time the AGP beats plain digitized annealing."""
    problem = OracleDCQOProblem(cost_fn, make_oracle(cost_fn), N)
    with_agp = problem.evaluate(problem.run(QuantumVariable(N), 10, 1.0))
    without = problem.evaluate(problem.run(QuantumVariable(N), 10, 1.0, agp=False))
    assert with_agp["expected_cost"] < without["expected_cost"]


def test_evaluate():
    """evaluate() reports expectation, optimum probability and feasibility from the cost function."""

    def feasible_fn(B):
        return np.asarray(B) @ _A <= 4

    problem = OracleDCQOProblem(cost_fn, None, N, feasible_fn=feasible_fn)
    c_min, opt = problem.exact_minimum()
    B, _, C = _dense(N, cost_fn)
    worst = "".join(map(str, B[np.argmax(C)]))
    res = {opt[0]: [0.25, c_min], worst: [0.75, C.max()]}

    out = problem.evaluate(res)
    assert np.isclose(out["expected_cost"], 0.25 * c_min + 0.75 * C.max())
    assert np.isclose(out["p_optimal"], 0.25)
    assert out["best"] == (opt[0], c_min)
    feas_opt, feas_worst = feasible_fn(np.array([[int(c) for c in label] for label in (opt[0], worst)]))
    assert np.isclose(out["p_feasible"], 0.25 * feas_opt + 0.75 * feas_worst)


# ----- control pulse (COLD), local AGP and the numpy model -----


def _dense_ops(n):
    idx = np.arange(1 << n)
    B = (idx[:, None] >> np.arange(n)) & 1
    Zs = 1 - 2 * B
    Xs = []
    for i in range(n):
        Xi = np.zeros((1 << n, 1 << n))
        Xi[idx ^ (1 << i), idx] = 1
        Xs.append(Xi)
    Ys = [1j * Xs[i] @ np.diag(Zs[:, i]) for i in range(n)]
    return Xs, Ys, np.diag(Zs.sum(axis=1)).astype(float)


def _dense_action_minimum(ops, H, dH, uniform):
    """Coefficients of A = sum_k c_k ops[k] (or c sum_k ops[k]) minimizing Tr[(dH + i[A, H])^2]."""
    if uniform:
        ops = [sum(ops)]
    Cs = [1j * (o @ H - H @ o) for o in ops]
    M = np.array([[np.real(np.trace(a @ b)) for b in Cs] for a in Cs])
    g = np.array([-np.real(np.trace(dH @ a)) for a in Cs])
    c = np.linalg.solve(M, g)
    return c[0] if uniform else c


@pytest.mark.parametrize("agp_type", ["nc", "local"])
@pytest.mark.parametrize("uniform", [True, False])
@pytest.mark.parametrize(
    ("lam", "s_mix", "f", "fp"), [(0.3, 1.0, 0.0, 0.0), (0.6, 0.7, 0.4, -1.1), (0.2, 1.3, -0.8, 2.0)]
)
def test_agp_coeff_with_control_matches_dense_action_minimum(agp_type, uniform, lam, s_mix, f, fp):  # noqa: PLR0913, PLR0917
    """Coefficients with a control pulse f sum Z minimize the action for both ansatz types."""
    problem = OracleDCQOProblem(cost_fn, None, N, mixer_strength=s_mix, uniform=uniform, agp_type=agp_type)
    _, HX, C = _dense(N, cost_fn)
    Xs, Ys, HZ = _dense_ops(N)
    H = (1 - lam) * s_mix * HX + lam * np.diag(C) + f * HZ
    dH = -s_mix * HX + np.diag(C) + fp * HZ
    if agp_type == "local":
        ops = Ys
    else:
        Cm = np.diag(C) + (f + (1 - lam) * fp) * HZ
        ops = [1j * s_mix * (Xi @ Cm - Cm @ Xi) for Xi in Xs]
    assert np.allclose(problem.agp_coeff(lam, f, fp), _dense_action_minimum(ops, H, dH, uniform))


def test_agp_coeffs_match_qubo_formulas():
    """For a QUBO, the local coefficients equal create_COLD_instance, the uniform nc ones up to the template sign."""
    from qrisp.algorithms.cold.problems.QUBO import _local_agp_coeffs, _nc_uniform_agp_coeffs

    Q = np.array([[-1.2, 0.4, 0.0, 0.0], [0.4, 0.3, 0.2, 0.0], [0.0, 0.2, -1.1, 0.3], [0.0, 0.0, 0.3, -0.8]])
    h, J = -0.5 * Q.sum(axis=1), 0.5 * Q

    def qubo(B):
        B = np.asarray(B, dtype=float)
        return np.einsum("mi,ij,mj->m", B, Q, B)

    local = OracleDCQOProblem(qubo, None, 4, uniform=False, agp_type="local")
    nc = OracleDCQOProblem(qubo, None, 4)
    nc_qubo = _nc_uniform_agp_coeffs(h, J)
    for lam, f, fp in ((0.3, 0.0, 0.0), (0.6, 0.4, -1.1)):
        assert np.allclose(local.agp_coeff(lam, f, fp), _local_agp_coeffs(h, J, lam, f, fp, uniform=False))
        # Qrisp's nested-commutator template is -i[H_X, C]; with f = 0 both apply the same operator
        assert np.isclose(nc.agp_coeff(lam), -nc_qubo(lam)[0])


@pytest.mark.parametrize(
    ("agp_type", "uniform", "beta", "CRAB"),
    [
        ("nc", True, None, False),
        ("local", False, None, False),
        ("nc", False, [0.3, -0.5], False),
        ("nc", True, [0.4, 0.2], True),
        ("local", True, [0.3, -0.5], False),
    ],
)
def test_model_matches_circuit(agp_type, uniform, beta, CRAB):
    """model_state reproduces the distribution of the Qrisp circuit, for LCD and COLD."""
    N_steps, T = 3, 1.0
    problem = OracleDCQOProblem(
        cost_fn, make_oracle(cost_fn), N, mixer_strength=0.8, uniform=uniform, agp_type=agp_type, seed=3
    )
    qarg = QuantumVariable(N)
    if beta is None:
        problem.apply_lcd_hamiltonian(qarg, N_steps, T, alternate=True)
        psi = problem.model_state(N_steps, T, alternate=True)
    else:
        problem.apply_cold_hamiltonian(qarg, N_steps, T, beta, CRAB=CRAB)
        psi = problem.model_state(N_steps, T, beta, CRAB=CRAB)
    meas = qarg.get_measurement()
    p_circuit = np.zeros(1 << N)
    for label, prob in meas.items():
        p_circuit[sum(int(c) << i for i, c in enumerate(label))] = prob
    assert np.allclose(p_circuit, np.abs(psi) ** 2, atol=1e-4)


def test_local_agp_needs_no_oracle_calls():
    """The local AGP calls the oracle once per step, the nested commutator 1 + 2n times."""
    calls = []
    oracle = make_oracle(cost_fn)

    def counting_oracle(qarg, gamma):
        calls.append(gamma)
        oracle(qarg, gamma)

    for agp_type, expected in (("local", 3), ("nc", 3 * (1 + 2 * N))):
        calls.clear()
        problem = OracleDCQOProblem(cost_fn, counting_oracle, N, agp_type=agp_type)
        problem.apply_cold_hamiltonian(QuantumVariable(N), 3, 1.0, [0.2])
        assert len(calls) == expected


@pytest.mark.parametrize("agp_type", ["nc", "local"])
def test_cold_lowers_expected_cost(agp_type):
    """The optimized pulse lowers the expected cost below LCD (beta = 0), and run() applies it."""
    problem = OracleDCQOProblem(cost_fn, make_oracle(cost_fn), N, agp_type=agp_type, seed=0)
    lcd = problem.model_expectation(5, 1.0)
    res = problem.run(QuantumVariable(N), 5, 1.0, method="COLD", N_opt=2, n_restarts=2)
    assert problem.opt_value < lcd
    assert np.isclose(problem.evaluate(res)["expected_cost"], problem.opt_value, atol=1e-3)


def test_invalid_agp_type():
    """Unknown agp_type values are rejected."""
    with pytest.raises(ValueError, match="agp_type"):
        OracleDCQOProblem(cost_fn, None, N, agp_type="order2")
