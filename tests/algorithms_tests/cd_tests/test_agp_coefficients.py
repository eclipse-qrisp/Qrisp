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

"""Pins the AGP coefficient formulas against an explicit minimal-action calculation.

The coefficients in ``problems/QUBO.py`` are closed forms, and ``AGP_params`` builds and
solves the linear system symbolically over Pauli strings. Both are supposed to minimise the
action

    S(A) = Tr[G**2],    G = d_lambda H + i [A, H]

for the Hamiltonian ``DCQOProblem`` actually evolves. This module recomputes that minimisation
from explicit Pauli matrices -- slow, but transparent and independent of the implementations --
and compares. A closed form that drifts from the variational solution still produces plausible
looking distributions, so comparing against the action itself is what catches it.
"""

import itertools

import numpy as np
import pytest

from qrisp.algorithms.cold.AGP_params import solve_alpha
from qrisp.algorithms.cold.problems.QUBO import create_COLD_instance, create_LCD_instance

# the driven run must clear this, and beat the undriven baseline by a wide margin
MIN_DRIVEN_PROBABILITY = 0.5
PROBABILITY_TOLERANCE = 1e-6

I2 = np.eye(2)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]])
PZ = np.diag([1, -1]).astype(complex)


def _op(P, i, N):
    """Embed the single-qubit operator ``P`` on qubit ``i`` of an ``N`` qubit register."""
    out = np.array([[1]], dtype=complex)
    for k in range(N):
        out = np.kron(out, P if k == i else I2)
    return out


def _op2(P, i, R, j, N):
    """Embed ``P`` on qubit ``i`` and ``R`` on qubit ``j`` of an ``N`` qubit register."""
    out = np.array([[1]], dtype=complex)
    for k in range(N):
        out = np.kron(out, P if k == i else (R if k == j else I2))
    return out


def _hamiltonian(h, J, lam, f):
    """H = (1-lam) sum_i X_i + lam (sum_{i<j} J_ij Z_i Z_j + sum_i h_i Z_i) + f sum_i Z_i."""
    N = len(h)
    Sx = sum(_op(PX, i, N) for i in range(N))
    Sz = sum(_op(PZ, i, N) for i in range(N))
    ZZ = sum(J[i][j] * _op2(PZ, i, PZ, j, N) for i in range(N) for j in range(i + 1, N))
    Hz = sum(h[i] * _op(PZ, i, N) for i in range(N))
    return (1 - lam) * Sx + lam * (ZZ + Hz) + f * Sz


def _d_hamiltonian(h, J, f_deriv):
    """The lambda-derivative of :func:`_hamiltonian`; f is a function of lambda, hence f_deriv."""
    N = len(h)
    Sx = sum(_op(PX, i, N) for i in range(N))
    Sz = sum(_op(PZ, i, N) for i in range(N))
    ZZ = sum(J[i][j] * _op2(PZ, i, PZ, j, N) for i in range(N) for j in range(i + 1, N))
    Hz = sum(h[i] * _op(PZ, i, N) for i in range(N))
    return -Sx + ZZ + Hz + f_deriv * Sz


def _minimise_action(ansatz, H, dH):
    """Minimise Tr[(dH + i sum_k a_k [A_k, H])**2] over the coefficients a_k.

    The action is quadratic in the coefficients, so the minimum solves a linear system.
    """
    K = [1j * (A @ H - H @ A) for A in ansatz]
    M = np.array([[np.trace(a @ b).real for b in K] for a in K])
    g = np.array([-np.trace(dH @ k).real for k in K])
    return np.linalg.solve(M, g)


def _order1_ansatz(N, uniform):
    """A = sum_i Y_i (one shared coefficient) or the per-site operators Y_i."""
    if uniform:
        return [sum(_op(PY, i, N) for i in range(N))]
    return [_op(PY, i, N) for i in range(N)]


def _nc_site_operators(h, J, N):
    """A_i = -2 (h_i Y_i + sum_{j<i} J_ij (Y_i Z_j + Z_i Y_j)), as built by AGP_params."""
    out = []
    for i in range(N):
        A_i = h[i] * _op(PY, i, N)
        for j in range(i):
            A_i = A_i + J[i][j] * (_op2(PY, i, PZ, j, N) + _op2(PZ, i, PY, j, N))
        out.append(-2 * A_i)
    return out


def _random_qubo(N, seed):
    """A symmetric QUBO matrix and the h, J it induces."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.2, 0.8, (N, N))
    Q = (A + A.T) / 2
    return Q, -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1), 0.5 * Q


@pytest.mark.parametrize("N", [3, 4])
@pytest.mark.parametrize("uniform", [True, False])
def test_lcd_order1_coefficients_minimise_the_action(N, uniform):
    """The order1 closed forms in create_LCD_instance solve the variational problem."""
    Q, h, J = _random_qubo(N, seed=100 + N)
    lam = 0.37

    coeff_func = create_LCD_instance(Q, agp_type="order1", uniform_AGP_coeffs=uniform)[4]
    got = np.asarray(coeff_func(lam), dtype=float)

    exact = _minimise_action(_order1_ansatz(N, uniform), _hamiltonian(h, J, lam, 0.0), _d_hamiltonian(h, J, 0.0))
    expected = np.full(N, exact[0]) if uniform else exact

    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("N", [3, 4])
@pytest.mark.parametrize("uniform", [True, False])
def test_cold_order1_coefficients_minimise_the_action(N, uniform):
    """The COLD coefficients stay variational once the control pulse f is switched on.

    f and f_deriv enter d_lambda H, so a formula that is right at f = 0 can still be wrong here.
    """
    Q, h, J = _random_qubo(N, seed=200 + N)
    lam, f, f_deriv = 0.42, 0.31, -0.23

    coeff_func = create_COLD_instance(Q, uniform_AGP_coeffs=uniform)[4]
    got = np.asarray(coeff_func(lam, f, f_deriv), dtype=float)

    exact = _minimise_action(_order1_ansatz(N, uniform), _hamiltonian(h, J, lam, f), _d_hamiltonian(h, J, f_deriv))
    expected = np.full(N, exact[0]) if uniform else exact

    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("N", [3, 4, 5])
def test_nc_uniform_coefficient_minimises_the_action(N):
    """The nested-commutator closed form matches the merged single-operator ansatz."""
    Q, h, J = _random_qubo(N, seed=300 + N)
    lam = 0.61

    coeff_func = create_LCD_instance(Q, agp_type="nc", uniform_AGP_coeffs=True)[4]
    got = np.asarray(coeff_func(lam), dtype=float)

    merged = [sum(_nc_site_operators(h, J, N))]
    exact = _minimise_action(merged, _hamiltonian(h, J, lam, 0.0), _d_hamiltonian(h, J, 0.0))

    np.testing.assert_allclose(got, np.full(N, exact[0]), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("N", [3, 4])
def test_solve_alpha_matches_the_action_for_the_circuit_hamiltonian(N):
    """AGP_params must model the +(1-lam) sum_i X_i the circuit evolves, not its negative.

    Describing the transverse field with the opposite sign gives a Hamiltonian related by
    prod_i Z_i, under which sigma^y -> -sigma^y, so every coefficient comes back negated.
    """
    _, h, J = _random_qubo(N, seed=400 + N)
    lam = 0.28

    got = np.asarray(solve_alpha(h, J, lam), dtype=float)
    exact = _minimise_action(_nc_site_operators(h, J, N), _hamiltonian(h, J, lam, 0.0), _d_hamiltonian(h, J, 0.0))

    np.testing.assert_allclose(got, exact, rtol=1e-10, atol=1e-12)


def test_nc_uniform_coefficient_handles_degenerate_inputs():
    """A QUBO with no couplings, and one with no local fields, must stay finite."""
    coeff_no_coupling = create_LCD_instance(np.diag([-1.0, -0.6, 0.4]), agp_type="nc", uniform_AGP_coeffs=True)[4]
    assert np.all(np.isfinite(coeff_no_coupling(0.5)))

    # zero diagonal and zero row sums, so h = -0.5 diag(Q) - 0.5 sum(Q, axis=1) vanishes.
    # A 4-cycle with alternating signs is the smallest such QUBO with non-zero couplings.
    Q = np.array([[0.0, 1.0, 0.0, -1.0], [1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-1.0, 0.0, 1.0, 0.0]])
    assert np.allclose(-0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1), 0.0)
    coeff_no_field = create_LCD_instance(Q, agp_type="nc", uniform_AGP_coeffs=True)[4]
    assert np.all(np.isfinite(coeff_no_field(0.5)))


def test_order1_coefficients_are_independent_of_qubit_labelling():
    """Relabelling the qubits must permute the coefficients, not change them."""
    N = 4
    Q, _, _ = _random_qubo(N, seed=500)
    perm = np.array([3, 0, 2, 1])

    coeff = create_LCD_instance(Q, agp_type="order1", uniform_AGP_coeffs=False)[4]
    coeff_permuted = create_LCD_instance(Q[np.ix_(perm, perm)], agp_type="order1", uniform_AGP_coeffs=False)[4]

    got = np.asarray(coeff_permuted(0.4))[np.argsort(perm)]
    np.testing.assert_allclose(got, np.asarray(coeff(0.4)), rtol=1e-10, atol=1e-12)


def test_agp_coefficients_stay_cheap_for_a_large_dense_qubo():
    """The closed forms must not regress into a per-timestep minimal-action solve.

    An explicit solve is O(N**4) and takes tens of seconds for N = 100, which would dominate
    any run; the closed forms are O(N**2) to set up and O(1) per timestep.
    """
    rng = np.random.default_rng(600)
    A = rng.uniform(-1.0, 1.0, (100, 100))
    Q = (A + A.T) / 2

    for agp_type in ("order1", "nc"):
        coeff = create_LCD_instance(Q, agp_type=agp_type, uniform_AGP_coeffs=True)[4]
        values = [coeff(lam) for lam in np.linspace(0.05, 0.95, 50)]
        assert all(np.all(np.isfinite(v)) for v in values)


@pytest.mark.parametrize(
    ("agp_type", "uniform"),
    [("order1", False), ("nc", True), ("nc", False)],
)
def test_counterdiabatic_drive_helps_at_short_evolution_time(agp_type, uniform):
    """At short T the AGP is what suppresses excitations, so it must beat no drive at all.

    This is the regime the counterdiabatic term exists for. A coefficient with the wrong sign or
    magnitude can still look reasonable at long T, where the sweep is near-adiabatic anyway, but
    here it shows up immediately -- the nested-commutator coefficient used to land below the
    no-drive baseline.
    """
    from qrisp import QuantumVariable
    from qrisp.algorithms.cold import DCQOProblem

    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])
    solution = "1011"
    N = Q.shape[0]

    def probability(disable_agp):
        operators = list(create_LCD_instance(Q, agp_type=agp_type, uniform_AGP_coeffs=uniform))
        if disable_agp:
            operators[4] = lambda lam: [0.0] * N
        res = DCQOProblem(*operators).run(QuantumVariable(N), N_steps=10, T=1, method="LCD")
        return res.get(solution, [0.0])[0]

    driven = probability(disable_agp=False)
    undriven = probability(disable_agp=True)

    assert driven > 4 * undriven, f"AGP barely helps: driven={driven:.4f} undriven={undriven:.4f}"
    assert driven > MIN_DRIVEN_PROBABILITY


@pytest.mark.parametrize("objective", ["exp_value", "agp_coeff_magnitude"])
def test_cold_runs_at_short_evolution_time_for_both_objectives(objective):
    """Both objectives must stay usable in the short-T regime, and return a real distribution."""
    from qrisp.algorithms.cold import solve_QUBO

    np.random.seed(42)  # Deterministic for reproducible test results
    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    res = solve_QUBO(
        Q,
        problem_args={"method": "COLD", "uniform": False},
        run_args={"N_steps": 10, "T": 1, "N_opt": 1, "CRAB": False, "objective": objective, "bounds": (-3, 3)},
    )

    assert abs(sum(prob for prob, _ in res.values()) - 1.0) < PROBABILITY_TOLERANCE
    assert "1011" in list(res.keys())[0:5]


def test_cold_and_lcd_agree_on_the_cost_of_every_returned_state():
    """The reported cost must be x^T Q x for the returned bitstring, for either method."""
    from qrisp.algorithms.cold import solve_QUBO

    np.random.seed(42)  # Deterministic for reproducible test results
    Q = np.array([[-1.2, 0.40, 0.0, 0.0], [0.40, 0.30, 0.20, 0.0], [0.0, 0.20, -1.1, 0.30], [0.0, 0.0, 0.30, -0.80]])

    for problem_args, run_args in (
        ({"method": "LCD", "agp_type": "nc", "uniform": True}, {"N_steps": 10, "T": 2}),
        ({"method": "COLD", "uniform": True}, {"N_steps": 10, "T": 2, "N_opt": 1, "bounds": (-3, 3)}),
    ):
        res = solve_QUBO(Q, problem_args=problem_args, run_args=run_args)
        for state, (_, cost) in res.items():
            x = np.array([int(b) for b in state])
            assert np.isclose(cost, x @ Q @ x), f"{state}: reported {cost}, expected {x @ Q @ x}"


def test_every_agp_type_and_uniformity_combination_is_reachable():
    """Guards the LCD builder dispatch table against a silently dropped combination."""
    Q, _, _ = _random_qubo(4, seed=700)
    for agp_type, uniform in itertools.product(("order1", "nc"), (True, False)):
        coeff = create_LCD_instance(Q, agp_type=agp_type, uniform_AGP_coeffs=uniform)[4]
        assert np.all(np.isfinite(np.asarray(coeff(0.5), dtype=float)))


@pytest.mark.parametrize("N", [3, 4, 5])
def test_cold_nc_uniform_coefficient_minimises_the_action_with_control_field(N):
    """The nested-commutator closed form stays variational with the COLD control pulse on.

    f enters H and f_deriv enters d_lambda H, so the coefficient picks up terms in both that are
    absent from the LCD case. This is what makes agp_type="nc" usable from COLD at all.
    """
    Q, h, J = _random_qubo(N, seed=800 + N)
    lam, f, f_deriv = 0.46, -0.37, 0.52

    coeff_func = create_COLD_instance(Q, uniform_AGP_coeffs=True, agp_type="nc")[4]
    got = np.asarray(coeff_func(lam, f, f_deriv), dtype=float)

    merged = [sum(_nc_site_operators(h, J, N))]
    exact = _minimise_action(merged, _hamiltonian(h, J, lam, f), _d_hamiltonian(h, J, f_deriv))

    np.testing.assert_allclose(got, np.full(N, exact[0]), rtol=1e-10, atol=1e-12)


def test_cold_nc_reduces_to_the_lcd_coefficient_when_the_control_is_off():
    """Switching the control pulse off must reproduce the LCD coefficient exactly."""
    Q, _, _ = _random_qubo(4, seed=900)

    cold = create_COLD_instance(Q, uniform_AGP_coeffs=True, agp_type="nc")[4]
    lcd = create_LCD_instance(Q, agp_type="nc", uniform_AGP_coeffs=True)[4]

    for lam in (0.1, 0.5, 0.9):
        np.testing.assert_allclose(cold(lam, 0.0, 0.0), lcd(lam), rtol=1e-12, atol=1e-14)


def test_cold_rejects_non_uniform_nested_commutators():
    """COLD compiles a symbolic control pulse, which the non-uniform nc solver cannot consume.

    Refusing up front beats failing deep inside compile_U_cold with a sympy conversion error.
    """
    Q, _, _ = _random_qubo(4, seed=901)
    with pytest.raises(NotImplementedError, match="uniform_AGP_coeffs=True"):
        create_COLD_instance(Q, uniform_AGP_coeffs=False, agp_type="nc")


def test_invalid_agp_type_is_rejected_by_both_instance_builders():
    """An unknown agp_type must name the offending value and the valid options."""
    Q, _, _ = _random_qubo(4, seed=902)
    with pytest.raises(ValueError, match="bogus"):
        create_COLD_instance(Q, uniform_AGP_coeffs=True, agp_type="bogus")
    with pytest.raises(KeyError):
        create_LCD_instance(Q, agp_type="bogus", uniform_AGP_coeffs=True)


@pytest.mark.parametrize("agp_type", ["order1", "nc"])
def test_solve_qubo_routes_agp_type_to_both_methods(agp_type):
    """agp_type must reach COLD as well as LCD, and find the optimum at short evolution time."""
    from qrisp.algorithms.cold import solve_QUBO

    Q = np.array(
        [
            [-0.6, 0.2, -0.5, -0.4, -0.6, 0.0],
            [0.2, -1.0, 0.0, 0.0, 0.0, 0.0],
            [-0.5, 0.0, -1.2, 0.5, -0.5, 0.0],
            [-0.4, 0.0, 0.5, -0.8, 0.0, 0.1],
            [-0.6, 0.0, -0.5, 0.0, -1.2, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.0, 0.3],
        ]
    )
    solution = "111110"

    np.random.seed(42)  # Deterministic for reproducible test results
    res = solve_QUBO(
        Q,
        problem_args={"method": "LCD", "uniform": True, "agp_type": agp_type},
        run_args={"N_steps": 20, "T": 1},
    )
    assert solution in list(res.keys())[0:3]

    np.random.seed(42)  # Deterministic for reproducible test results
    res = solve_QUBO(
        Q,
        problem_args={"method": "COLD", "uniform": True, "agp_type": agp_type},
        run_args={"N_steps": 20, "T": 1, "N_opt": 1, "CRAB": False, "bounds": (-3, 3)},
    )
    assert solution in list(res.keys())[0:3]


def test_solve_qubo_rejects_an_unknown_method():
    """A typo in 'method' used to fall through and raise UnboundLocalError."""
    from qrisp.algorithms.cold import solve_QUBO

    Q, _, _ = _random_qubo(4, seed=903)
    with pytest.raises(ValueError, match="LCD"):
        solve_QUBO(Q, problem_args={"method": "COLDD", "uniform": True}, run_args={"N_steps": 4, "T": 1})
