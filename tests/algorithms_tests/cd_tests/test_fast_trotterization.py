import numpy as np
import sympy as sp
from scipy.linalg import expm, norm

from qrisp import QuantumVariable, h
from qrisp.algorithms.cold.fast_trotterization import fast_trotterization, is_flat_ising_operator
from qrisp.operators.qubit import A, C, P0, X, Y, Z


def _up_to_global_phase_close(sv1, sv2, tol=1e-6):
    overlap = np.vdot(sv1, sv2)
    return abs(abs(overlap) - 1.0) < tol


def test_is_flat_ising_operator():
    N = 4
    assert is_flat_ising_operator(sum(X(i) for i in range(N)))
    assert is_flat_ising_operator(sum(Y(i) for i in range(N)))
    assert is_flat_ising_operator(sum(Z(i) for i in range(N)))
    assert is_flat_ising_operator(
        sum(Z(i) * Z(j) for i in range(N) for j in range(i + 1, N)) + sum(Z(i) for i in range(N))
    )
    # Not eligible: ladder operators
    assert not is_flat_ising_operator(A(0) * C(1))
    # Not eligible: projector
    assert not is_flat_ising_operator(P0(0) * Z(1))
    # Not eligible: 3-qubit term
    assert not is_flat_ising_operator(X(0) * Y(1) * Z(2))
    # Not eligible: X*X cross term
    assert not is_flat_ising_operator(X(0) * X(1))


def test_fast_trotterization_matches_general_path():
    N = 4
    np.random.seed(0)
    cx_, cy_, cz_ = (np.random.uniform(-1, 1, N) for _ in range(3))
    J = np.random.uniform(-1, 1, (N, N))
    J = (J + J.T) / 2

    hamiltonians = [
        sum(cx_[i] * X(i) for i in range(N)),
        sum(cy_[i] * Y(i) for i in range(N)),
        sum(cz_[i] * Z(i) for i in range(N)),
        sum(J[i][j] * Z(i) * Z(j) for i in range(N) for j in range(i + 1, N)) + sum(cz_[i] * Z(i) for i in range(N)),
    ]

    for H in hamiltonians:
        assert is_flat_ising_operator(H)
        for forward_evolution in [True, False]:
            for steps in [1, 3]:
                U_fast = fast_trotterization(H, forward_evolution=forward_evolution)
                U_general = H.trotterization(forward_evolution=forward_evolution)

                qv_fast = QuantumVariable(N)
                h(qv_fast)
                U_fast(qv_fast, t=0.31, steps=steps)
                sv_fast = qv_fast.qs.statevector_array()

                qv_general = QuantumVariable(N)
                h(qv_general)
                U_general(qv_general, t=0.31, steps=steps)
                sv_general = qv_general.qs.statevector_array()

                assert _up_to_global_phase_close(sv_fast, sv_general)


def test_fast_trotterization_falls_back_correctly():
    # A ladder-operator Hamiltonian is not eligible for the fast path;
    # fast_trotterization must produce the exact same circuit/behavior as
    # calling .trotterization() directly.
    H = A(0) * C(1) * Z(2) + 0.5 * Y(3)
    assert not is_flat_ising_operator(H)

    U_fast = fast_trotterization(H)
    U_general = H.trotterization()

    qv_fast = QuantumVariable(4)
    U_fast(qv_fast, t=0.5)
    qv_general = QuantumVariable(4)
    U_general(qv_general, t=0.5)

    assert np.allclose(qv_fast.qs.statevector_array(), qv_general.qs.statevector_array(), atol=1e-8)
