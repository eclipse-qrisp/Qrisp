"""Tests for the VQEProblem class in qrisp.vqe.vqe_problem."""

import numpy as np
from qrisp.vqe.vqe_problem import *

from qrisp import QuantumVariable, cx, ry
from qrisp.operators.qubit import X, Y, Z


def test_vqe():

    c = [-0.81054, 0.16614, 0.16892, 0.17218, -0.22573, 0.12091, 0.166145, 0.04523]
    H = (
        c[0]
        + c[1] * Z(0) * Z(2)
        + c[2] * Z(1) * Z(3)
        + c[3] * (Z(3) + Z(1))
        + c[4] * (Z(2) + Z(0))
        + c[5] * (Z(2) * Z(3) + Z(0) * Z(1))
        + c[6] * (Z(0) * Z(3) + Z(1) * Z(2))
        + c[7]
        * (
            Y(0) * Y(1) * Y(2) * Y(3)
            + X(0) * X(1) * Y(2) * Y(3)
            + Y(0) * Y(1) * X(2) * X(3)
            + X(0) * X(1) * X(2) * X(3)
        )
    )

    def ansatz(qv, theta):
        for i in range(4):
            ry(theta[i], qv[i])
        for i in range(3):
            cx(qv[i], qv[i + 1])
        cx(qv[3], qv[0])

    assert np.abs(H.ground_state_energy() - (-1.8657159209215166)) < 1e-5

    vqe = VQEProblem(hamiltonian=H, ansatz_function=ansatz, num_params=4)

    results = []
    for i in range(5):
        res = vqe.run(QuantumVariable(4), depth=1, max_iter=50)
        results.append(res)

    assert np.abs(min(results) - (-1.8657159209215166)) < 1e-1
