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

"""Tests expectation_value for qubit operators across backends and diagonalisation methods."""

import random

from qrisp import QuantumFloat, QuantumVariable, cx, h, jaspify, x
from qrisp.default_backend import QrispSimulatorBackend
from qrisp.operators import P0, P1, A, C, X, Y, Z


def test_expectation_value(sample_size=100, seed=42, exhaustive=False):

    non_sampling_backend = QrispSimulatorBackend()

    def testing_helper(state_prep, operator_combinations):
        for H in operator_combinations:
            if isinstance(H, int):
                continue

            print(H)
            assert (
                abs(
                    H.expectation_value(state_prep, precision=0.0005, backend=non_sampling_backend)()
                    - H.to_pauli().expectation_value(state_prep, precision=0.0005, backend=non_sampling_backend)()
                )
                < 1e-1
            )
            assert (
                abs(
                    H.expectation_value(
                        state_prep, precision=0.0005, diagonalisation_method="commuting", backend=non_sampling_backend
                    )()
                    - H.to_pauli().expectation_value(
                        state_prep, precision=0.0005, diagonalisation_method="commuting", backend=non_sampling_backend
                    )()
                )
                < 1e-1
            )

            # Jasp tests
            @jaspify(terminal_sampling=True)
            def main():
                return H.expectation_value(state_prep, precision=0.01)()

            assert abs(main() - H.expectation_value(state_prep, precision=0.01, backend=non_sampling_backend)()) < 1e-1

    # Set the random seed for reproducibility
    random.seed(seed)

    # Define the full list of operators
    operator_list = [lambda x: 1, X, Y, Z, A, C, P0, P1]

    # Generate all possible combinations of operators
    all_combinations = []

    if exhaustive:
        for op1 in operator_list:
            for op2 in operator_list:
                for op3 in operator_list:
                    for op4 in operator_list:
                        H = op1(0) * op2(1) * op3(2) * op4(3)

                        if H == 1:
                            continue

                        all_combinations.append(H)
    else:
        for _ in range(sample_size):
            combination = [random.choice(operator_list) for _ in range(4)]  # Choose 4 operators
            H = combination[0](0) * combination[1](1) * combination[2](2) * combination[3](3)
            all_combinations.append(H)

    def state_prep():
        qv = QuantumFloat(4)
        return qv

    # Perform tests with the randomly generated operator combinations
    testing_helper(state_prep, all_combinations)

    def state_prep():
        qv = QuantumFloat(4)
        h(qv[0])
        return qv

    testing_helper(state_prep, all_combinations)

    def state_prep():
        qv = QuantumFloat(4)
        h(qv[0])
        cx(qv[0], qv[1])
        return qv

    testing_helper(state_prep, all_combinations)

    def state_prep():
        qv = QuantumFloat(4)
        h(qv[0])
        cx(qv[0], qv[1])
        cx(qv[0], qv[2])
        return qv

    testing_helper(state_prep, all_combinations)


def test_expectation_value_issue_165():
    """Test the expectation value method for the specific case of issue #165."""

    def state_prep():
        qv = QuantumVariable(4)
        x(qv[0])
        x(qv[1])
        return qv

    H = A(0) * C(1) * C(2) * A(3) + P1(0) * P1(2) + P1(1) * P1(3)

    assert H.expectation_value(state_prep, diagonalisation_method="commuting")() == 0
    assert H.expectation_value(state_prep, diagonalisation_method="commuting_qw")() == 0


def test_expectation_value_batched_backend():
    """Test the expectation value method with the BatchedBackend."""

    def state_prep():
        d = QuantumFloat(4)
        e = QuantumFloat(3)
        d[:] = 2
        e[:] = 2
        f = d + e
        return f

    H = Z(0) * Z(1) * Z(2) * Z(3) + X(0) * X(1) * X(2) * X(3)

    bb = QrispSimulatorBackend().batched()

    ev = H.expectation_value(state_prep, backend=bb)()


def test_commuting_measurement_bias():
    """Test that measurement on >10 qubits does not truncate low-probability outcomes
    and that expectation_value under 'commuting' matches 'commuting_qw' without bias."""
    import networkx as nx
    import numpy as np

    from qrisp.simulator.bi_arrays import DenseBiArray
    from qrisp.vqe import create_heisenberg_hamiltonian, heisenberg_problem

    # 1. Direct unit test on DenseBiArray.multi_measure for >10 qubits:
    # Verifies that dense_measurement_brute does not truncate non-zero probability states
    # relative to the maximum amplitude.
    n_qubits = 11
    arr = np.zeros(2**n_qubits, dtype=np.complex64)
    arr[0] = np.sqrt(0.8)  # dominant state, max_p = 0.8
    small_prob = 0.2 / (2**n_qubits - 1)  # ~9.77e-5 (< 0.8 * 2e-4 = 1.6e-4)
    arr[1:] = np.sqrt(small_prob)

    bi_arr = DenseBiArray(arr)
    new_arrays, p_list, outcome_index_list = bi_arr.multi_measure(
        list(range(n_qubits)), return_new_arrays=False
    )
    assert len(outcome_index_list) == 2**n_qubits
    assert np.isclose(np.sum(p_list), 1.0, atol=1e-5)

    # 2. End-to-end expectation value test on a 12-qubit Heisenberg Hamiltonian:
    rng_state = np.random.get_state()
    try:
        np.random.seed(42)

        L = 12
        G = nx.Graph([(k, (k + 1) % L) for k in range(L - 1)])
        H = create_heisenberg_hamiltonian(G, 1, 0)
        vqe = heisenberg_problem(G, 1, 0, ansatz_type="per hamiltonian")

        def state_prep():
            qv = QuantumVariable(L)
            vqe.init_function(qv)
            params = [0.2] * (len(G.edges) * 2)
            vqe.ansatz_function(qv, params)
            return qv

        ev_c = H.expectation_value(state_prep, diagonalisation_method="commuting", precision=0.01)
        ev_qw = H.expectation_value(state_prep, diagonalisation_method="commuting_qw", precision=0.01)

        val_c = ev_c()
        val_qw = ev_qw()

        assert abs(val_c - val_qw) < 0.05
    finally:
        np.random.set_state(rng_state)
