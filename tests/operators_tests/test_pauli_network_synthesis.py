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

from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from scipy.linalg import expm

import qrisp.operators.qubit.pauli_network.depth as depth_module
from qrisp import QuantumSession, QuantumVariable
from qrisp.operators import P0, P1, A, C, QubitOperator, X, Y, Z
from qrisp.operators.qubit.pauli_network import PauliSet
from qrisp.operators.qubit.pauli_network.common import PauliNetwork
from qrisp.operators.qubit.pauli_network.depth import depth_synthesis
from qrisp.operators.qubit.pauli_network.synthesis import count_synthesis

Synthesis = Callable[[PauliSet], PauliNetwork]

SYNTHESES: list[Synthesis] = [count_synthesis, depth_synthesis]
METHODS = ["count_synthesis", "depth_synthesis"]
AXIS_BITS = {"Z": (True, False), "X": (False, True), "Y": (True, True)}  # (z, x)
FACTORS = {(True, False): Z, (False, True): X, (True, True): Y}  # keyed by (z, x)


def random_pauli_set(rng: np.random.Generator, n: int, m: int, identities: bool = False) -> PauliSet:
    z = rng.random((n, m)) < 0.5
    x = rng.random((n, m)) < 0.5
    if not identities:
        z[0, ~(z | x).any(axis=0)] = True
    return PauliSet(z, x)


def random_commuting_pauli_set(rng: np.random.Generator, n: int, m: int, depth: int = 20) -> PauliSet:
    """Z strings, which commute, conjugated by a random Clifford, which keeps them commuting."""
    pauli_set = PauliSet(rng.random((n, m)) < 0.5, np.zeros((n, m)))
    for _ in range(depth):
        gate = rng.choice(["h", "s", "sx", "cx"])
        if gate == "cx":
            control, target = rng.choice(n, size=2, replace=False)
            pauli_set.cx(int(control), int(target))
        else:
            getattr(pauli_set, gate)(int(rng.integers(n)))
    return pauli_set


def heisenberg_chain(n: int) -> QubitOperator:
    bonds = [X(i) * X(i + 1) + Y(i) * Y(i + 1) + Z(i) * Z(i + 1) for i in range(n - 1)]
    return cast(QubitOperator, sum(bonds) + sum(0.5 * Z(i) for i in range(n)))


def assert_rotations_are_correct(network: PauliNetwork, pauli_set: PauliSet) -> None:
    """Replay the gates; at each Rotation the named string must be exactly (-1)^negated * axis on qubit."""
    current = pauli_set.copy()
    applied = 0
    for rotation in network.rotations:
        assert rotation.position >= applied, "rotations must be sorted by position"
        for gate in network.gates[applied : rotation.position]:
            getattr(current, gate.name)(*gate.qubits)
        applied = rotation.position
        k = rotation.term_index
        assert list(np.flatnonzero(current.support_mask(k))) == [rotation.qubit]
        assert (bool(current.z[rotation.qubit, k]), bool(current.x[rotation.qubit, k])) == AXIS_BITS[rotation.axis]
        assert bool(current.signs[k]) == rotation.negated
    # Every string is implemented exactly once: by a Rotation, or as a global phase if it is the identity.
    popped = [r.term_index for r in network.rotations] + network.identity_indices
    assert sorted(popped) == list(range(pauli_set.num_paulis))
    assert network.identity_indices == np.flatnonzero(pauli_set.support_sizes() == 0).tolist()


def term_matrix(pauli_set: PauliSet, k: int) -> npt.NDArray[np.complex128]:
    """Dense matrix of string k, with its sign."""
    operator = QubitOperator({}) + 1
    for q in np.flatnonzero(pauli_set.support_mask(k)):
        operator = operator * FACTORS[bool(pauli_set.z[q, k]), bool(pauli_set.x[q, k])](int(q))
    sign = -1 if pauli_set.signs[k] else 1
    return sign * operator.to_array(pauli_set.num_qubits)


def network_unitary(network: PauliNetwork, n: int, thetas: npt.NDArray[np.float64], mirror: bool = False):
    qv = QuantumVariable(n)
    network.apply(qv, thetas, mirror=mirror)
    return cast(QuantumSession, qv.qs).compile().get_unitary()


def cx_count(network: PauliNetwork) -> int:
    return sum(gate.name == "cx" for gate in network.gates)


def trotter_unitary(operator: QubitOperator, method: str, num_qubits: int = 3, **kwargs):
    qv = QuantumVariable(num_qubits)
    operator.trotterization(method=method, order=kwargs.pop("order", 1), **kwargs.pop("init", {}))(qv, **kwargs)
    return cast(QuantumSession, qv.qs).compile().get_unitary()


# Small and degenerate inputs, as (z, x) bit tables of shape (n, m).
EDGE_CASES = {
    "empty": (np.zeros((3, 0)), np.zeros((3, 0))),
    "only identities": (np.zeros((2, 2)), np.zeros((2, 2))),
    "single string": ([[1], [1]], [[0], [1]]),  # Z0 Y1
    "duplicates": ([[1, 1, 0], [1, 1, 1]], [[0, 0, 1], [1, 1, 1]]),
    "one qubit": ([[1, 0, 1, 1]], [[0, 1, 1, 0]]),
    "single-qubit strings": ([[1, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 1]]),  # Z0, Y1, X2
    "all commuting": ([[1, 0, 1], [1, 0, 1], [0, 0, 0]], [[0, 1, 1], [0, 1, 1], [0, 0, 0]]),  # Z0Z1, X0X1, Y0Y1
    "anticommuting": ([[1, 0], [0, 0]], [[0, 1], [0, 0]]),  # Z0, X0
    "full-support Z": (np.ones((6, 1)), np.zeros((6, 1))),
    "full-support Y": (np.ones((6, 1)), np.ones((6, 1))),
    "support on the last qubit only": (np.eye(8)[:, [7]], np.zeros((8, 1))),
}


class TestEdgeCases:
    # Returning at all is the termination check.

    @pytest.mark.parametrize("synthesis", SYNTHESES)
    @pytest.mark.parametrize("case", EDGE_CASES)
    def test_rotations_are_correct(self, synthesis, case):
        pauli_set = PauliSet(*EDGE_CASES[case])
        assert_rotations_are_correct(synthesis(pauli_set), pauli_set)

    @pytest.mark.parametrize("synthesis", SYNTHESES)
    def test_all_two_qubit_strings(self, synthesis):
        bits = np.array([[(k >> b) & 1 for k in range(1, 16)] for b in range(4)], dtype=bool)
        pauli_set = PauliSet(bits[:2], bits[2:])
        assert_rotations_are_correct(synthesis(pauli_set), pauli_set)

    @pytest.mark.parametrize("synthesis", SYNTHESES)
    def test_single_qubit_strings_need_no_gates(self, synthesis):
        pauli_set = PauliSet(*EDGE_CASES["single-qubit strings"])
        assert synthesis(pauli_set).gates == []

    @pytest.mark.parametrize("synthesis", SYNTHESES)
    def test_input_not_modified(self, synthesis):
        pauli_set = random_pauli_set(np.random.default_rng(0), 3, 5)
        before = pauli_set.copy()
        synthesis(pauli_set)
        assert (pauli_set.z == before.z).all() and (pauli_set.x == before.x).all()
        assert (pauli_set.signs == before.signs).all()

    @pytest.mark.parametrize("synthesis", SYNTHESES)
    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("n, m", [(2, 6), (4, 12), (6, 16)])
    def test_random_rotations_are_correct(self, synthesis, seed, n, m):
        pauli_set = random_pauli_set(np.random.default_rng(seed), n, m, identities=True)
        assert_rotations_are_correct(synthesis(pauli_set), pauli_set)


class TestLargeHamiltonians:
    @pytest.mark.parametrize("synthesis", SYNTHESES)
    def test_heisenberg_chain(self, synthesis):
        pauli_set = PauliSet.from_operator(heisenberg_chain(12))
        assert_rotations_are_correct(synthesis(pauli_set), pauli_set)

    def test_count_synthesis_on_a_wide_chain(self):
        pauli_set = PauliSet.from_operator(heisenberg_chain(40))
        assert_rotations_are_correct(count_synthesis(pauli_set), pauli_set)

    @pytest.mark.parametrize("synthesis, n, m", [(count_synthesis, 12, 40), (depth_synthesis, 8, 30)])
    def test_dense_random_set(self, synthesis, n, m):
        pauli_set = random_pauli_set(np.random.default_rng(1), n, m)
        assert_rotations_are_correct(synthesis(pauli_set), pauli_set)


class TestBounds:
    @pytest.mark.parametrize("synthesis", SYNTHESES)
    @pytest.mark.parametrize("seed", range(6))
    def test_cnot_lower_bound(self, synthesis, seed):
        # A chunk acts on two qubits, so it lowers a string's support by at most 1 (a Clifford never maps a
        # non-identity Pauli to the identity). Every string with support s needs at least s - 1 CNOTs.
        pauli_set = random_pauli_set(np.random.default_rng(seed), 6, 10)
        assert cx_count(synthesis(pauli_set)) >= pauli_set.support_sizes().max() - 1

    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("n, m", [(4, 12), (8, 20)])
    def test_count_synthesis_cnot_upper_bound(self, seed, n, m):
        # Each round adds one chunk (one CNOT) that clears the smallest remaining string from one of its qubits
        # (the chosen chunk has score >= 1, see test_every_reducible_pair_has_a_chunk). So the smallest support
        # drops by 1 per round until a string is popped: at most n - 1 rounds per pop, and at most m pops. This
        # is the O(nm) loop count of the paper (Sec. 3, Complexity), and it bounds the run time as well.
        pauli_set = random_pauli_set(np.random.default_rng(seed), n, m)
        assert cx_count(count_synthesis(pauli_set)) <= (n - 1) * m

    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("n, m", [(3, 6), (5, 10)])
    def test_depth_synthesis_round_count(self, monkeypatch, seed, n, m):
        # The paper proves no bound for Algorithm 2. Each round's matching includes a chunk of positive score
        # (the smallest string has one on its own qubits), and on fuzzed inputs the rounds never exceeded the
        # (n - 1) m of count_synthesis, so that cap catches a loop that stops making progress.
        rounds = 0
        matching = depth_module.nx.max_weight_matching

        def counted_matching(*args, **kwargs):
            nonlocal rounds
            rounds += 1
            assert rounds <= (n - 1) * m, "depth_synthesis exceeded (n - 1) m rounds"
            return matching(*args, **kwargs)

        monkeypatch.setattr(depth_module.nx, "max_weight_matching", counted_matching)
        pauli_set = random_pauli_set(np.random.default_rng(seed), n, m)
        assert_rotations_are_correct(depth_synthesis(pauli_set), pauli_set)

    def test_z_string_uses_a_cnot_ladder(self):
        n = 5
        pauli_set = PauliSet(np.ones((n, 1)), np.zeros((n, 1)))
        network = count_synthesis(pauli_set)
        assert cx_count(network) == n - 1
        assert_rotations_are_correct(network, pauli_set)


class TestNetworkUnitary:
    @pytest.mark.parametrize("synthesis", SYNTHESES)
    @pytest.mark.parametrize("mirror", [False, True])
    @pytest.mark.parametrize("seed", range(3))
    def test_unitary_is_the_product_of_term_exponentials(self, synthesis, mirror, seed):
        # mirror=False: e^{-i theta_k P_k} in rotation order. mirror=True: the same at half angle,
        # followed by the reverse order at half angle. Identity strings give a global phase.
        rng = np.random.default_rng(seed)
        n = 3
        pauli_set = random_pauli_set(rng, n, 6, identities=True)
        thetas = rng.uniform(-1, 1, pauli_set.num_paulis)
        network = synthesis(pauli_set)

        def exponential(k, scale=1.0):
            return expm(-1j * scale * thetas[k] * term_matrix(pauli_set, k))

        order = [rotation.term_index for rotation in network.rotations]
        if mirror:
            factors = [exponential(k, 0.5) for k in order + order[::-1]]
        else:
            factors = [exponential(k) for k in order]
        expected = np.exp(-1j * thetas[network.identity_indices].sum()) * np.eye(2**n)
        for factor in factors:
            expected = factor @ expected
        assert np.allclose(network_unitary(network, n, thetas, mirror), expected)

    @pytest.mark.parametrize("synthesis", SYNTHESES)
    @pytest.mark.parametrize("seed", range(4))
    def test_commuting_strings_give_the_exact_exponential(self, synthesis, seed):
        # Commuting strings: e^{-i sum_k theta_k P_k} is the product of the e^{-i theta_k P_k} in any order.
        rng = np.random.default_rng(seed)
        n = 4
        pauli_set = random_commuting_pauli_set(rng, n, 8)
        thetas = rng.uniform(-1, 1, pauli_set.num_paulis)
        generator = sum(thetas[k] * term_matrix(pauli_set, k) for k in range(pauli_set.num_paulis))
        expected = expm(-1j * generator)
        assert np.allclose(network_unitary(synthesis(pauli_set), n, thetas), expected)

    @pytest.mark.parametrize("synthesis", SYNTHESES)
    def test_mirror_adds_no_gates(self, synthesis):
        network = synthesis(random_pauli_set(np.random.default_rng(0), 4, 10))
        counts = []
        for mirror in (False, True):
            qv = QuantumVariable(4)
            network.apply(qv, np.ones(10), mirror=mirror)
            qs = cast(QuantumSession, qv.qs)
            counts.append(sum(instruction.op.name == "cx" for instruction in qs.data))
        assert counts[0] == counts[1] == 2 * cx_count(network)


# Commuting terms: one Trotter step is exact, whatever the term order.
COMMUTING_OPERATORS = [
    Z(0) * X(1),
    0.7 * Y(0) * Z(1) * X(2),
    X(0) * X(1) + Y(0) * Y(1) + 0.3 * Z(0) * Z(1),
    Z(0) * Z(1) + 2,  # the constant is a global phase
    Z(0) * Z(1) * Z(2) + X(0) * X(1) + 0.5 * Y(0) * Y(1) * Z(2),
]

# Non-Pauli factors are expanded with to_pauli(); these expansions commute, so one step is exact.
NON_PAULI_OPERATORS = [
    A(0) * C(1),  # not Hermitian: its hermitization (X_0 X_1 + Y_0 Y_1) / 4 is simulated
    A(0) * C(1) + C(0) * A(1),
    P0(0) * Z(1) + 0.5 * P1(0) * P1(2),
]


class TestTrotterization:
    @pytest.mark.parametrize("operator", COMMUTING_OPERATORS)
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("forward_evolution", [True, False])
    def test_commuting_operator_is_exact(self, operator, method, forward_evolution):
        t = 0.3
        sign = -1 if forward_evolution else 1
        expected = expm(sign * 1j * t * operator.to_array(3))
        actual = trotter_unitary(operator, method, t=t, init={"forward_evolution": forward_evolution})
        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("method", METHODS)
    def test_constant_operator_is_a_global_phase(self, method):
        actual = trotter_unitary(QubitOperator({}) + 2, method, num_qubits=2, t=0.3)
        assert np.allclose(actual, np.exp(-0.6j) * np.eye(4))

    @pytest.mark.parametrize("operator", NON_PAULI_OPERATORS)
    @pytest.mark.parametrize("method", METHODS)
    def test_non_pauli_factors_are_expanded(self, operator, method):
        t = 0.3
        expected = expm(-1j * t * operator.hermitize().to_array(3))
        actual = trotter_unitary(operator, method, t=t)
        assert np.isclose(abs(np.vdot(expected.ravel(), actual.ravel())), 2**3)
