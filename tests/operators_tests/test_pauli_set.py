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


import itertools
from functools import reduce

import numpy as np
import pytest

import qrisp.operators as ops
from qrisp.operators.qubit.pauli_network import PauliSet

# Qrisp convention: qubit 0 is the leftmost tensor factor.
I2 = np.eye(2)
# Keyed by (z, x), as in the paper.
PAULI = {
    (0, 0): I2,
    (0, 1): np.array([[0, 1], [1, 0]]),
    (1, 0): np.diag([1, -1]),
    (1, 1): np.array([[0, -1j], [1j, 0]]),
}
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
S = np.diag([1, 1j])
SX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2
P0, P1 = np.diag([1, 0]), np.diag([0, 1])


def single(gate, q):
    return np.kron(gate, I2) if q == 0 else np.kron(I2, gate)


GATES = {
    "h": (lambda ps: ps.h(1), single(H, 1)),
    "s": (lambda ps: ps.s(0), single(S, 0)),
    "sx": (lambda ps: ps.sx(0), single(SX, 0)),
    "cx_01": (lambda ps: ps.cx(0, 1), np.kron(P0, I2) + np.kron(P1, PAULI[0, 1])),
    "cx_10": (lambda ps: ps.cx(1, 0), np.kron(I2, P0) + np.kron(PAULI[0, 1], P1)),
}


def pauli_matrix(pauli_set: PauliSet, k):
    sign = -1 if pauli_set.signs[k] else 1
    return sign * reduce(
        np.kron, [PAULI[int(pauli_set.z[q, k]), int(pauli_set.x[q, k])] for q in range(pauli_set.num_qubits)]
    )


def all_two_qubit_paulis():
    """All 16 two-qubit Paulis, each once with sign +1 and once with sign -1."""
    bits = np.array(list(itertools.product([False, True], repeat=4))).T
    bits = np.concatenate([bits, bits], axis=1)
    ps = PauliSet(bits[:2], bits[2:])
    ps.signs[16:] = True
    return ps


class TestGates:
    @pytest.mark.parametrize("name", GATES)
    def test_conjugation_matches_matrices(self, name):
        apply, G = GATES[name]
        ps = all_two_qubit_paulis()
        before = [pauli_matrix(ps, k) for k in range(ps.num_paulis)]
        apply(ps)
        for k in range(ps.num_paulis):
            assert np.allclose(pauli_matrix(ps, k), G @ before[k] @ G.conj().T)

    def test_cx_needs_distinct_qubits(self):
        with pytest.raises(ValueError, match="distinct"):
            all_two_qubit_paulis().cx(1, 1)


class TestCopyAndSlicing:
    def test_keeps_signs(self):
        ps = all_two_qubit_paulis()
        copied = ps.copy()
        copied.signs[0] = True
        assert not ps.signs[0]
        assert list(ps[:, np.array([16, 0])].signs) == [True, False]

    def test_slicing_shapes(self):
        ps = all_two_qubit_paulis()
        assert (ps[1].num_qubits, ps[1].num_paulis) == (1, 32)
        assert (ps[:, 3].num_qubits, ps[:, 3].num_paulis) == (2, 1)
        assert (ps[0, 3].num_qubits, ps[0, 3].num_paulis) == (1, 1)
        assert (ps[:, 1:].num_qubits, ps[:, 1:].num_paulis) == (2, 31)

    def test_slice_is_independent(self):
        ps = all_two_qubit_paulis()
        before = ps.copy()
        sub = ps[:, 1:]
        sub.h(0)
        sub.s(1)
        assert (ps.z == before.z).all() and (ps.x == before.x).all() and (ps.signs == before.signs).all()


class TestSupportAndSorting:
    def test_support_sizes(self):
        # Strings: X0 X1, Z0, Y2, I
        ps = PauliSet([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]], [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
        assert list(ps.support_sizes()) == [2, 1, 1, 0]
        assert list(ps.support_mask(0)) == [True, True, False]

    def test_sorted_columns_returns_order(self):
        # Strings: X0 X1, Z0, I
        ps = PauliSet([[0, 1, 0], [0, 0, 0]], [[1, 0, 0], [1, 0, 0]])
        sorted_ps, order = ps.sorted_columns()
        assert list(order) == [2, 1, 0]
        assert list(sorted_ps.support_sizes()) == [0, 1, 2]

    def test_sorted_columns_keeps_ties_in_input_order(self):
        # Strings: Z0 Z1, X0, Y1, X0 X1; the synthesis relies on a stable sort to keep term order among ties.
        ps = PauliSet([[1, 0, 0, 0], [1, 0, 1, 0]], [[0, 1, 0, 1], [0, 0, 1, 1]])
        assert list(ps.sorted_columns()[1]) == [1, 2, 0, 3]


class TestFromOperator:
    def test_encodes_each_term_as_its_column(self):
        operator = ops.X(0) * ops.Y(2) + 0.5 * ops.Z(1) + 1
        ps = PauliSet.from_operator(operator)
        assert (ps.num_qubits, ps.num_paulis) == (3, 3)
        assert not ps.signs.any()
        expected_z_x = {
            "X(0)*Y(2)": ([0, 0, 1], [1, 0, 1]),
            "Z(1)": ([0, 1, 0], [0, 0, 0]),
            "1": ([0, 0, 0], [0, 0, 0]),
        }
        for k, term in enumerate(operator.coeff_by_term):
            z, x = expected_z_x[str(term)]
            assert list(ps.z[:, k]) == z and list(ps.x[:, k]) == x

    def test_identity_only(self):
        ps = PauliSet.from_operator(ops.QubitOperator() + 1)
        assert (ps.num_qubits, ps.num_paulis) == (0, 1)

    @pytest.mark.parametrize("factor", [ops.A, ops.C, ops.P0, ops.P1])
    def test_rejects_non_pauli_factors(self, factor):
        with pytest.raises(ValueError, match="to_pauli"):
            PauliSet.from_operator(ops.X(0) + factor(1) * ops.Z(2))
