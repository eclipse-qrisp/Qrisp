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

"""Tests for QubitOperator."""

import numpy as np
import pytest

from qrisp import QuantumVariable
from qrisp.operators.qubit import P0, P1, A, C, I, QubitOperator, X, Y, Z
from qrisp.operators.qubit.qubit_term import QubitTerm

FACTORS = [X, Y, Z, A, C, P0, P1]


def test_qubit_operator_mul_by_zero():
    """Test that multiplying a QubitOperator by zero results in an operator with an empty terms dict,
    rather than a dict with a single term with zero coefficient.
    """
    op = X(1) * 0
    assert op.terms_dict == {}
    assert not op.find_minimal_qubit_amount()

    op = 0 * X(1)
    assert op.terms_dict == {}
    assert not op.find_minimal_qubit_amount()

    op = X(1)
    op *= 0
    assert op.terms_dict == {}
    assert not op.find_minimal_qubit_amount()


@pytest.mark.parametrize(
    "operator",
    [Y(0), X(0) * Y(1), Y(0) * Z(1) + 0.5 * Y(0), Y(0) * Y(1) * X(2)],
)
def test_change_of_basis_reconstructs_operator(operator):
    """The gates U applied by change_of_basis and the returned diagonal operator D
    satisfy U^dg D U = operator, so measuring D after U measures the operator.
    """
    n = operator.find_minimal_qubit_amount()
    qv = QuantumVariable(n)
    diagonal_operator = operator.change_of_basis(qv)
    U = qv.qs.compile().get_unitary()
    assert np.allclose(U.conj().T @ diagonal_operator.to_array(n) @ U, operator.to_array(n))


@pytest.mark.parametrize("factor", FACTORS + [I])
@pytest.mark.parametrize("k", [0, 3])
def test_a_factor_on_qubit_k_acts_on_k_plus_one_qubits(factor, k):
    """X(k).num_qubits is k + 1, and so is I(k).num_qubits."""
    assert factor(k).num_qubits == k + 1


@pytest.mark.parametrize("factor", FACTORS)
def test_num_qubits_of_a_factor_is_its_minimal_qubit_amount(factor):
    """Only the identity factor raises num_qubits above find_minimal_qubit_amount."""
    assert factor(4).num_qubits == factor(4).find_minimal_qubit_amount()


def test_the_identity_factor_only_changes_num_qubits():
    """I(k) is the identity operator, on k + 1 qubits."""
    assert I(2).find_minimal_qubit_amount() == 0
    assert np.allclose(I(2).to_array(), np.eye(8))

    padded = X(3) * X(4) * I(9)
    assert padded.terms_dict == (X(3) * X(4)).terms_dict
    assert padded.num_qubits == 10
    assert padded.find_minimal_qubit_amount() == 5

    with pytest.raises(TypeError):
        I(1.5)


@pytest.mark.parametrize(
    "operation",
    [
        lambda P: P + X(0),
        lambda P: X(0) + P,
        lambda P: P - X(0),
        lambda P: X(0) - P,
        lambda P: P * Z(0),
        lambda P: Z(0) * P,
        lambda P: 2 * P,
        lambda P: P * 2,
        lambda P: P + 1,
        lambda P: 1 - P,
        lambda P: P**2,
        lambda P: QubitOperator.sum([X(0), P]),
        lambda P: P.subs({3: 1}),
        lambda P: P.adjoint(),
        lambda P: P.hermitize(),
        lambda P: P.apply_threshold(0.1),
        lambda P: P.commutator(Z(3)),
        lambda P: X(0).commutator(P),
        lambda P: P.change_of_basis(QuantumVariable(10)),
        lambda P: P.get_conjugation_circuit()[1],
    ],
)
def test_derived_operators_keep_num_qubits(operation):
    """Arithmetic and the methods returning operators pass num_qubits on."""
    assert operation(X(3) * Z(4) * I(9)).num_qubits == 10


@pytest.mark.parametrize(
    "operation",
    [
        lambda P: P.to_pauli(),
        lambda P: P.eliminate_ladder_conjugates(),
        lambda P: (P + P.adjoint()).to_pauli(),
    ],
)
def test_ladder_operator_methods_keep_num_qubits(operation):
    """The methods that rewrite ladder operators pass num_qubits on as well."""
    assert operation(A(3) * C(4) * I(9)).num_qubits == 10


@pytest.mark.parametrize(
    "grouping",
    [
        lambda H: H.commuting_groups(),
        lambda H: H.commuting_qw_groups(),
        lambda H: H.commuting_qw_groups(use_graph_coloring=False),
        lambda H: H.group_up(lambda a, b: a.commute(b)),
    ],
)
def test_groups_keep_num_qubits(grouping):
    """Every group of a partition acts on the qubits of the whole operator."""
    H = (X(0) * X(1) + Z(0) * Z(1) + Y(0)) * I(5)
    groups = grouping(H)
    assert len(groups) > 1
    assert all(group.num_qubits == 6 for group in groups)


def test_num_qubits_does_not_shrink_when_terms_cancel():
    """Unlike find_minimal_qubit_amount, num_qubits remembers every factor."""
    assert (X(9) * X(9)).num_qubits == 10
    assert (X(9) * X(9)).find_minimal_qubit_amount() == 0
    assert (X(5) + Z(0) - X(5)).num_qubits == 6
    assert (X(1) * 0).num_qubits == 2

    op = X(5) + Z(0)
    op -= X(5)
    assert op.num_qubits == 6


@pytest.mark.parametrize("in_place", ["+=", "-=", "*="])
def test_in_place_arithmetic_keeps_num_qubits(in_place):
    """In-place arithmetic with an operator takes the maximum, with a scalar keeps it."""
    op = X(0)
    if in_place == "+=":
        op += I(4)
        op += 1
    elif in_place == "-=":
        op -= I(4)
        op -= 1
    else:
        op *= I(4)
        op *= 2
    assert op.num_qubits == 5


def test_explicit_num_qubits():
    """num_qubits can be given to the constructor, but is never less than the terms need."""
    assert QubitOperator({QubitTerm({1: "Z"}): 1}, num_qubits=6).num_qubits == 6
    assert QubitOperator({QubitTerm({7: "Z"}): 1}, num_qubits=2).num_qubits == 8
    assert QubitOperator().num_qubits == 0
    assert QubitOperator({}, num_qubits=np.int64(3)).num_qubits == 3

    with pytest.raises(ValueError):
        QubitOperator({}, num_qubits=2.5)


def test_editing_terms_dict_directly_is_counted():
    """A term written into terms_dict directly still counts toward num_qubits."""
    op = QubitOperator()
    op.terms_dict[QubitTerm({4: "Z"})] = 1
    assert op.num_qubits == 5


def test_matrices_default_to_num_qubits():
    """to_array and to_sparse_matrix use num_qubits unless factor_amount is given."""
    op = Z(0) * I(2)
    expected = np.kron(np.diag([1, -1]), np.eye(4))
    assert np.allclose(op.to_array(), expected)
    assert op.to_sparse_matrix().shape == (8, 8)
    assert op.to_array(1).shape == (2, 2)


def test_matrix_constructors_count_the_identity_qubits():
    """A matrix that is the identity on its last qubit still round-trips."""
    matrix = np.kron(np.array([[0, 1], [1, 0]]), np.eye(2))
    op = QubitOperator.from_numpy_array(matrix)
    assert op.num_qubits == 2
    assert op.find_minimal_qubit_amount() == 1
    assert np.allclose(op.to_array(), matrix)

    assert QubitOperator.from_matrix(np.zeros((4, 4))).num_qubits == 2
