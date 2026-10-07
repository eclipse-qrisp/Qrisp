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

"""Tests that multiplying a QubitOperator by zero produces an operator with an empty terms dict."""

# ruff: noqa: PLR2004 -- magic values are the expected term counts in these assertions

import operator as op

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.sparse import csr_matrix

from qrisp import QuantumVariable
from qrisp.operators import P0, P1, A, C, I, QubitOperator, QubitTerm, X, Y, Z

# Operators used to exercise the grouping/basis routines.
GROUPING_OPERATORS = [
    X(0) * X(1) + Y(0) * Y(1) + Z(0) + Z(1),
    A(0) * C(1) + C(0) * A(1) + Z(2),
    X(0) + Y(1) + Z(2),
]


def _commute(a, b):
    """Predicate testing full commutativity of two terms."""
    return a.commute(b)


def _commute_qw(a, b):
    """Predicate testing qubit-wise commutativity of two terms."""
    return a.commute_qw(b)


def _always_true(a, b):
    """Predicate placing every pair of terms in the same group."""
    return True


def _always_false(a, b):
    """Predicate placing every pair of terms in separate groups."""
    return False


def _operator_unitary(qv):
    """Extracts the unitary of a QubitOperator simulation acting on ``qv``.

    Ancilla qubits introduced by the simulation are moved to the front so that
    the desired unitary is the top-left block of the full circuit unitary.
    """
    qc = qv.qs.copy()
    for _ in range(qc.num_qubits() - len(qv)):
        qc.qubits.insert(0, qc.qubits.pop(-1))
    unitary = qc.get_unitary()
    return unitary[: 2**qv.size, : 2**qv.size]


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


def test_len_and_coeffs():
    """Tests ``len`` and ``coeffs`` on operators with and without terms."""
    H = X(0) * X(1) + Y(0) * Y(1) + 0.5 * Z(0) * Z(1)
    assert H.len() == 3
    assert np.allclose(H.coeffs(), [1.0, 1.0, 0.5])

    empty = QubitOperator()
    assert empty.len() == 0
    assert empty.coeffs().shape == (0,)


def test_repr_and_latex():
    """Tests ``__str__``, ``__repr__``, ``to_expr`` and ``_repr_latex_``."""
    H = X(0) + Z(1)

    assert str(H) == "X(0) + Z(1)"
    assert repr(H) == str(H)
    assert str(H) == str(H.to_expr())

    latex = H._repr_latex_()
    assert latex.startswith("$") and latex.endswith("$")
    assert "X" in latex

    assert str(QubitOperator()) == "0"


@pytest.mark.parametrize(
    "operators",
    [
        [Z(i) for i in range(4)],
        [X(0), Z(1), Y(2)],
        [0.5 * X(0), 0.5 * X(0)],
    ],
)
def test_sum_classmethod(operators):
    """Tests that ``QubitOperator.sum`` matches the builtin ``sum``."""
    fast = QubitOperator.sum(operators)
    slow = sum(operators)

    assert fast.terms_dict == slow.terms_dict
    assert np.allclose(fast.to_array(4), slow.to_array(4))


@pytest.mark.parametrize(
    "operators, expected",
    [
        ([X(0), -X(0), Z(1)], Z(1)),
        ([X(0), Z(0), -X(0), -Z(0)], QubitOperator()),
        ([], QubitOperator()),
    ],
)
def test_sum_classmethod_prunes_zero_terms(operators, expected):
    """Tests that exactly cancelling terms are pruned by ``QubitOperator.sum``."""
    assert QubitOperator.sum(operators).terms_dict == expected.terms_dict


@pytest.mark.parametrize(
    "operator, exponent, expected",
    [
        (X(0), 0, 1),
        (X(0), 1, X(0)),
        (X(0), 2, QubitOperator() + 1),
        (X(0), 3, X(0)),
        (X(0) + Z(0), 2, (X(0) + Z(0)) * (X(0) + Z(0))),
    ],
)
def test_pow(operator, exponent, expected):
    """Tests ``__pow__`` for several exponents."""
    result = operator**exponent
    if isinstance(expected, int):
        assert result == expected
    else:
        assert np.allclose(result.to_array(1), expected.to_array(1))


@pytest.mark.parametrize(
    "operator",
    [X(0), X(0) * Z(1) + Y(0), 2 * X(0) - 3 * Z(1), QubitOperator() + 5],
)
def test_neg(operator):
    """Tests unary negation."""
    negated = -operator
    assert np.allclose(negated.to_array() + operator.to_array(), 0)
    assert (-negated).terms_dict == operator.terms_dict


@pytest.mark.parametrize("arg", [0, 3, 7])
def test_identity_factory(arg):
    """Tests the ``I`` identity factory."""
    identity = I(arg)

    # The identity is the scalar 1 and does not depend on the index.
    assert identity.terms_dict == (QubitOperator() + 1).terms_dict
    assert identity.terms_dict == I(0).terms_dict
    assert identity.find_minimal_qubit_amount() == 0

    # Multiplying by the identity leaves an operator unchanged.
    assert (identity * X(1)).terms_dict == X(1).terms_dict
    assert (X(1) * identity).terms_dict == X(1).terms_dict


def test_identity_factory_invalid_type():
    """Tests that the ``I`` factory rejects non-integer indices."""
    with pytest.raises(TypeError):
        I(0.5)


@pytest.mark.parametrize("scalar", [1, 2.5, 1j, -3])
def test_add_and_radd(scalar):
    """Tests ``__add__``/``__radd__`` with a scalar."""
    plus = X(0) + scalar
    assert np.allclose(plus.to_array(1), X(0).to_array(1) + scalar * np.eye(2))
    assert (scalar + X(0)).terms_dict == plus.terms_dict


def test_add_and_radd_operators():
    """Tests ``__add__``/``__radd__`` with operators."""
    assert (X(0) + Z(0)).len() == 2
    assert (X(0) - X(0)).len() == 0
    assert (X(0) + (-X(0))).len() == 0


@pytest.mark.parametrize("scalar", [1, 2.5, 1j, -3])
def test_sub_and_rsub(scalar):
    """Tests ``__sub__``/``__rsub__`` with a scalar."""
    minus = X(0) - scalar
    assert np.allclose(minus.to_array(1), X(0).to_array(1) - scalar * np.eye(2))

    reversed_ = scalar - X(0)
    assert np.allclose(reversed_.to_array(1), scalar * np.eye(2) - X(0).to_array(1))


def test_sub_and_rsub_operators():
    """Tests ``__sub__``/``__rsub__`` with operators."""
    diff = (X(0) + Z(0)) - (X(0) - Z(0))
    assert np.allclose(diff.to_array(1), 2 * Z(0).to_array(1))


@pytest.mark.parametrize("scalar", [2, 2j, -0.5, 3 + 1j])
def test_mul_and_rmul(scalar):
    """Tests ``__mul__``/``__rmul__`` with a scalar."""
    left = scalar * X(0)
    right = X(0) * scalar
    assert np.allclose(left.to_array(1), scalar * X(0).to_array(1))
    assert left.terms_dict == right.terms_dict


def test_mul_and_rmul_operators():
    """Tests ``__mul__`` with operators."""
    product = X(0) * Z(0)
    assert np.allclose(product.to_array(1), X(0).to_array(1) @ Z(0).to_array(1))

    H1 = X(0) + Z(0)
    H2 = X(0) - Z(0)
    assert np.allclose((H1 * H2).to_array(1), H1.to_array(1) @ H2.to_array(1))


@pytest.mark.parametrize(
    "operation, left, right",
    [
        (op.add, X(0), "not an operator"),
        (op.sub, X(0), "not an operator"),
        (op.sub, "not an operator", X(0)),
        (op.mul, X(0), "not an operator"),
    ],
)
def test_operator_arithmetic_type_errors(operation, left, right):
    """Tests that unsupported operand types raise ``TypeError``."""
    with pytest.raises(TypeError):
        operation(left, right)


def test_inplace_arithmetic():
    """Tests ``__iadd__``, ``__isub__`` and ``__imul__``."""
    H = X(0)
    H += Z(0)
    assert H.terms_dict == (X(0) + Z(0)).terms_dict

    H -= Z(0)
    assert H.terms_dict == X(0).terms_dict

    H *= 3
    assert np.allclose(H.to_array(1), 3 * X(0).to_array(1))

    H *= X(0)
    assert np.allclose(H.to_array(1), 3 * np.eye(2))

    # Scalar in-place operations
    H2 = X(0)
    H2 += 2
    assert np.allclose(H2.to_array(1), X(0).to_array(1) + 2 * np.eye(2))
    H2 -= 2
    assert np.allclose(H2.to_array(1), X(0).to_array(1))

    H3 = X(0)
    H3 *= 2 + 1j
    assert np.allclose(H3.to_array(1), (2 + 1j) * X(0).to_array(1))


@pytest.mark.parametrize("inplace_op", [op.iadd, op.isub, op.imul])
def test_inplace_arithmetic_type_errors(inplace_op):
    """Tests that in-place operations reject unsupported operand types."""
    with pytest.raises(TypeError):
        inplace_op(X(0), "not an operator")


@pytest.mark.parametrize(
    "subs_dict, expected",
    [
        ({0: 1}, Z(1)),
        ({0: 3}, 3 * Z(1)),
        ({0: 3, 1: 2}, QubitOperator() + 6),
        ({5: 7}, X(0) * Z(1)),
    ],
)
def test_subs(subs_dict, expected):
    """Tests ``subs`` replacing qubit indices by scalar values."""
    assert (X(0) * Z(1)).subs(subs_dict).terms_dict == expected.terms_dict


@pytest.mark.parametrize(
    "operator, expected",
    [
        (X(0) * Z(2), 3),
        (X(5), 6),
        (QubitOperator(), 0),
        (QubitOperator() + 3, 0),
    ],
)
def test_find_minimal_qubit_amount(operator, expected):
    """Tests ``find_minimal_qubit_amount``."""
    assert operator.find_minimal_qubit_amount() == expected


@pytest.mark.parametrize(
    "H1, H2",
    [
        (X(0) * Z(1) + Y(0), Z(0) - X(1)),
        (Z(0), Z(1)),
        (Z(0) * Z(1), Z(0) * Z(1)),
        (X(0), Y(0)),
    ],
)
def test_commutator(H1, H2):
    """Tests ``commutator`` against the matrix commutator."""
    comm = H1.commutator(H2)
    M1 = H1.to_array(2)
    M2 = H2.to_array(2)
    assert np.allclose(comm.to_array(2), M1 @ M2 - M2 @ M1)


@pytest.mark.parametrize(
    "operator, threshold, expected",
    [
        (
            QubitOperator({QubitTerm({0: "X"}): 1.0, QubitTerm({1: "Z"}): 1e-12}),
            1e-9,
            X(0),
        ),
        (X(0) + 5, 1, QubitOperator() + 5),
        (X(0) + 5, 0, X(0) + 5),
    ],
)
def test_apply_threshold(operator, threshold, expected):
    """Tests ``apply_threshold`` removes sub-threshold terms."""
    assert operator.apply_threshold(threshold).terms_dict == expected.terms_dict


@pytest.mark.parametrize(
    "operator",
    [X(0), Y(0) * Z(1), X(0) * X(1) + 0.5 * Z(0)],
)
def test_from_numpy_array(operator):
    """Tests that ``from_numpy_array`` inverts ``to_array``."""
    arr = operator.to_array()
    assert np.allclose(QubitOperator.from_numpy_array(arr).to_array(), arr)


@pytest.mark.parametrize(
    "reverse_endianness, expected",
    [(False, X(0) * Z(1)), (True, Z(0) * X(1))],
)
def test_from_matrix_endianness(reverse_endianness, expected):
    """Tests ``from_matrix`` for both endianness conventions."""
    matrix = (X(0) * Z(1)).to_array()
    result = QubitOperator.from_matrix(matrix, reverse_endianness=reverse_endianness)
    assert np.allclose(result.to_array(), expected.to_array())


@pytest.mark.parametrize(
    "matrix",
    [
        (X(0) * Z(1)).to_array(),
        csr_matrix((X(0) * Z(1)).to_array()),
    ],
)
def test_from_matrix_input_types(matrix):
    """Tests ``from_matrix`` for dense and sparse input."""
    assert np.allclose(QubitOperator.from_matrix(matrix).to_array(), (X(0) * Z(1)).to_array())


def test_from_matrix_invalid_type():
    """Tests that ``from_matrix`` rejects unsupported input types."""
    with pytest.raises(Exception):
        QubitOperator.from_matrix([[1, 0], [0, 1]])


@pytest.mark.parametrize(
    "operator, expected",
    [
        (Z(0), [[1, 0], [0, -1]]),
        (QubitOperator() + 5, [[5]]),
        (QubitOperator(), [[0]]),
    ],
)
def test_to_sparse_matrix(operator, expected):
    """Tests ``to_sparse_matrix`` for Pauli, identity, and zero operators."""
    assert np.allclose(operator.to_sparse_matrix().toarray(), expected)

    # Expansion to a larger number of factors
    assert operator.to_sparse_matrix(2).shape == (4, 4)


@pytest.mark.parametrize("operator, factor_amount", [(Z(2), 1), (X(3), 2)])
def test_to_sparse_matrix_insufficient_factor_amount(operator, factor_amount):
    """Tests that too few factors raise a ``ValueError``."""
    with pytest.raises(ValueError):
        operator.to_sparse_matrix(factor_amount)


def test_to_array():
    """Tests ``to_array`` for a simple operator and against explicit dimensions."""
    operator_ = X(0) * X(1) + 2 * P0(0) * P0(1) + 3 * P1(0) * P1(1)
    expected = np.array(
        [
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 3.0],
        ]
    )
    assert np.allclose(operator_.to_array(), expected)

    # factor_amount controls the dimension
    assert X(0).to_array(2).shape == (4, 4)


@pytest.mark.parametrize(
    "operator",
    [A(0) * C(1) * Z(2), X(0) * Z(1) + Y(0), X(0) * X(1) + 0.5 * Z(0)],
)
def test_to_pauli(operator):
    """Tests ``to_pauli`` matches the original operator."""
    n = operator.find_minimal_qubit_amount()
    assert np.allclose(operator.to_pauli().to_array(n), operator.to_array(n))


def test_to_pauli_scalar():
    """Tests ``to_pauli`` preserves identity-only operators."""
    scalar = QubitOperator() + 5
    assert np.allclose(scalar.to_pauli().to_array(), scalar.to_array())


@pytest.mark.parametrize(
    "operator, expected",
    [
        (X(0) + 0.5 * Y(1), {((0, "X"),): 1, ((1, "Y"),): 0.5}),
        (QubitOperator() + 7, {(): 7}),
    ],
)
def test_to_pauli_dict(operator, expected):
    """Tests ``_to_pauli_dict`` values."""
    assert operator._to_pauli_dict() == expected


def test_to_pauli_dict_key_format():
    """Tests ``_to_pauli_dict`` key format."""
    res = (X(0) * Z(2))._to_pauli_dict()
    for key in res:
        indices = [index for index, _ in key]
        assert indices == sorted(indices)
        assert all(pauli in ("X", "Y", "Z") for _, pauli in key)


@pytest.mark.parametrize(
    "operator, expected",
    [
        (A(0) * C(1) * Z(2), C(0) * A(1) * Z(2)),
        (X(0) * Z(1), X(0) * Z(1)),
        (1j * X(0), -1j * X(0)),
    ],
)
def test_adjoint(operator, expected):
    """Tests ``adjoint`` swaps ladder operators and conjugates coefficients."""
    assert operator.adjoint().terms_dict == expected.terms_dict

    # Involution
    assert operator.adjoint().adjoint().terms_dict == operator.terms_dict


@pytest.mark.parametrize(
    "operator, expected",
    [
        (X(0) + 1j * Y(1), X(0)),
        (X(0) * Z(1) + Y(0), X(0) * Z(1) + Y(0)),
        (A(0) * C(1), 0.5 * (A(0) * C(1) + C(0) * A(1))),
    ],
)
def test_hermitize(operator, expected):
    """Tests ``hermitize`` returns the Hermitian part of the operator."""
    assert np.allclose(operator.hermitize().to_array(2), expected.to_array(2))


@pytest.mark.parametrize(
    "operator, expected",
    [
        (A(0) * C(1) + C(0) * A(1), 2 * A(0) * C(1)),
        (X(0) * Z(1) + 0.5 * Z(0), X(0) * Z(1) + 0.5 * Z(0)),
    ],
)
def test_eliminate_ladder_conjugates(operator, expected):
    """Tests ``eliminate_ladder_conjugates`` merges adjoint ladder term pairs."""
    assert operator.eliminate_ladder_conjugates().terms_dict == expected.terms_dict


@pytest.mark.parametrize(
    "operator, expected",
    [
        (QubitOperator(), 0),
        (Z(0) * Z(1) + X(0), -np.sqrt(2)),
        (Z(0) * Z(1), -1),
    ],
)
def test_ground_state_energy(operator, expected):
    """Tests ``ground_state_energy``."""
    assert np.isclose(operator.ground_state_energy(), expected)


@pytest.mark.parametrize("operator", GROUPING_OPERATORS)
def test_commuting_groups(operator):
    """Tests ``commuting_groups`` partitions into commuting sets."""
    _assert_grouping(operator, operator.commuting_groups(), _commute)


@pytest.mark.parametrize("operator", GROUPING_OPERATORS)
def test_group_up(operator):
    """Tests ``group_up`` with a custom grouping predicate."""
    groups = operator.group_up(_commute_qw)
    _assert_grouping(operator, groups, _commute_qw)


def test_group_up_edge_cases():
    """Tests ``group_up`` edge cases."""
    H = A(0) * C(1) + C(0) * A(1) + Z(2)

    # A predicate that always returns False yields one group per term.
    assert len(H.group_up(_always_false)) == H.len()

    # The empty operator is returned as a single (empty) group.
    empty_groups = QubitOperator().group_up(_always_true)
    assert len(empty_groups) == 1
    assert empty_groups[0].terms_dict == {}


@pytest.mark.parametrize("operator", GROUPING_OPERATORS)
@pytest.mark.parametrize("use_graph_coloring", [True, False])
def test_commuting_qw_groups(operator, use_graph_coloring):
    """Tests ``commuting_qw_groups`` for both grouping strategies."""
    groups = operator.commuting_qw_groups(use_graph_coloring=use_graph_coloring)
    _assert_grouping(operator, groups, _commute_qw)


@pytest.mark.parametrize("operator", GROUPING_OPERATORS)
def test_commuting_qw_groups_bases(operator):
    """Tests ``commuting_qw_groups`` returns a basis per group on request."""
    groups, bases = operator.commuting_qw_groups(show_bases=True)
    assert len(groups) == len(bases)
    for base in bases:
        assert all(factor in ("X", "Y", "Z") for factor in base.factor_dict.values())


@pytest.mark.parametrize(
    "operator, expected",
    [
        (X(0), Z(0)),
        (Y(0), Z(0)),
        (X(0) + Y(1) + Z(2), Z(0) + Z(1) + Z(2)),
        (A(0) * C(1), 0.5 * P1(0) * Z(1)),
    ],
)
def test_change_of_basis(operator, expected):
    """Tests ``change_of_basis`` maps the given operator to a diagonal one."""
    qv = QuantumVariable(operator.find_minimal_qubit_amount())
    assert operator.change_of_basis(qv).terms_dict == expected.terms_dict


@pytest.mark.parametrize(
    "operator, method",
    [
        (X(0) + Z(1), "commuting_qw"),
        (X(0) + Z(1), "commuting"),
        (X(0) * Z(1) + Z(0) * X(1), "commuting"),
    ],
)
def test_change_of_basis_diagonal(operator, method):
    """Tests that both change-of-basis methods yield diagonal factors."""
    res = operator.change_of_basis(QuantumVariable(2), method=method)
    for term in res.terms_dict:
        assert all(factor in ("I", "Z", "P0", "P1") for factor in term.factor_dict.values())


def test_change_of_basis_insufficient_qubits():
    """Tests that ``change_of_basis`` rejects too-small quantum arguments."""
    with pytest.raises(Exception):
        X(2).change_of_basis(QuantumVariable(2))


@pytest.mark.parametrize("operator", [X(0), X(0) * Z(1), A(0) * C(1)])
def test_get_conjugation_circuit(operator):
    """Tests ``get_conjugation_circuit`` returns a circuit and the operator."""
    qc, op = operator.get_conjugation_circuit()
    assert qc.num_qubits() == operator.find_minimal_qubit_amount()
    assert op.terms_dict == operator.terms_dict


@pytest.mark.parametrize(
    "operator, n, expected",
    [
        (X(0), 1, 1 - 1 / 3),
        (X(0) * X(1), 2, 1 - 1 / 5),
        (X(0) + Z(0), 1, 2 * (1 - 1 / 3)),
        (QubitOperator() + 5, 1, 0),
    ],
)
def test_get_operator_variance(operator, n, expected):
    """Tests ``get_operator_variance`` matches the analytic formula."""
    assert np.isclose(operator.get_operator_variance(n=n), expected)


def test_unitaries():
    """Tests ``unitaries`` returns nonnegative coefficients and matching unitaries."""
    H = 2 * X(0) * X(1) - Z(0) * Z(1)
    unitaries, coeffs = H.unitaries()

    assert len(unitaries) == 2
    assert np.all(coeffs >= 0)
    assert np.allclose(sorted(coeffs), [1, 2])


@pytest.mark.parametrize(
    "operator, t",
    [
        (0.4 * X(0) * X(1) + 0.3 * Z(0) - 0.2 * Y(1), 0.5),
        (0.5 * X(0) * Z(1) + 0.25 * Y(0), 0.8),
    ],
)
def test_trotterization_second_order(operator, t):
    """Tests ``trotterization`` for the second-order Suzuki formula."""
    qv = QuantumVariable(2)
    operator.trotterization(order=2)(qv, t=t, steps=20)

    exact = expm(-1j * t * operator.to_array(2))
    assert np.allclose(_operator_unitary(qv), exact, atol=1e-4)


def _assert_grouping(operator, groups, commutes):
    """Asserts that ``groups`` partitions ``operator`` into commuting sets."""
    reconstructed = {}
    for group in groups:
        terms = list(group.terms_dict)
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                assert commutes(terms[i], terms[j])
        for term, coeff in group.terms_dict.items():
            reconstructed[term] = reconstructed.get(term, 0) + coeff

    assert reconstructed == operator.terms_dict


@pytest.mark.parametrize(
    "operation, left, right, expected",
    [
        (op.add, QubitOperator({QubitTerm(): 0.0}), X(0), X(0)),
        (op.sub, QubitOperator({QubitTerm(): 0.0}), X(0), -X(0)),
        (op.sub, X(0), X(0), QubitOperator()),
    ],
)
def test_add_sub_zero_coefficient_branches(operation, left, right, expected):
    """Tests the zero-coefficient cleanup branches of ``__add__``/``__sub__``."""
    assert operation(left, right).terms_dict == expected.terms_dict


def test_isub_loops_over_multiple_terms():
    """Tests ``__isub__`` when ``other`` has more than one term."""
    H = X(0) + Z(0)
    H -= X(1) + Z(1)
    assert H.terms_dict == (X(0) + Z(0) - X(1) - Z(1)).terms_dict


@pytest.mark.parametrize(
    "operator",
    [
        X(0),
        Y(0),
        Z(0) * Z(2) + Z(0),
        A(0),
        C(0) * A(1),
        X(0) * Y(1) + X(0) * Z(2),
    ],
)
def test_get_conjugation_circuit_branches(operator):
    """Tests ``get_conjugation_circuit`` across Pauli, sparse, and ladder factors."""
    qc, op = operator.get_conjugation_circuit()
    assert qc.num_qubits() == operator.find_minimal_qubit_amount()
    assert op.terms_dict == operator.terms_dict


@pytest.mark.parametrize("use_graph_coloring", [True, False])
def test_commuting_qw_groups_with_bases(use_graph_coloring):
    """Tests ``commuting_qw_groups(show_bases=True)`` for both grouping methods."""
    groups, bases = (X(0) + Y(1)).commuting_qw_groups(show_bases=True, use_graph_coloring=use_graph_coloring)
    assert len(groups) == len(bases)


def test_change_of_basis_commuting_without_qarg():
    """Tests the ``commuting`` method when no quantum argument is supplied."""
    res = (X(0) * Z(1) + Z(0) * X(1)).change_of_basis(method="commuting")
    for term in res.terms_dict:
        assert all(factor in ("I", "Z", "P0", "P1") for factor in term.factor_dict.values())


def test_change_of_basis_non_matching_ladder_indices():
    """Tests that mismatched ladder index sets raise an exception."""
    operator = A(0) * C(1) + A(0) * C(2)
    with pytest.raises(Exception):
        operator.change_of_basis(QuantumVariable(3))


def test_trotterization_second_order_jasp():
    """Tests second-order Trotterization under Jasp tracing against the non-traced result."""
    from qrisp.jasp import terminal_sampling

    def main():
        qv = QuantumVariable(2)
        U = (Y(0) * X(1) + Z(0) * Z(1)).trotterization(order=2)
        U(qv, 1.0, steps=2)
        return qv

    jasp_res = terminal_sampling(main)()
    qrisp_res = main().get_measurement()

    for key, value in jasp_res.items():
        assert np.isclose(value, qrisp_res[format(int(key), "02b")], atol=1e-3)
