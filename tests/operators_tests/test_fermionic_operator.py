# ********************************************************************************
# * Copyright (c) 2024 the Qrisp authors
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

"""Tests the FermionicOperator arithmetic, reductions, and module-level helpers."""

# ruff: noqa: PLR2004 -- magic values are the expected term counts in these assertions

import operator

import pytest

from qrisp import QuantumVariable
from qrisp.operators.fermionic import FermionicOperator, a, c
from qrisp.operators.fermionic.fermionic_operator import (
    apply_fermionic_swap,
    get_swaps_for_permutation,
)
from qrisp.operators.fermionic.fermionic_term import FermionicTerm


def _t(index, creator=False):
    """Builds a single-ladder-operator FermionicTerm."""
    return FermionicTerm([(index, creator)])


def _always_true(x, y):
    """Predicate that places every pair of terms in the same group."""
    return True


def test_len_coeffs_and_printing():
    """Tests ``len``, ``coeffs``, and the string/LaTeX representations."""
    F = a(0) + 2 * c(1)
    assert F.len() == 2
    assert F.coeffs().shape == (2,)
    assert F._repr_latex_().startswith("$")
    assert str(F) == str(F.to_expr())


def test_reduce_cancellation_and_complex_coefficient():
    """Tests term cancellation and that complex coefficients survive ``reduce``."""
    # a1*a0 = -a0*a1, so the two terms cancel and the zero coefficient is removed
    F = FermionicOperator(
        {
            FermionicTerm([(0, False), (1, False)]): 1,
            FermionicTerm([(1, False), (0, False)]): 1,
        }
    )
    assert F.reduce().len() == 0

    # a complex coefficient must survive the int/float zero-cleanup branch
    F2 = FermionicOperator({_t(0): 1j, _t(1): 1.0})
    assert F2.reduce().len() == 2


def test_equality():
    """Tests structural equality of equal operators."""
    F = c(0) * a(1) + 3.0
    assert F == c(0) * a(1) + 3.0


@pytest.mark.parametrize(
    "left, right",
    [
        # different number of terms
        (c(0) * a(1) + 3.0, c(0) * a(1)),
        # same number of terms but different coefficients
        (FermionicOperator({_t(0): 1.0}), FermionicOperator({_t(0): 2.0})),
        # the (daggered) term is not present in the reduced other operator
        (FermionicOperator({_t(0): 1.0}), FermionicOperator({_t(1): 1.0})),
    ],
)
def test_inequality(left, right):
    """Tests the branches that make two operators compare unequal."""
    assert left != right


@pytest.mark.parametrize(
    "operation, left, right, expected_len",
    [
        (operator.add, a(0) + c(1), 1, 3),
        (operator.sub, a(0) + c(1), 1, 3),
        (operator.sub, 1, a(0) + c(1), 3),
        (operator.mul, a(0) + c(1), 2, 2),
    ],
)
def test_scalar_arithmetic(operation, left, right, expected_len):
    """Tests scalar addition, subtraction (both orders), and multiplication."""
    assert operation(left, right).len() == expected_len


@pytest.mark.parametrize(
    "operation, left, right",
    [
        (operator.add, a(0), "not an operator"),
        (operator.sub, a(0), "not an operator"),
        (operator.mul, a(0), "not an operator"),
        (operator.sub, "not an operator", a(0)),
    ],
)
def test_binary_arithmetic_type_errors(operation, left, right):
    """Tests that unsupported operand types raise ``TypeError``."""
    with pytest.raises(TypeError):
        operation(left, right)


@pytest.mark.parametrize(
    "operation, left_terms, right_terms, expected_len",
    [
        (operator.add, {_t(0): 0.0}, {_t(1): 1.0}, 1),
        (operator.sub, {_t(0): 0.0}, {_t(1): 1.0}, 1),
        (operator.sub, {_t(0): 1.0}, {_t(0): 1.0}, 0),
    ],
)
def test_add_sub_cancellation_branches(operation, left_terms, right_terms, expected_len):
    """Tests the zero-coefficient cleanup branches of ``__add__`` and ``__sub__``."""
    left = FermionicOperator(left_terms)
    right = FermionicOperator(right_terms)
    assert operation(left, right).len() == expected_len


def test_inplace_arithmetic():
    """Tests the in-place arithmetic operators."""
    F = a(0) + c(1)

    F += 1
    assert isinstance(F, FermionicOperator)

    F -= 1
    assert isinstance(F, FermionicOperator)

    F *= 2
    assert isinstance(F, FermionicOperator)

    # cancelling in-place addition removes the term
    G = FermionicOperator({_t(0): 1.0})
    G += FermionicOperator({_t(0): -1.0})
    assert G.len() == 0

    # in-place subtraction (scalar and operator paths)
    H = FermionicOperator({_t(0): 2.0})
    H -= 1
    assert H.len() == 2
    H = FermionicOperator({_t(0): 1.0})
    H -= FermionicOperator({_t(0): 1.0})
    assert H.len() == 0

    # in-place multiplication (scalar and operator paths)
    K = a(0) + c(1)
    K *= 2
    K *= FermionicOperator({_t(1): 1.0})
    assert isinstance(K, FermionicOperator)


@pytest.mark.parametrize("method", ["__iadd__", "__isub__", "__imul__"])
def test_inplace_arithmetic_type_errors(method):
    """Tests that the in-place operators reject unsupported operand types."""
    with pytest.raises(TypeError):
        getattr(a(0) + c(1), method)("not an operator")


@pytest.mark.parametrize(
    "coeffs, threshold, expected_len",
    [
        ({0: 0.1, 1: 5.0}, 1.0, 1),
        ({0: 0.1, 1: 0.2}, 0.5, 0),
    ],
)
def test_apply_threshold(coeffs, threshold, expected_len):
    """Tests that ``apply_threshold`` removes small terms in place."""
    F = FermionicOperator({_t(index): coeff for index, coeff in coeffs.items()})
    F.apply_threshold(threshold)
    assert F.len() == expected_len


def test_to_sparse_matrix_and_ground_state_energy():
    """Tests the sparse-matrix conversion and the ground-state energy."""
    H = c(0) * a(0)
    assert H.to_sparse_matrix().shape == (2, 2)

    H4 = sum(c(i) * a(i) for i in range(4))
    assert abs(H4.ground_state_energy()) < 1e-9


def test_to_qubit_operator_invalid_mapping():
    """Tests that an unknown fermion-to-qubit mapping raises an exception."""
    with pytest.raises(Exception):
        a(0).to_qubit_operator(mapping_type="not_a_mapping")


@pytest.mark.parametrize(
    "operator",
    [FermionicOperator({}), a(0) * c(1) + a(1) * c(0)],
)
def test_group_up(operator):
    """Tests ``group_up`` on the zero operator and on a non-empty operator."""
    groups = operator.group_up(_always_true)
    assert len(groups) == 1
    assert groups[0].terms_dict == operator.terms_dict


def test_from_openfermion():
    """Tests importing an operator-like object with OpenFermion-style terms."""

    class FakeOpenFermionOperator:
        def __init__(self, terms):
            self.terms = terms

    # OpenFermion stores (index, action) pairs with action 1 for creation
    of_operator = FakeOpenFermionOperator({((0, 1), (1, 1)): 1.0})
    F = FermionicOperator.from_openfermion(of_operator)
    assert F.len() == 1
    assert F.terms_dict == {FermionicTerm([(1, True), (0, True)]): 1.0}


@pytest.mark.parametrize(
    "operator, expected",
    [(FermionicOperator({}), 0), (a(0) + a(3), 4)],
)
def test_find_minimal_qubit_amount(operator, expected):
    """Tests the minimal qubit count for empty and non-trivial operators."""
    assert operator.find_minimal_qubit_amount() == expected


def test_dagger_hermitize_and_neg():
    """Tests ``dagger``, ``hermitize``, and negation."""
    F = a(0) * c(1)
    assert F.dagger().terms_dict == {FermionicTerm([(0, True), (1, False)]): 1}
    assert isinstance(F.hermitize(), FermionicOperator)

    negated = -F
    assert negated.terms_dict == {term: -coeff for term, coeff in F.terms_dict.items()}


def test_reduce_assume_hermitian():
    """Tests ``reduce`` with the ``assume_hermitian`` flag."""
    op = a(0) * a(1) + c(1) * c(0)
    reduced = op.reduce(assume_hermitian=True)
    assert reduced.len() == 1


@pytest.mark.parametrize(
    "other, expected_equal",
    [
        # the sorted dagger of the self term is present, but the coefficients differ
        (FermionicOperator({FermionicTerm([(0, True)]): 2.0}), False),
        # matching coefficients are treated as equal (Hermitian conjugate terms)
        (FermionicOperator({FermionicTerm([(0, True)]): 1.0}), True),
    ],
)
def test_equality_daggered_coefficient(other, expected_equal):
    """Tests the daggered-term coefficient comparison in ``__eq__``."""
    assert (FermionicOperator({_t(0): 1.0}) == other) is expected_equal


@pytest.mark.parametrize(
    "other, expected_len",
    [
        # zero coefficient of self is removed in __rsub__
        (FermionicOperator({_t(0): 0.0}), 1),
        # cancelling identity term of other is removed in __rsub__
        (FermionicOperator({FermionicTerm(): 1.0}), 0),
    ],
)
def test_rsub_cleanup(other, expected_len):
    """Tests the zero-coefficient cleanup branches of ``__rsub__``."""
    assert (1 - other).len() == expected_len


@pytest.mark.parametrize(
    "method, expected_len",
    [
        ("__iadd__", 2),
        ("__isub__", 2),
    ],
)
def test_inplace_non_cancelling_terms(method, expected_len):
    """Tests in-place arithmetic when terms do not cancel."""
    F = FermionicOperator({_t(0): 1.0})
    assert getattr(F, method)(FermionicOperator({_t(1): 1.0})).len() == expected_len


def test_from_pyscf():
    """Tests constructing a FermionicOperator from a PySCF molecule."""
    pytest.importorskip("pyscf")
    from pyscf import gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    H = FermionicOperator.from_pyscf(mol)
    assert isinstance(H, FermionicOperator)
    assert H.len() > 0


def test_apply_fermionic_swap_and_swaps_for_permutation():
    """Tests the fermionic swap and the adjacent-swap helper."""
    qv = QuantumVariable(3)
    swapped = apply_fermionic_swap(qv, [2, 0, 1])
    assert len(swapped) == 3

    swaps = get_swaps_for_permutation([2, 0, 1])
    assert all(len(swap) == 2 for swap in swaps)
