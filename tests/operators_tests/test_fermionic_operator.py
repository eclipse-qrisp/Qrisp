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

import pytest

from qrisp import QuantumVariable
from qrisp.operators.fermionic import FermionicOperator, a, c
from qrisp.operators.fermionic.fermionic_operator import (
    apply_fermionic_swap,
    get_swaps_for_permutation,
)
from qrisp.operators.fermionic.fermionic_term import FermionicTerm


def _t(index, creator=False):
    return FermionicTerm([(index, creator)])


def test_len_coeffs_and_printing():
    F = a(0) + 2 * c(1)
    assert F.len() == 2
    assert F.coeffs().shape == (2,)
    assert F._repr_latex_().startswith("$")
    assert str(F) == str(F.to_expr())


def test_reduce_cancellation_and_complex_coefficient():
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
    F = c(0) * a(1) + 3.0
    assert F == c(0) * a(1) + 3.0

    # different number of terms
    assert F != c(0) * a(1)

    # same number of terms but different coefficients
    assert FermionicOperator({_t(0): 1.0}) != FermionicOperator({_t(0): 2.0})

    # the (daggered) term is not present in the reduced other operator
    assert FermionicOperator({_t(0): 1.0}) != FermionicOperator({_t(1): 1.0})


def test_add_sub_mul_and_type_errors():
    F = a(0) + c(1)

    assert (F + 1).len() == 3
    assert (F - 1).len() == 3
    assert (1 - F).len() == 3
    assert (F * 2).len() == 2

    with pytest.raises(TypeError):
        F + "not an operator"
    with pytest.raises(TypeError):
        F - "not an operator"
    with pytest.raises(TypeError):
        F * "not an operator"


def test_add_sub_cancellation_branches():
    zero_term = FermionicOperator({_t(0): 0.0})

    # __add__ / __sub__ first loop removes a zero coefficient of self
    assert (zero_term + FermionicOperator({_t(1): 1.0})).len() == 1
    assert (zero_term - FermionicOperator({_t(1): 1.0})).len() == 1

    # __sub__ second loop removes a cancelling term of other
    F = FermionicOperator({_t(0): 1.0})
    assert (F - FermionicOperator({_t(0): 1.0})).len() == 0


def test_inplace_arithmetic():
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

    with pytest.raises(TypeError):
        F.__iadd__("not an operator")
    with pytest.raises(TypeError):
        F.__isub__("not an operator")
    with pytest.raises(TypeError):
        F.__imul__("not an operator")


def test_apply_threshold():
    F = FermionicOperator({_t(0): 0.1, _t(1): 5.0})
    F.apply_threshold(1.0)
    assert F.len() == 1


def test_to_sparse_matrix_and_ground_state_energy():
    H = c(0) * a(0)
    assert H.to_sparse_matrix().shape == (2, 2)

    H4 = sum(c(i) * a(i) for i in range(4))
    assert abs(H4.ground_state_energy()) < 1e-9


def test_to_qubit_operator_invalid_mapping():
    with pytest.raises(Exception):
        a(0).to_qubit_operator(mapping_type="not_a_mapping")


def test_group_up_empty_operator():
    F = FermionicOperator({})
    groups = F.group_up(lambda x, y: True)
    assert groups == [F]


def test_from_openfermion():
    class FakeOpenFermionOperator:
        def __init__(self, terms):
            self.terms = terms

    # OpenFermion stores (index, action) pairs with action 1 for creation
    of_operator = FakeOpenFermionOperator({((0, 1), (1, 1)): 1.0})
    F = FermionicOperator.from_openfermion(of_operator)
    assert F.len() == 1
    assert F.terms_dict == {FermionicTerm([(1, True), (0, True)]): 1.0}


def test_find_minimal_qubit_amount():
    assert FermionicOperator({}).find_minimal_qubit_amount() == 0
    assert (a(0) + a(3)).find_minimal_qubit_amount() == 4


def test_dagger_hermitize_and_neg():
    F = a(0) * c(1)
    assert F.dagger().terms_dict == {FermionicTerm([(0, True), (1, False)]): 1}
    assert isinstance(F.hermitize(), FermionicOperator)

    negated = -F
    assert negated.terms_dict == {term: -coeff for term, coeff in F.terms_dict.items()}


def test_reduce_assume_hermitian():
    O = a(0) * a(1) + c(1) * c(0)
    reduced = O.reduce(assume_hermitian=True)
    assert reduced.len() == 1


def test_equality_daggered_coefficient_mismatch():
    # the sorted dagger of the self term is present in the reduced other operator,
    # but the coefficients do not match
    assert FermionicOperator({_t(0): 1.0}) != FermionicOperator({FermionicTerm([(0, True)]): 2.0})

    # matching coefficients are treated as equal (Hermitian conjugate terms)
    assert FermionicOperator({_t(0): 1.0}) == FermionicOperator({FermionicTerm([(0, True)]): 1.0})


def test_rsub_branches():
    with pytest.raises(TypeError):
        FermionicOperator({_t(0): 1.0}).__rsub__("not an operator")

    # zero coefficient of self is removed in __rsub__
    assert (1 - FermionicOperator({_t(0): 0.0})).len() == 1

    # cancelling identity term of other is removed in __rsub__
    assert (1 - FermionicOperator({FermionicTerm(): 1.0})).len() == 0


def test_inplace_non_cancelling_terms():
    F = FermionicOperator({_t(0): 1.0})
    F += FermionicOperator({_t(1): 1.0})
    assert F.len() == 2

    G = FermionicOperator({_t(0): 1.0})
    G -= FermionicOperator({_t(1): 1.0})
    assert G.len() == 2


def test_from_pyscf():
    pytest.importorskip("pyscf")
    from pyscf import gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    H = FermionicOperator.from_pyscf(mol)
    assert isinstance(H, FermionicOperator)
    assert H.len() > 0


def test_group_up_nonempty_operator():
    F = a(0) * c(1) + a(1) * c(0)
    groups = F.group_up(lambda x, y: True)
    assert len(groups) == 1



def test_apply_fermionic_swap_and_swaps_for_permutation():
    qv = QuantumVariable(3)
    swapped = apply_fermionic_swap(qv, [2, 0, 1])
    assert len(swapped) == 3

    swaps = get_swaps_for_permutation([2, 0, 1])
    assert all(len(swap) == 2 for swap in swaps)
