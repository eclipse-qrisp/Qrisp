"""Tests for immutable QubitTerm value semantics."""

import pickle

import dill
import pytest

from qrisp.operators import X, Y
from qrisp.operators.qubit.qubit_term import QubitTerm


def test_qubit_term_is_canonical_and_hashable():
    """Canonicalize factor order and identity while preserving indices."""
    first = QubitTerm({2: "C", 1: "A"})
    second = QubitTerm({1: "A", 3: "I", 2: "C"})
    shifted = QubitTerm({2: "A", 3: "C"})

    assert first == second
    assert first != shifted
    assert hash(first) == hash(second)
    assert list(first.factor_dict.items()) == [(1, "A"), (2, "C")]
    assert {first: 1}[second] == 1


def test_qubit_term_is_immutable():
    """Reject changes to both factors and internal attributes."""
    factors = {0: "X"}
    term = QubitTerm(factors)
    factors[1] = "Z"

    assert term.factor_dict == {0: "X"}
    with pytest.raises(TypeError):
        term.factor_dict[1] = "Z"
    with pytest.raises(AttributeError):
        term.factor_dict = {1: "Z"}
    with pytest.raises(AttributeError):
        term._hash = 0
    with pytest.raises(AttributeError):
        del term._hash


def test_qubit_term_with_factors_returns_new_term():
    """Return a distinct term when factors are functionally replaced."""
    term = QubitTerm({0: "X"})
    updated_term = term._with_factors({1: "Z"})

    assert term.factor_dict == {0: "X"}
    assert updated_term.factor_dict == {0: "X", 1: "Z"}
    assert term.copy() is term


def test_qubit_term_cached_masks():
    """Use cached support and ladder masks for term relationship checks."""
    term = QubitTerm({0: "X", 2: "A", 4: "C"})
    matching_ladders = QubitTerm({1: "Z", 2: "C", 4: "A"})
    overlapping_ladder = QubitTerm({0: "Y", 4: "A"})
    disjoint = QubitTerm({1: "X", 3: "C"})
    expected_support_mask = sum(1 << index for index in (0, 2, 4))
    expected_ladder_mask = sum(1 << index for index in (2, 4))

    assert term.support_mask == expected_support_mask
    assert term.ladder_mask == expected_ladder_mask
    assert term.intersect(matching_ladders)
    assert term.ladders_agree(matching_ladders)
    assert term.ladders_intersect(overlapping_ladder)
    assert not term.intersect(disjoint)
    assert not term.ladders_agree(overlapping_ladder)
    assert not term.ladders_intersect(disjoint)


@pytest.mark.parametrize("serializer", [pickle, dill])
def test_qubit_term_pickle_round_trip(serializer):
    """Preserve canonical equality and hashing through serialization."""
    term = QubitTerm({2: "C", 1: "A"})
    restored_term = serializer.loads(serializer.dumps(term))

    assert restored_term == term
    assert hash(restored_term) == hash(term)


def test_sorted_insertion_grouping_with_immutable_bases():
    """Keep the insertion grouping heuristic compatible with immutable bases."""
    operator = X(0) + X(1) + Y(0)
    groups = operator.commuting_qw_groups(use_graph_coloring=False)

    assert sum(len(group.terms_dict) for group in groups) == len(operator.terms_dict)
