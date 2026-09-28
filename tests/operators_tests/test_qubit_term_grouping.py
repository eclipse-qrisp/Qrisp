"""********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

import itertools
import random

import numpy as np

from qrisp.operators.hamiltonian_tools import group_up_iterable
from qrisp.operators.qubit import QubitOperator
from qrisp.operators.qubit.qubit_term import QubitTerm

FACTORS = ["X", "Y", "Z", "A", "C", "P0", "P1"]


def _random_terms(n_terms, n_qubits, max_locality, seed):
    rng = random.Random(seed)
    terms = {}
    while len(terms) < n_terms:
        qubits = rng.sample(range(n_qubits), rng.randint(1, max_locality))
        terms[QubitTerm({qubit: rng.choice(FACTORS) for qubit in qubits})] = None
    return list(terms)


def test_commute_qw_matches_matrix_commutation():
    """Test that QubitTerm.commute_qw agrees with per-qubit matrix commutation.

    Covers ladder operators, projectors and terms of different support size.
    """
    matrices = {factor: QubitOperator({QubitTerm({0: factor}): 1}).to_array() for factor in FACTORS}
    matrices["I"] = np.eye(2)
    commutes = {(f, g): np.allclose(M @ N, N @ M) for (f, M), (g, N) in itertools.product(matrices.items(), repeat=2)}

    terms = _random_terms(150, 5, 5, seed=0)
    for s, t in itertools.product(terms, repeat=2):
        qubits = set(s.factor_dict) | set(t.factor_dict)
        expected = all(commutes[s.factor_dict.get(q, "I"), t.factor_dict.get(q, "I")] for q in qubits)
        assert s.commute_qw(t) == expected


def test_group_up_iterable():
    """Test that group_up_iterable partitions the items such that all pairs within a group are compatible."""
    terms = _random_terms(200, 6, 3, seed=1)
    groups = group_up_iterable(terms, lambda a, b: a.commute_qw(b))

    assert sorted(map(id, itertools.chain(*groups))) == sorted(map(id, terms))
    for group in groups:
        for a, b in itertools.combinations(group, 2):
            assert a.commute_qw(b)

    assert group_up_iterable([], lambda a, b: True) == []
    assert group_up_iterable(["x"], lambda a, b: True) == [["x"]]
    n = 5
    assert len(group_up_iterable(list(range(n)), lambda a, b: True)) == 1
    assert len(group_up_iterable(list(range(n)), lambda a, b: False)) == n
