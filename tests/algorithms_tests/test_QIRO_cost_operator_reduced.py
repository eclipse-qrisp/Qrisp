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

"""Tests that the reduced QIRO cost operators encode H_C = -sum x_i + 6 sum_E x_i x_j."""

import networkx as nx
import numpy as np
import pytest

from qrisp import QuantumVariable, x
from qrisp.algorithms.qiro import (
    create_max_clique_cost_operator_reduced,
    create_max_indep_cost_operator_reduced,
)


def energy(cost_operator, n, S, gamma=0.1):
    def phase(T):
        qv = QuantumVariable(n)
        for i in T:
            x(qv[i])
        cost_operator(qv, gamma)
        sv = qv.qs.statevector_array()
        return np.angle(sv[np.argmax(np.abs(sv))])

    d = phase(S) - phase([])
    return -((d + np.pi) % (2 * np.pi) - np.pi) / gamma


def expected(G, S):
    return -len(S) + 6 * sum(1 for i, j in G.edges() if i in S and j in S)


G = nx.Graph([(0, 1), (0, 2), (0, 3), (3, 4), (3, 5)])
SUBSETS = [[], [1, 2, 3], [1, 2, 4, 5], [0], [0, 3], [3, 4, 5]]


@pytest.mark.parametrize("S", SUBSETS)
def test_max_indep_cost_operator_reduced_matches_hamiltonian(S):
    op = create_max_indep_cost_operator_reduced([G, [], []])
    assert energy(op, 6, S) == pytest.approx(expected(G, S), abs=1e-3)


def test_max_indep_cost_operator_reduced_prefers_maximum_independent_set():
    op = create_max_indep_cost_operator_reduced([G, [], []])
    assert energy(op, 6, [1, 2, 4, 5]) < energy(op, 6, [1, 2, 3])


@pytest.mark.parametrize("S", SUBSETS)
def test_max_clique_cost_operator_reduced_matches_hamiltonian(S):
    H = nx.complement(G)
    op = create_max_clique_cost_operator_reduced([H, [], []])
    assert energy(op, 6, S) == pytest.approx(expected(G, S), abs=1e-3)
