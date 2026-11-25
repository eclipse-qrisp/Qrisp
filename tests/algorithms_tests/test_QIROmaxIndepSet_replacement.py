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

"""Tests the QIRO replacement routines for MaxIndepSet and MaxClique on hand-made measurement results."""

import networkx as nx

from qrisp.algorithms.qiro import (
    create_max_clique_replacement_routine,
    create_max_indep_replacement_routine,
)

# Bitstring keys: key[i] is the measured value of qubit (node) i.


def test_max_indep_negative_edge_removes_only_common_neighbours():
    """Rule (d): a negatively correlated edge removes only the common neighbours of its nodes."""
    # triangle 0-1-2 plus pendant node 3 at node 1
    G = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3)])
    # <Z0 Z1> = -1 is the unique largest correlation (nodes 2 and 3 tie at +1, edges come first)
    res = {"1000": 0.5, "0100": 0.5}

    new_graph, solutions, sign, exclusions = create_max_indep_replacement_routine(res, [G, [], []])

    assert sign < 0
    assert sorted(new_graph.nodes()) == [0, 1, 3]
    assert exclusions == [2]
    assert solutions == []


def test_max_indep_does_not_modify_input_graph():
    """The replacement routine returns a reduced copy and leaves the input graph unchanged."""
    G = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3)])
    edges_before = sorted(G.edges())
    res = {"1000": 0.5, "0100": 0.5}

    create_max_indep_replacement_routine(res, [G, [], []])

    assert sorted(G.nodes()) == [0, 1, 2, 3]
    assert sorted(G.edges()) == edges_before


def test_max_indep_falls_back_to_next_largest_correlation():
    """If rule (d) would remove no node, the next-largest correlation is used."""
    # path 0-1-2: the edge (0, 1) has no common neighbours
    G = nx.Graph([(0, 1), (1, 2)])
    # <Z0 Z1> = -1.0 (largest), <Z2> = +0.8, <Z1 Z2> = +0.2, <Z0> = <Z1> = 0
    res = {"100": 0.5, "010": 0.4, "011": 0.1}

    new_graph, solutions, sign, exclusions = create_max_indep_replacement_routine(res, [G, [], []])

    # rule (d) would remove nothing, so node 2 (mostly "0") is removed instead
    assert sign > 0
    assert sorted(new_graph.nodes()) == [0, 1]
    assert exclusions == [2]
    assert solutions == []


def test_max_indep_no_applicable_correlation_returns_graph_unchanged():
    """If no correlation reduces the graph, the routine returns it unchanged with sign 0."""
    G = nx.Graph([(0, 1)])
    res = {"10": 0.5, "01": 0.5}

    new_graph, solutions, sign, exclusions = create_max_indep_replacement_routine(res, [G, [], []])

    assert sign == 0
    assert sorted(new_graph.nodes()) == [0, 1]
    assert solutions == [] and exclusions == []


def test_max_clique_positive_edge_keeps_only_common_neighbours():
    """A positively correlated edge keeps only the nodes adjacent to both of its nodes."""
    # star with centre 0: nodes 1 and 2 are not adjacent
    G = nx.Graph([(0, 1), (0, 2)])
    # <Z0 Z2> = +1 is the largest correlation, all others are 0
    res = {"101": 0.5, "111": 0.5}

    new_graph, solutions, sign, exclusions = create_max_clique_replacement_routine(res, [G, [], []])

    # node 1 is not adjacent to node 2, so it cannot be in a clique containing 0 and 2
    assert sign > 0
    assert sorted(new_graph.nodes()) == [0, 2]
    assert exclusions == [1]
    assert sorted(solutions) == [0, 2]
