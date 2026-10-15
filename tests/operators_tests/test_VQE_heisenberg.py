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

"""Tests for the Heisenberg VQE problem in qrisp.vqe.problems.heisenberg."""

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest

from qrisp import QuantumFloat, QuantumVariable
from qrisp.jasp import jaspify
from qrisp.vqe.problems.heisenberg import (
    create_heisenberg_ansatz,
    greedy_edge_coloring,
    heisenberg_problem,
)


def test_vqe_heisenberg():

    # Create a graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])

    vqe = heisenberg_problem(G, 1, 1)

    results = []
    for i in range(5):
        res = vqe.run(QuantumVariable(G.number_of_nodes()), depth=2, max_iter=50)
        results.append(res)

    assert np.abs(min(results) - (-8.0)) < 1e-1


def test_jasp_vqe_heisenberg():

    @jaspify(terminal_sampling=True)
    def main():
        # Create a graph
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])

        vqe = heisenberg_problem(G, 1, 1)

        results = jnp.array([0.0] * 5)
        for i in range(5):
            res = vqe.run(QuantumFloat(G.number_of_nodes()), depth=1, max_iter=50, optimizer="SPSA")
            results = results.at[i].set(res)

        return results

    results = main()

    assert np.abs(min(results) - (-8.0)) < 2


def _cycle_graph():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])
    return G


@pytest.mark.parametrize("E", [None, [(0, 1)]])
def test_greedy_edge_coloring(E):
    """Tests ``greedy_edge_coloring`` with and without excluded edges."""
    G = _cycle_graph()
    coloring = greedy_edge_coloring(G, E)
    assert len(coloring) >= 1


@pytest.mark.parametrize("ansatz_type", ["per edge color", "per edge"])
def test_heisenberg_ansatz_types(ansatz_type):
    """Tests the ``per edge color`` and ``per edge`` Heisenberg ansatz variants."""
    G = _cycle_graph()
    M = nx.maximal_matching(G)
    C = greedy_edge_coloring(G, M)

    ansatz = create_heisenberg_ansatz(G, 1.0, 1.0, M, C, ansatz_type=ansatz_type)
    qv = QuantumVariable(G.number_of_nodes())
    ansatz(qv, [0.1] * 20)

    vqe = heisenberg_problem(G, 1.0, 1.0, ansatz_type=ansatz_type)
    assert vqe is not None
