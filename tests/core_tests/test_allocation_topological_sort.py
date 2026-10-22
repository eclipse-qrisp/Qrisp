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

"""Tests the topological sort that orders (de)allocations and measurements in the qompiler."""

import numpy as np
import pytest
import scipy.sparse as sps

from qrisp.permeability.qc_transformations.memory_management import toposort_helper


def reference_toposort(indptr, indices, node_amount, delay_nodes, prefered_nodes):
    """The greedy of toposort_helper with dense matrices (only for small graphs)."""
    adjacency = sps.csr_matrix(
        (np.ones(indices.size, dtype=np.int64), indices, indptr), shape=(node_amount, node_amount)
    ).toarray()

    # reachable[u, v] is True if there is a path from u to v (or u == v), by repeated squaring
    reachable = (np.eye(node_amount, dtype=np.int64) + adjacency) > 0
    while True:
        squared = (reachable.astype(np.float64) @ reachable.astype(np.float64)) > 0
        if np.array_equal(squared, reachable):
            break
        reachable = squared
    ancestors = reachable.T

    res = []
    remaining_nodes = np.ones(node_amount, dtype=bool)
    if delay_nodes.size:
        dependency_matrix = ancestors[np.ix_(prefered_nodes, delay_nodes)].astype(np.int8)
        for _ in range(prefered_nodes.size):
            min_node_index = np.argmin(dependency_matrix.sum(axis=1))
            ancestor_indices = np.flatnonzero(ancestors[prefered_nodes[min_node_index]] & remaining_nodes)
            res.extend(ancestor_indices)
            remaining_nodes[ancestor_indices] = False
            dependency_matrix = np.clip(dependency_matrix - dependency_matrix[min_node_index, :], 0, 1)
            dependency_matrix[min_node_index, :] = 1
    res.extend(np.flatnonzero(remaining_nodes))
    return np.array(res, dtype=np.int32)


def random_dag(rng, node_amount, edge_probability, topological_numbering):
    """CSR arrays of a random DAG; the numbering is a topological order only if requested."""
    labels = np.arange(node_amount) if topological_numbering else rng.permutation(node_amount)
    sources, targets = np.triu_indices(node_amount, 1)
    keep = rng.random(sources.size) < edge_probability
    adjacency = sps.csr_matrix(
        (np.ones(keep.sum(), dtype=np.int8), (labels[sources[keep]], labels[targets[keep]])),
        shape=(node_amount, node_amount),
    )
    return adjacency.indptr.astype(np.int32), adjacency.indices.astype(np.int32)


@pytest.mark.parametrize("roles", ["mixed", "few_delay_nodes", "no_delay_nodes", "delay_all_others"])
def test_toposort_helper_matches_reference(roles):
    rng = np.random.default_rng(["mixed", "few_delay_nodes", "no_delay_nodes", "delay_all_others"].index(roles))
    for _ in range(250):
        node_amount = int(rng.integers(2, 60))
        indptr, indices = random_dag(rng, node_amount, rng.uniform(0.02, 0.4), topological_numbering=False)
        if roles == "mixed":
            role = rng.choice(3, size=node_amount, p=[0.5, 0.25, 0.25])
        elif roles == "few_delay_nodes":
            role = rng.choice(3, size=node_amount, p=[0.8, 0.15, 0.05])
        elif roles == "no_delay_nodes":
            role = rng.choice(2, size=node_amount, p=[0.7, 0.3])
        else:  # like the light cone reduction: every node that is not prefered is delayed
            role = np.where(rng.random(node_amount) < 0.1, 1, 2)
        prefered_nodes = np.flatnonzero(role == 1).astype(np.int32)
        delay_nodes = np.flatnonzero(role == 2).astype(np.int32)
        if prefered_nodes.size == 0:
            continue
        args = (indptr, indices, node_amount, delay_nodes, prefered_nodes)
        assert np.array_equal(toposort_helper(*args), reference_toposort(*args))


def test_toposort_helper_matches_reference_beyond_one_bitset_word():
    # the dependencies are bitsets of 64 delay nodes per word: graphs with 64 to about 250 delay nodes
    rng = np.random.default_rng(5)
    for _ in range(60):
        node_amount = int(rng.integers(100, 260))
        indptr, indices = random_dag(rng, node_amount, rng.uniform(0.005, 0.05), topological_numbering=False)
        role = rng.choice(3, size=node_amount, p=[0.1, 0.1, 0.8])
        prefered_nodes = np.flatnonzero(role == 1).astype(np.int32)
        delay_nodes = np.flatnonzero(role == 2).astype(np.int32)
        assert delay_nodes.size > 64
        args = (indptr, indices, node_amount, delay_nodes, prefered_nodes)
        assert np.array_equal(toposort_helper(*args), reference_toposort(*args))


@pytest.mark.parametrize("delay_amount, dependency", [(64, 63), (65, 64), (128, 63), (128, 64), (128, 127)])
def test_toposort_helper_at_bitset_word_boundaries(delay_amount, dependency):
    # The first prefered node depends only on delay node number `dependency` (the last bit of the first word, or the
    # first or last bit of the second word), the second prefered node on none. The second one has to be placed first;
    # if the dependency on the boundary bit were lost, both would require no delay node and the first one would win.
    node_amount = delay_amount + 2
    dependent, independent = delay_amount, delay_amount + 1
    adjacency = sps.csr_matrix(([1], ([dependency], [dependent])), shape=(node_amount, node_amount), dtype=np.int8)
    args = (
        adjacency.indptr.astype(np.int32),
        adjacency.indices.astype(np.int32),
        node_amount,
        np.arange(delay_amount, dtype=np.int32),
        np.array([dependent, independent], dtype=np.int32),
    )
    res = toposort_helper(*args)
    assert res[0] == independent
    assert np.array_equal(res, reference_toposort(*args))


def test_toposort_helper_returns_topological_order():
    rng = np.random.default_rng(4)
    for _ in range(100):
        node_amount = int(rng.integers(2, 200))
        indptr, indices = random_dag(rng, node_amount, rng.uniform(0.01, 0.2), topological_numbering=True)
        role = rng.choice(3, size=node_amount)
        prefered_nodes = np.flatnonzero(role == 1).astype(np.int32)
        delay_nodes = np.flatnonzero(role == 2).astype(np.int32)
        res = toposort_helper(indptr, indices, node_amount, delay_nodes, prefered_nodes)

        assert np.array_equal(np.sort(res), np.arange(node_amount))
        position = np.empty(node_amount, dtype=np.int64)
        position[res] = np.arange(node_amount)
        sources = np.repeat(np.arange(node_amount), np.diff(indptr))
        assert np.all(position[sources] < position[indices])


def test_allocations_are_reused_across_grover_iterations():
    from qrisp import QuantumFloat, auto_uncompute, z
    from qrisp.grover import grovers_alg

    @auto_uncompute
    def sqrt_oracle(qf):
        temp_qbool = qf * qf == 0.25
        z(temp_qbool)

    qubit_amounts = []
    for iterations in (1, 32):
        qf = QuantumFloat(3, -1, signed=True)
        grovers_alg(qf, sqrt_oracle, iterations=iterations)
        qubit_amounts.append(qf.qs.compile().num_qubits())

    assert qubit_amounts[0] == qubit_amounts[1]
