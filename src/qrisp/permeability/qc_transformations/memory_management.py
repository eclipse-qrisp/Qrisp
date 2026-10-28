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

"""Reorders circuit allocation/deallocation gates via topological sort to minimize qubit lifetime."""

import networkx as nx
import numpy as np
from numba import njit


def optimize_allocations(qc):
    from qrisp.permeability import PermeabilityGraph, TerminatorNode

    # G = dag_from_qc(qc, remove_init_nodes=True)
    G = PermeabilityGraph(qc, remove_artificials=True)
    qc_new = qc.clearcopy()

    dealloc_identifier = lambda x: x.op.name == "qb_dealloc"
    alloc_identifer = lambda x: x.op.name == "qb_alloc"

    # mcx_identifier = lambda x : isinstance(x.op, PTControlledOperation) and x.op.base_operation.name == "x"
    # nmcx_identifier = lambda x : not mcx_identifier(x)
    # sub_sort = lambda G : topological_sort(G, prefer = mcx_identifier, delay = nmcx_identifier)
    # for n in topological_sort(G, prefer = dealloc_identifier, delay = alloc_identifer, sub_sort = sub_sort):

    # The sub_sort function takes the indices of the gates in the source qc as
    # a sorting index. This induces a valid topological ordering because of [proof]
    def sub_sort(dag):
        nodes = list(dag.nodes())

        def sort_key(x):
            if isinstance(x, TerminatorNode):
                return 0
            else:
                return x.qc_index

        nodes.sort(key=sort_key)
        return nodes

    for n in topological_sort(G, prefer=dealloc_identifier, delay=alloc_identifer, sub_sort=sub_sort):
        if n.instr:
            qc_new.append(n.instr)

    # The above algorithm does not move allocation gates to their latest possible
    # position (only compared to other deallocation gates)
    # We therefore manually move the allocation gates to the position right
    # before the first actual instruction on that qubit.
    new_data = []
    delayed_qubit_alloc_dic = {}

    for i in range(len(qc_new.data)):
        instr = qc_new.data[i]

        if instr.op.name == "qb_alloc":
            delayed_qubit_alloc_dic[instr.qubits[0]] = instr
        else:
            # We sort the qubits in order to prevent non-deterministic compilation behavior
            alloc_qubits = list(set(delayed_qubit_alloc_dic.keys()).intersection(instr.qubits))
            alloc_qubits.sort(key=lambda x: hash(x))
            for qb in alloc_qubits:
                new_data.append(delayed_qubit_alloc_dic[qb])
                del delayed_qubit_alloc_dic[qb]

            new_data.append(instr)

    for instr in delayed_qubit_alloc_dic.values():
        new_data.append(instr)

    qc_new.data = new_data

    return qc_new


# This function performs a topological sort of the graph G where we try to execute any
# deallocation gates as early as possible while still adhering to the topological order.
# We try to perform a depth-first search as described here:
# https://en.wikipedia.org/wiki/Topological_sorting
# According to the Wikipedia page, we are allowed to pick any node as a "starting point"
# of the DF-search, which allows us to modify the algorithm such that it optimizes the
# (de)allocation order. The general idea is to pick the deallocation nodes as
# starting points, where we order them, such that those deallocation nodes that
# "require" the least allocation nodes are executed first. "Require" here means that
# there is a causal relationship between the allocation and deallocation nodes,
# i.e. there is a path from the allocation node to the deallocation node.


# We can thus determine the amount of allocation nodes required for a deallocation node
# n by counting, the amount of allocation nodes in the "ancestors" subgraph of n.
def topological_sort(G, prefer=None, delay=None, sub_sort=nx.topological_sort):
    """Function to perform a topological sort on an Unqomp DAG which allows preferring/
    delaying specific types of nodes

    Parameters
    ----------
    G : nx.DiGraph
        The Unqomp DAG.
    prefer : function, optional
        Function which returns True, when presented with an Instruction, that should be
        preferred. The default is the function that returns False on all Operations
    delay : function, optional
        Function which returns True, when presented with an Instruction, that should be
        delayed. The default is the function that returns False on all Operations
    sub_sort : function, optional
        A function which performs a topological sort, which can sorting preferences of
        secondary importance. The default is nx.topological_sort.

    Returns
    -------
    lin : list[UnqompNode]
        The linearized list of UnqompNodes. The init nodes are not included.

    """
    if prefer is None:
        prefer = lambda x: False

    if delay is None:
        delay = lambda x: True

    delay_nodes = []
    prefered_nodes = []

    node_list = list(G.nodes())

    for i in range(len(node_list)):
        n = node_list[i]
        if n.instr is None:
            continue
        elif prefer(n.instr):
            prefered_nodes.append(i)
        elif delay(n.instr):
            delay_nodes.append(i)

    if len(prefered_nodes) == 0:
        return node_list

    sprs_mat = nx.to_scipy_sparse_array(G, format="csr")

    res = toposort_helper(
        sprs_mat.indptr.astype(np.int32),
        sprs_mat.indices.astype(np.int32),
        len(G),
        np.array(delay_nodes, dtype=np.int32),
        np.array(prefered_nodes, dtype=np.int32),
    )

    return [node_list[i] for i in res]


@njit(cache=True)
def toposort_helper(indptr, indices, node_amount, delay_nodes, prefered_nodes):
    """Topological order that places prefered nodes early and delay nodes late.

    The DAG is given in CSR form (``indptr``, ``indices``). As long as there are
    prefered nodes, the prefered node that requires the fewest delay nodes that have
    not been placed yet (the first one on ties) is chosen and all of its ancestors
    that have not been placed yet are appended in index order. The remaining nodes
    follow in index order.

    With ``N`` nodes, ``E`` edges, ``n`` prefered and ``m`` delay nodes, this takes
    ``O(E m / 64 + n m + n^2 + N log N)`` time. The memory is ``O(n m)`` for the
    dependency matrix plus one bitset of ``m`` bits per node that is still waiting
    for one of its successors.

    Parameters
    ----------
    indptr, indices : np.ndarray
        CSR representation of the adjacency matrix (edges point to the successors).
    node_amount : int
        The number of nodes.
    delay_nodes : np.ndarray
        Indices of the nodes that should be delayed (e.g. qubit allocations).
    prefered_nodes : np.ndarray
        Indices of the nodes that should be prefered (e.g. qubit deallocations).

    Returns
    -------
    res : np.ndarray
        The node indices in the new order.

    """
    n = prefered_nodes.size
    m = delay_nodes.size

    # This array will contain the result
    res = np.zeros(node_amount, dtype=np.int32)

    # This array tracks which nodes have not yet been placed.
    remaining_nodes = np.ones(node_amount, dtype=np.int8)

    # This integer will contain the amount of nodes that have been placed
    node_counter = 0

    if m != 0:
        pred_ptr, pred_idx = _reversed_csr(indptr, indices, node_amount)
        order = _topological_order(indptr, indices, node_amount)

        # dependency_matrix[i, j] = 1 if delay node j is an ancestor of prefered node i
        dependency_matrix = _dependency_matrix(pred_ptr, pred_idx, order, delay_nodes, prefered_nodes)

        # For each prefered node, the number of delay nodes it requires. This is the row sum
        # of the dependency matrix, which is kept up to date instead of being recomputed.
        required_delay_nodes = np.zeros(n, dtype=np.int64)
        for i in range(n):
            required_delay_nodes[i] = np.sum(dependency_matrix[i, :])

        batch = np.empty(node_amount, dtype=np.int64)

        for _ in range(n):
            # We determine the prefered node that requires the least delay nodes
            min_node_index = np.argmin(required_delay_nodes)

            _remove_delay_nodes(dependency_matrix, required_delay_nodes, min_node_index)

            # We determine the ancestors of this node that have not been placed yet
            batch_size = _remaining_ancestors(
                prefered_nodes[min_node_index], pred_ptr, pred_idx, remaining_nodes, batch
            )

            # We insert the nodes in the result array.
            # We can assume that order of the nodes induces by their numbering
            # is already a topological ordering. Therefore inserting them in
            # order is also a topological sub sort.
            res[node_counter : node_counter + batch_size] = np.sort(batch[:batch_size])
            node_counter += batch_size

    # Insert the remaining nodes
    res[node_counter:] = np.nonzero(remaining_nodes)[0]

    # return the result
    return res


@njit(cache=True)
def _remove_delay_nodes(dependency_matrix, required_delay_nodes, row):
    # The delay nodes required by the prefered node of the given row are placed now, so they
    # don't need to be considered again: we remove them from all rows. Then all entries of
    # the row are set to 1, so that it is not chosen again. This is the same update as
    # np.clip(dependency_matrix - dependency_matrix[row, :], 0, 1) followed by
    # dependency_matrix[row, :] = 1, but each delay node is removed only once, so all
    # updates together cost O(n m).
    n, m = dependency_matrix.shape
    for j in np.nonzero(dependency_matrix[row, :])[0]:
        for i in range(n):
            if dependency_matrix[i, j]:
                dependency_matrix[i, j] = 0
                required_delay_nodes[i] -= 1
    dependency_matrix[row, :] = 1
    required_delay_nodes[row] = m


@njit(cache=True)
def _remaining_ancestors(node, pred_ptr, pred_idx, remaining_nodes, batch):
    # Writes the ancestors of node (itself included) that have not been placed yet into
    # batch, marks them as placed and returns their number. The placed nodes are closed
    # under taking ancestors, so a backward search that only visits remaining nodes finds
    # all of them. The batch array serves as the queue of the search.
    if not remaining_nodes[node]:
        return 0
    remaining_nodes[node] = 0
    batch[0] = node
    head = 0
    size = 1
    while head < size:
        current = batch[head]
        head += 1
        for k in range(pred_ptr[current], pred_ptr[current + 1]):
            parent = pred_idx[k]
            if remaining_nodes[parent]:
                remaining_nodes[parent] = 0
                batch[size] = parent
                size += 1
    return size


@njit(cache=True)
def _reversed_csr(indptr, indices, node_amount):
    # CSR representation of the reversed graph, i.e. the predecessors of every node
    counts = np.zeros(node_amount + 1, dtype=np.int64)
    for k in range(indices.size):
        counts[indices[k] + 1] += 1
    pred_ptr = np.cumsum(counts)

    pred_idx = np.empty(indices.size, dtype=np.int64)
    fill = pred_ptr[:-1].copy()
    for node in range(node_amount):
        for k in range(indptr[node], indptr[node + 1]):
            child = indices[k]
            pred_idx[fill[child]] = node
            fill[child] += 1

    return pred_ptr, pred_idx


@njit(cache=True)
def _topological_order(indptr, indices, node_amount):
    # Kahn's algorithm
    in_degree = np.zeros(node_amount, dtype=np.int64)
    for k in range(indices.size):
        in_degree[indices[k]] += 1

    order = np.empty(node_amount, dtype=np.int64)
    head = 0
    tail = 0
    for node in range(node_amount):
        if in_degree[node] == 0:
            order[tail] = node
            tail += 1

    while head < tail:
        node = order[head]
        head += 1
        for k in range(indptr[node], indptr[node + 1]):
            child = indices[k]
            in_degree[child] -= 1
            if in_degree[child] == 0:
                order[tail] = child
                tail += 1

    return order


@njit(cache=True)
def _assign_slots(pred_ptr, pred_idx, order, unread):
    # Assigns every node a slot for its bitset in _dependency_matrix. Nodes are visited in
    # topological order; the slot of a node is released as soon as all of its successors
    # have been visited (unread counts the successors that have not been visited yet).
    # Returns the slot of every node and the number of slots needed.
    slot = np.empty(order.size, dtype=np.int64)
    free_slots = np.empty(order.size, dtype=np.int64)
    free_amount = 0
    slot_amount = 0
    for node in order:
        if free_amount > 0:
            free_amount -= 1
            slot[node] = free_slots[free_amount]
        else:
            slot[node] = slot_amount
            slot_amount += 1
        for k in range(pred_ptr[node], pred_ptr[node + 1]):
            parent = pred_idx[k]
            unread[parent] -= 1
            if unread[parent] == 0:
                free_slots[free_amount] = slot[parent]
                free_amount += 1
        if unread[node] == 0:
            free_slots[free_amount] = slot[node]
            free_amount += 1
    return slot, slot_amount


@njit(cache=True)
def _dependency_matrix(pred_ptr, pred_idx, order, delay_nodes, prefered_nodes):
    # Computes the matrix that contains the ancestor relations between the prefered and
    # the delay nodes: dependency_matrix[i, j] = 1 if delay node j is an ancestor of
    # prefered node i. Every node gets a bitset of the delay nodes among its ancestors,
    # the union of the bitsets of its predecessors (plus itself if it is a delay node),
    # computed in topological order. A bitset is only kept until all successors of its
    # node have read it, so the slots of _assign_slots suffice.
    node_amount = order.size
    n = prefered_nodes.size
    m = delay_nodes.size
    words = (m + 63) // 64
    one = np.uint64(1)

    delay_index = -np.ones(node_amount, dtype=np.int64)
    delay_index[delay_nodes] = np.arange(m)
    prefer_index = -np.ones(node_amount, dtype=np.int64)
    prefer_index[prefered_nodes] = np.arange(n)

    # Number of successors of every node
    successor_amount = np.zeros(node_amount, dtype=np.int64)
    for k in range(pred_idx.size):
        successor_amount[pred_idx[k]] += 1

    slot, slot_amount = _assign_slots(pred_ptr, pred_idx, order, successor_amount)
    bitsets = np.zeros((slot_amount, words), dtype=np.uint64)

    dependency_matrix = np.zeros((n, m), dtype=np.int8)

    for node in order:
        s = slot[node]
        bitsets[s, :] = 0
        if delay_index[node] >= 0:
            j = delay_index[node]
            bitsets[s, j >> 6] |= one << np.uint64(j & 63)

        for k in range(pred_ptr[node], pred_ptr[node + 1]):
            bitsets[s, :] |= bitsets[slot[pred_idx[k]], :]

        if prefer_index[node] >= 0:
            for j in range(m):
                dependency_matrix[prefer_index[node], j] = (bitsets[s, j >> 6] >> np.uint64(j & 63)) & one

    return dependency_matrix
