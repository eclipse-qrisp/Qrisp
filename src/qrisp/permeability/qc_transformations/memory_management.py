"""
\********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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
********************************************************************************/
"""

import networkx as nx
import numpy as np
from numba import njit
import psutil


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

    for n in topological_sort(
        G, prefer=dealloc_identifier, delay=alloc_identifer, sub_sort=sub_sort
    ):
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
            alloc_qubits = list(
                set(delayed_qubit_alloc_dic.keys()).intersection(instr.qubits)
            )
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
    """
    Function to perform a topological sort on an Unqomp DAG which allows preferring/
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


memory = psutil.virtual_memory().total


def toposort_helper(indptr, indices, node_amount, delay_nodes, prefered_nodes):

    if memory / 4 < node_amount**2:
        return toposort_helper_sparse(
            indptr, indices, node_amount, delay_nodes, prefered_nodes
        )
    else:
        return toposort_helper_dense(
            indptr, indices, node_amount, delay_nodes, prefered_nodes
        )


@njit(cache=True)
def toposort_helper_dense(indptr, indices, node_amount, delay_nodes, prefered_nodes):
    # This array returns a graph that reflects all ancestor relations
    # i.e. ancestor_graph[42] is True at all ancestors of node 42
    ancestor_graph = compute_all_ancestors_dense(indptr, indices, node_amount)

    n = prefered_nodes.size
    m = delay_nodes.size

    # This array will contain the ancestor relations between the
    # prefered/delay nodes
    dependency_matrix = np.zeros((n, m), dtype=np.int8)

    # Fill with information from ancestor_graph
    for i in range(n):
        for j in range(m):
            if ancestor_graph[prefered_nodes[i], delay_nodes[j]]:
                dependency_matrix[i, j] = 1

    # This array will contain the result
    res = np.zeros(node_amount, dtype=np.int32)

    # This array array tracks which nodes have not yet been processed.
    # It is initialized to all True because no nodes have been processed yet.
    remaining_nodes = np.ones(node_amount, dtype=np.int8)

    # This integer will contain the amount of nodes that have been processed
    node_counter = 0

    if m != 0:
        for i in range(n):
            # For each prefer nodes we compute how many delay nodes are required.
            required_delay_nodes = np.sum(dependency_matrix, axis=1)

            # We determine the prefer node that requires the least delay nodes
            min_node_index = np.argmin(required_delay_nodes)
            prefer_node = prefered_nodes[min_node_index]

            # We determine the ancestor nodes of this node that have
            # not been processed yet
            to_be_processed = ancestor_graph[prefer_node, :] & remaining_nodes

            ancestor_indices = np.nonzero(to_be_processed)[0]

            # We insert the nodes in the result array.
            # We can assume that order of the nodes induces by their numbering
            # is already a topological ordering. Therefore inserting them in
            # order is also a topological sub sort.
            res[node_counter : node_counter + len(ancestor_indices)] = ancestor_indices
            node_counter += len(ancestor_indices)

            # Mark the nodes as processed
            remaining_nodes[ancestor_indices] = 0

            # Update the depedency matrix: All delay nodes that have been processed
            # don't need to be considered again for all following iterations,
            # we therefore remove them from the other columns
            dependency_matrix = np.clip(
                dependency_matrix - dependency_matrix[min_node_index, :], 0, 1
            )

            # Finaly we set all nodes in the processed column to 1 so this column
            # is not processed again.
            dependency_matrix[min_node_index, :] = 1

    # Insert the remaining nodes
    res[node_counter:] = np.nonzero(remaining_nodes)[0]

    # return the result
    return res


@njit(cache=True)
def compute_all_ancestors_dense(indptr, indices, node_amount):
    # Initialize ancestor sets for all nodes
    ancestors = np.zeros((node_amount, node_amount), dtype=np.bool_)

    # Topological sort
    in_degree = np.zeros(node_amount, dtype=np.int64)
    for i in range(node_amount):
        for j in range(indptr[i], indptr[i + 1]):
            in_degree[indices[j]] += 1

    queue = [i for i in range(node_amount) if in_degree[i] == 0]

    while queue:
        node = queue.pop(0)
        ancestors[node, node] = True  # A node is its own ancestor

        for i in range(indptr[node], indptr[node + 1]):
            child = indices[i]
            # Add current node and its ancestors to child's ancestors
            ancestors[child, :] |= ancestors[node, :]
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    return ancestors


@njit(cache=True)
def compute_all_ancestors_sparse(indptr, indices, node_amount, prefered_nodes):
    # Initialize ancestor sets for all nodes using a list of sets
    ancestors = [
        np.zeros(1, dtype=np.int8) for i in np.arange(node_amount, dtype=np.int32)
    ]

    # Topological sort
    in_degree = np.zeros(node_amount, dtype=np.int64)

    for i in range(node_amount):
        for j in range(indptr[i], indptr[i + 1]):
            in_degree[indices[j]] += 1

    queue = []
    for i in range(node_amount):
        if in_degree[i] == 0:
            queue.append(i)
            ancestors[i] = np.zeros(node_amount, dtype=np.int8)
            ancestors[i][i] = 1

    keep_result = np.zeros(node_amount, dtype=np.int32)

    for i in range(len(prefered_nodes)):
        keep_result[prefered_nodes[i]] = 1

    while queue:
        node = queue.pop(0)

        curr_anc = ancestors[node]

        for i in range(indptr[node], indptr[node + 1]):
            child = indices[i]
            if len(ancestors[child]) != node_amount:
                ancestors[child] = np.zeros(node_amount, dtype=np.int8)
                ancestors[child][child] = 1

            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

        for i in range(indptr[node], indptr[node + 1]):
            child = indices[i]
            # Add current node and its ancestors to child's ancestors
            ancestors[child] = curr_anc | ancestors[child]

        # Delete the non-needed values
        if not keep_result[node]:
            ancestors[node] = np.zeros(1, dtype=np.int8)

    return ancestors


@njit(cache=True)
def toposort_helper_sparse(indptr, indices, node_amount, delay_nodes, prefered_nodes):
    # This array returns a graph that reflects all ancestor relations
    # i.e. ancestor_graph[42] is True at all ancestors of node 42
    ancestor_graph = compute_all_ancestors_sparse(
        indptr, indices, node_amount, prefered_nodes
    )

    n = prefered_nodes.size
    m = delay_nodes.size

    # This array will contain the ancestor relations between the
    # prefered/delay nodes
    dependency_matrix = np.zeros((n, m), dtype=np.int8)

    # Fill with information from ancestor_graph
    for i in range(n):
        for j in range(m):
            if ancestor_graph[prefered_nodes[i]][delay_nodes[j]]:
                dependency_matrix[i, j] = 1

    # This array will contain the result
    res = np.zeros(node_amount, dtype=np.int32)

    # This array array tracks which nodes have not yet been processed.
    # It is initialized to all True because no nodes have been processed yet.
    remaining_nodes = np.ones(node_amount, dtype=np.int8)

    # This integer will contain the amount of nodes that have been processed
    node_counter = 0

    if m != 0:
        for i in range(n):
            # For each prefer nodes we compute how many delay nodes are required.
            required_delay_nodes = np.sum(dependency_matrix, axis=1)

            # We determine the prefer node that requires the least delay nodes
            min_node_index = np.argmin(required_delay_nodes)
            prefer_node = prefered_nodes[min_node_index]

            # We determine the ancestor nodes of this node that have
            # not been processed yet
            to_be_processed = ancestor_graph[prefer_node] & remaining_nodes

            ancestor_indices = np.nonzero(to_be_processed)[0]

            # We insert the nodes in the result array.
            # We can assume that order of the nodes induces by their numbering
            # is already a topological ordering. Therefore inserting them in
            # order is also a topological sub sort.
            res[node_counter : node_counter + len(ancestor_indices)] = ancestor_indices
            node_counter += len(ancestor_indices)

            # Mark the nodes as processed
            remaining_nodes[ancestor_indices] = 0

            # Update the depedency matrix: All delay nodes that have been processed
            # don't need to be considered again for all following iterations,
            # we therefore remove them from the other columns
            dependency_matrix = np.clip(
                dependency_matrix - dependency_matrix[min_node_index, :], 0, 1
            )

            # Finaly we set all nodes in the processed column to 1 so this column
            # is not processed again.
            dependency_matrix[min_node_index, :] = 1

    # Insert the remaining nodes
    res[node_counter:] = np.nonzero(remaining_nodes)[0]

    # return the result
    return res
