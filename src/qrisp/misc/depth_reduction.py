"""
/*********************************************************************
* Copyright (c) 2023 the Qrisp Authors
*
* This program and the accompanying materials are made
* available under the terms of the Eclipse Public License 2.0
* which is available at https://www.eclipse.org/legal/epl-2.0/
*
* SPDX-License-Identifier: EPL-2.0
**********************************************************************/
"""


# Modification to Kahns algorithm to reduce the depth of general circuits by applying
# trivial and non-trivial commutation relations based on the dag representation of
# unqomp.

import networkx as nx
import numpy as np
from numba import njit

from qrisp.uncomputation.unqomp import dag_from_qc


def qb_set_to_int(qubits, qc):
    res = 0
    for qb in qubits:
        res |= 1 << qc.qubits.index(qb)
    return res


# Kahns Algorithm based on
# https://www.geeksforgeeks.org/topological-sorting-indegree-based-solution/


def depth_sensitive_topological_sort(indices, indptr, int_qc, num_qubits):
    # Create a vector to store indegrees of all
    # vertices. Initialize all indegrees as 0.
    n = len(indptr) - 1
    in_degree = np.zeros(n, dtype=np.int32)

    depths = np.zeros(num_qubits, dtype=np.int32)
    # Traverse adjacency lists to fill indegrees of
    # vertices.  This step takes O(V + E) time

    for i in range(n):
        for j in indices[indptr[i] : indptr[i + 1]]:
            in_degree[j] += 1

    # Create a queue and enqueue all vertices with
    # indegree 0
    queue = []

    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    # Initialize count of visited vertices
    cnt = 0

    # Create a vector to store result (A topological
    # ordering of the vertices)
    top_order = np.zeros(n, dtype=np.int32)

    # One by one dequeue vertices from queue and enqueue
    # adjacents if indegree of adjacent becomes 0

    while queue:
        # The depth sensitive part is now to deque the node with the least depth
        node_costs = np.zeros(len(queue))

        for i in range(len(queue)):
            node = queue[i]

            qubits = int_qc[node]

            depth_sum = 0
            for j in range(num_qubits):
                if qubits & 1 << j:
                    depth_sum += depths[j]

            node_costs[i] = depth_sum

        u = queue.pop(np.argmin(node_costs))

        top_order[cnt] = u

        # Update depths array
        max_depth = 0

        for i in range(num_qubits):
            if int_qc[u] & 1 << i:
                if depths[i] > max_depth:
                    max_depth = depths[i]

        for i in range(num_qubits):
            if int_qc[u] & 1 << i:
                depths[i] = max_depth + 1

        # Update in degree array
        for i in indices[indptr[u] : indptr[u + 1]]:
            in_degree[i] -= 1
            if in_degree[i] == 0:
                queue.append(i)

        cnt += 1

    return top_order


depth_sensitive_topological_sort_jitted = njit(cache=True)(
    depth_sensitive_topological_sort
)


def reduce_depth(qc):
    if len(qc.data) <= 1:
        return qc.copy()

    dag = dag_from_qc(qc, remove_init_nodes=True)

    sprs_mat = nx.to_scipy_sparse_array(dag, format="csr")

    node_list = list(dag.nodes())
    qubit_ints = [qb_set_to_int(n.instr.qubits, qc) for n in node_list]

    try:
        qubit_ints = np.array(qubit_ints, dtype=np.int64)
        res = depth_sensitive_topological_sort_jitted(
            sprs_mat.indices, sprs_mat.indptr, qubit_ints, num_qubits=qc.num_qubits()
        )
    except OverflowError:
        res = depth_sensitive_topological_sort(
            sprs_mat.indices, sprs_mat.indptr, qubit_ints, num_qubits=qc.num_qubits()
        )

    qc_new = qc.clearcopy()

    for i in range(len(res)):
        qc_new.append(node_list[res[i]].instr)

    return qc_new
