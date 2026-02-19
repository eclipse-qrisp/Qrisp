"""
********************************************************************************
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
********************************************************************************
"""

# Modification to Kahns algorithm to reduce the depth of general circuits by applying
# trivial and non-trivial commutation relations based on the dag representation of
# unqomp.

import networkx as nx
import numpy as np
from numba import njit

from qrisp.permeability import PermeabilityGraph, TerminatorNode

# The following code aims to represent quantum circuits as an array of integers.
# The idea is here that in a 6 qubit quantum circuit, a gate that is executed
# on the qubits 1 and 3 is represented by the number 001010 = 5 (ie. the digits
# with significance 2**1 and 2**3 are set to 1).
# This construction might remove some of the information (ie. gate type or qubit order)
# but allows efficient processing since, the quantum circuit can be given to numba
# as a numpy array.

# Unfortunately, numpy integers can only represent 64 bit numbers. This means,
# that this technique could only handle quantum circuits with up to 64 qubits.
# We therefore SPLIT the number in the following way:
# The qubits from 0 to 63 represented by a numpy int,
# the qubits from 64 to 127 are represented by another numpy int,
# the qubits from 128 to 255 are represented by another numpy int,
# etc.
# We therefore represent the quantum circuit with a two-index numpy array
# where the first index indicates which gate is described and the second index
# indicates what qubit range is meant.


# This function receives the indices of the qubits of a gate and turns them into
# the list according to the above construction. The parameter k indicates
# how many integers the result should contain.
def split_integer(digit_list, k):

    digit_list.sort()

    if k <= 0:
        raise ValueError("Parameter k must be greater than zero")

    result = [0] * k
    i = 0
    for d in digit_list:
        while d // 64 > i:
            i += 1
        result[i] |= 1 << (d % 64)
        if result[i] >= 1 << 64:
            raise

    return result


# This function receives a list of qubits and a quantum circuit and turns it into
# a list according to the above construction
def qb_set_to_int(qubits, index_dict):
    qb_indices = [index_dict[qb] for qb in qubits]
    return split_integer(qb_indices, int(np.ceil(len(index_dict) / 64)))


# This function reverses the split_integer function (usefull for debugging)
def reverse_split(split_int, num_qubits):

    res = []
    for i in range(num_qubits):
        if not i % 64:
            current_partial_int = split_int.pop(0)

        if (current_partial_int) & (1 << (i % 64)):
            res.append(i)

    return res


# This function performs the parallelization. As described above, the idea is to
# build up the DAG from the quantum circuit and determine a linearization, which
# is in-turn used to achieve a more optimal quantum circuit.
# The linearization is performed using a modified version of Kahn's algorithm.
# This modification is informed about the depth of the circuit because of the above
# constructions.
def parallelize_qc(qc, depth_indicator=None):
    if len(qc.data) <= 1:
        return qc.copy()

    if depth_indicator is None:
        depth_indicator = lambda x: 1

    dag = PermeabilityGraph(qc, remove_artificials=True)
    # dag = dag_from_qc(qc, True)

    sprs_mat = nx.to_scipy_sparse_array(dag, format="csr")

    node_list = list(dag.nodes())

    # This list will contain the participating qubits of each gate in the above
    # discussed representation
    qubit_ints = []

    # This list will contain the depth of each gate, which can be specified via
    # the depth indicator function.
    depth_indicators = []

    index_dict = {qc.qubits[i]: i for i in range(len(qc.qubits))}

    for n in node_list:
        if not isinstance(n, TerminatorNode):
            qubit_ints.append(qb_set_to_int(n.instr.qubits, index_dict))
            depth_indicators.append(depth_indicator(n.instr.op))
        else:
            qubit_ints.append(qb_set_to_int([n.qubit], index_dict))
            depth_indicators.append(0)

    # Convert to array
    qubit_ints = np.array(qubit_ints, dtype=np.uint64)

    # Execute topological sort
    res = depth_sensitive_topological_sort_jitted(
        sprs_mat.indices,
        sprs_mat.indptr,
        qubit_ints,
        num_qubits=qc.num_qubits(),
        depth_indicators=np.array(depth_indicators),
    )

    # Build new circuit
    qc_new = qc.clearcopy()

    for i in range(len(res)):
        if not isinstance(node_list[res[i]], TerminatorNode):
            qc_new.append(node_list[res[i]].instr)

    return qc_new


# Kahns Algorithm based on
# https://www.geeksforgeeks.org/topological-sorting-indegree-based-solution/
def depth_sensitive_topological_sort(
    indices, indptr, int_qc, num_qubits, depth_indicators
):
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
    max_time = np.max(depth_indicators)
    # One by one dequeue vertices from queue and enqueue
    # adjacents if indegree of adjacent becomes 0

    while queue:
        # The depth sensitive part is now to deque the node with the least depth
        node_costs = np.zeros(len(queue))

        for i in range(len(queue)):
            node = queue[i]

            qubits = int_qc[node]

            depth_list = []
            for j in range(num_qubits):
                if int(qubits[j // 64]) & (1 << (j % 64)):
                    depth_list.append(depths[j])

            # If multiple gates have the same max depth, the faster ones should
            # be executed first, because they might block other gates
            depth_array = np.array(depth_list)
            node_costs[i] = np.max(depth_array) + depth_indicators[node] / 10**8

            # Multiple possible heuristics
            node_costs[i] = (
                np.max(depth_array)
                + depth_indicators[node] / 10**8
                - np.min(depth_array) / 10**12
            )
            # node_costs[i] = np.sum((np.max(depth_array) + depth_indicators[node]) - depth_array)/num_qubits
            # node_costs[i] = depth_indicators[node]/1E8 + np.sum((np.max(depth_array) + depth_indicators[node]) - depth_array)/1E8
            # node_costs[i] = depth_indicators[node]/1E8 + np.sum((np.max(depth_array) + depth_indicators[node]) - depth_array)*len(depth_list)
            # node_costs[i] = depth_indicators[node]/1E8 + np.sum((np.max(depth_array) + depth_indicators[node]) - depth_array)*len(depth_list)
            # a = 10
            # b = 1

            # node_costs[i] = depth_indicators[node] * a * max_time + b * np.sum(
            #     (np.max(depth_array) + depth_indicators[node]) - depth_array
            # )

        u = queue.pop(np.argmin(node_costs))

        top_order[cnt] = u

        # Update depths array
        max_depth = 0

        for i in range(num_qubits):
            if int_qc[u, i // 64] & 1 << (i % 64):
                if depths[i] > max_depth:
                    max_depth = depths[i]

        for i in range(num_qubits):
            if int_qc[u, i // 64] & 1 << (i % 64):
                depths[i] = max_depth + depth_indicators[u]

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
