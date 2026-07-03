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

# -*- coding: utf-8 -*-

# The functions in this file are supposed to reorder circuits in such a way,
# that measurements, resets and disentanglers are performed as early as possible.
# This is beneficial because the simulation of two decoherent states can be parallelized
# better. Furthermore, might the measurement of these imply that one of the states has
# vanishing probability, which means that this state does not require further
# simulation.

# The idea to achieve the ordering is to convert the circuit into a directed acyclic
# graph (now called causal graph), where two successive operations with overlapping
# qubits are represented with two nodes, connected by a directed edge. The directed
# edge, however points into the opposite direction of the sequential order of the
# operations. This way, if we want to evaluate what gates are necessary in order to
# perform a measurement, disentangling or reset, we simply have to look at the set of
# gates that is reachable from that specific node in the causal graph.

# Consider the following circuit
#           ┌───┐
#  qubit_9: ┤ X ├──■─────
#           ├───┤┌─┴─┐┌─┐
# qubit_10: ┤ Y ├┤ x ├┤M├
#           ├───┤└───┘└╥┘
# qubit_11: ┤ H ├──────╫─
#           └───┘      ║
#  clbit_0: ═══════════╩═

# In order to perform the measurement, we have to execute the x and the cx gate
# The corresponding causal graph
# (measure)-> (CX)->(X)
#               L->(Y)
# (H)
# We see that CX, X and Y are reachable from the measurement
# We now perform a depth first traversal on the subgraph of the nodes which are
# reachable from the measurement.

# It is important to node that for a regular depth first traversal,
# we are free to choose which child of a node to traverse next.
# This is not the casehere, because this might disturb the topological
# order, which would imply that we performed an illegitimat reordering
# Therefore, in this depth first traversal, we always traverse the child
# with the highest index in a topological order first.

from __future__ import annotations

from collections import deque
import itertools
from typing import Any, Callable

import networkx as nx
from networkx import descendants, topological_sort
import numpy as np

from qrisp.circuit import Operation, QuantumCircuit


def reorder_circuit(qc: QuantumCircuit, preferential_gates: list[str] | None = None) -> QuantumCircuit:
    """
    Reorder the given quantum circuit based on a topological sorting of its causal graph.

    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit to be reordered.
    preferential_gates : list[str], optional
        List of gate names to prioritize during reordering. Defaults to None.

    Returns
    -------
    QuantumCircuit
        The reordered quantum circuit.
    """
    if preferential_gates is None:
        preferential_gates = []

    return nx_reorder_circuit(qc, preferential_gates)


def nx_get_causal_graph(
    qc: QuantumCircuit,
    inverted: bool = False,
    get_non_unitary_nodes: bool = False,
    preferential_gates: list[str] | None = None,
) -> nx.DiGraph | tuple[nx.DiGraph, list[int]]:
    """
    Generate the causal graph of a quantum circuit.

    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit for which to generate the causal graph.
    inverted : bool, optional
        If True, invert the direction of the edges in the causal graph. Defaults to False.
    get_non_unitary_nodes : bool, optional
        If True, return a list of non-unitary nodes in addition to the causal graph. Defaults to False.
    preferential_gates : list[str], optional
        List of gate names to prioritize during graph construction. Defaults to None.

    Returns
    -------
    nx.DiGraph
        The causal graph of the quantum circuit.
    tuple[nx.DiGraph, list[int]]
        If get_non_unitary_nodes is True, also return a list of non-unitary nodes.
    """

    if preferential_gates is None:
        preferential_gates = []

    # Create graph object
    graph = nx.DiGraph()

    # This distionary contains the information, which node of the graph
    # if the most up to date noce for a qubit
    current_node_qubits = {}
    current_node_clbits = {}

    # List to collect the non-unitary nodes
    non_unitary_nodes = []

    # Now we traverse the circuit instructions
    for new_node, instruction in enumerate(qc.data):
        graph.add_node(new_node)

        # Determine the set of nodes, which the new node will connect to
        # ie. the nodes which are most up to date on the particular qubits
        node_set = []
        for qb in instruction.qubits:
            # If there is a node which has been connected to this qubit before,
            # append the node to the node set, that the new node will be connected to
            try:
                node_set.append(current_node_qubits[qb])
            except KeyError:
                pass

            # Update the dictionary
            current_node_qubits[qb] = new_node

        for cb in instruction.clbits:
            # If there is a node which has been connected to this clbit before,
            # append the node to the node set, that the new node will be connected to
            try:
                node_set.append(current_node_clbits[cb])
            except KeyError:
                pass
            # Update the dictionary
            current_node_clbits[cb] = new_node

        # Make sure every node is listed only once
        node_set = list(set(node_set))

        # Add the edges
        if inverted:
            for node in node_set:
                graph.add_edge(new_node, node)
        else:
            for node in node_set:
                graph.add_edge(node, new_node)

        # Log if the new node is non unitary
        if instruction.op.name in preferential_gates + ["final_op"]:
            non_unitary_nodes.append(new_node)

    # Return result
    if get_non_unitary_nodes:
        return graph, non_unitary_nodes

    return graph


def nx_reorder_circuit(qc: QuantumCircuit, preferential_gates: list[str] | None = None) -> QuantumCircuit:
    """
    Reorder the given quantum circuit based on a topological sorting of its causal graph.

    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit to be reordered.
    preferential_gates : list[str], optional
        List of gate names to prioritize during reordering. Defaults to None.

    Returns
    -------
    QuantumCircuit
        The reordered quantum circuit.
    """

    if preferential_gates is None:
        preferential_gates = []

    for qb in qc.qubits:
        qc.append(Operation("final_op", num_qubits=1), [qb])

    # Acquire causal graph
    graph, non_unitary_nodes = nx_get_causal_graph(
        qc,
        inverted=True,
        get_non_unitary_nodes=True,
        preferential_gates=preferential_gates,
    )

    # We now order the non-unitary nodes according to how many decendents they all have
    # Measurements/Resets/Disentanglings with only a few descendants need only a few
    # gates to be simulated until the measurement can be executed
    node_costs = []
    for node in non_unitary_nodes:
        if qc.data[hash(node)].op.name == "final_op":
            node_costs.append((node, np.inf))
        else:
            node_costs.append((node, len(descendants(graph, node))))

    node_costs.sort(key=lambda x: x[1])

    # We now determine the topological dictionary of the nodes.
    # This dictionary assigns each node an integer, which indicates its position in a
    # topological ordering. The topological ordering of a graph has the feature that two
    # nodes N1, N2 which stand in a causal relationship to each other (i.e. N1 has to be
    # executed after N2) also have increasing index in the topological ordering i.e.
    # tp_dic[N1] > tp_dic[N2]
    sorted_nodes = list(topological_sort(graph))
    tp_dic = {sorted_nodes[i]: i for i in range(len(sorted_nodes))}

    # This function now performs depth first traversal of the given causal graph,
    # starting at node. Each time a node is visited the callback function is called

    # The callback function that this will mainly be used with, is a logger,
    # which notes in which order the nodes have been visited

    # As mentioned in the comment at the beginning of this file,
    # a regular depth first traversal basically allows picking any child
    # to traverse next. In our case, this can mess with the topological ordering
    # Therefore we allways traverse the child with the highest topological index

    visited_nodes = set()

    def topological_desc_traversal(
        graph: nx.DiGraph, node: int, tp_dic: dict[int, int], callback: Callable[[int], None]
    ) -> None:
        """
        Traverse all unvisited descendants of a given node in decreasing topological order.

        Unlike a standard recursive depth-first search, this function retrieves all
        descendants of the starting node at once, sorts them based on their
        topological index, and applies the callback function sequentially.#
        """

        if node in visited_nodes:
            return

        # Only fetch descendants that haven't been visited yet
        node_list = [x for x in nx.descendants(graph, node) if x not in visited_nodes]
        node_list.sort(key=lambda x: -tp_dic[x])

        for n in node_list:
            callback(n)

        callback(node)

    def topological_df_traversal(
        graph: nx.DiGraph, node: int, tp_dic: dict[int, int], callback: Callable[[int], None]
    ) -> None:
        """
        Perform a recursive depth-first traversal prioritizing higher topological indices.

        This function recursively visits the unvisited neighbors of a node. At each
        branching step, it sorts the neighbors so that nodes with a higher
        topological index are traversed first, preserving the intended causal order.
        """
        if node in visited_nodes:
            return

        # Get unvisited neighbors and sort by topological index
        neighbors = [x for x in graph.neighbors(node) if x not in visited_nodes]
        neighbors.sort(key=lambda x: -tp_dic[x])

        # Recursively traverse
        for n in neighbors:
            if n not in visited_nodes:
                topological_df_traversal(graph, n, tp_dic, callback)

        callback(node)

    # The circuits in this list will be the circuits whose execution is the absolute
    # minimum in order to evaluate a certain non-unitary operation
    new_qc_list = []

    # Now we succesively determine the minimal circuit required to execute
    # for each non-unitary operation
    node_costs_queue = deque(node_costs)
    while node_costs_queue:
        # This node contains the non-unitary operation
        evaluation_node = node_costs_queue.popleft()[0]

        # This list will contain the unitary operations that are necessary
        # in order to perform the operation described by evaluation_node
        evaluation_list = []

        def callback(x: int) -> None:
            # The callback acts as the gatekeeper, marking nodes as processed
            if x not in visited_nodes:
                evaluation_list.append(x)
                visited_nodes.add(x)

        # Traverse causal graph
        topological_desc_traversal(graph, evaluation_node, tp_dic, callback)
        # topological_df_traversal(graph, evaluation_node, tp_dic, callback)

        # Create circuit
        new_qc = qc.clearcopy()
        # Append the corresponding instruction to the circuit
        for node in evaluation_list:
            instr = qc.data[hash(node)]
            if not instr.op.name == "final_op":
                new_qc.data.append(instr)

        new_qc_list.append(new_qc)

    # Create result quantum circuit
    new_qc = qc.clearcopy()

    # Concatenate the data of the newly created circuits
    new_qc.data = list(itertools.chain.from_iterable(qc.data for qc in new_qc_list))

    # Remove final_op operations
    for i in range(len(qc.qubits)):
        qc.data.pop(-1)

    # Return result
    return new_qc
