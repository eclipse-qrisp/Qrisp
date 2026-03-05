"""
********************************************************************************
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

import networkx as nx
import numpy as np


def nx_get_causal_graph(
    qc, inverted=False, get_non_unitary_nodes=False, preferential_gates=[]
):
    from qrisp.circuit import Instruction, Operation

    # Create graph object
    G = nx.DiGraph()

    # This distionary contains the information, which node of the graph
    # if the most up to date noce for a qubit
    current_node_qubits = {}
    current_node_clbits = {}

    # List to collect the non-unitary nodes
    non_unitary_nodes = []

    # Now we traverse the circuit instructions
    for i in range(len(qc.data)):
        # Add new node to the graph
        new_node = i
        G.add_node(new_node)

        # Determine the set of nodes, which the new node will connect to
        # ie. the nodes which are most up to date on the particular qubits
        node_set = []
        for qb in qc.data[i].qubits:
            # If there is a node which has been connected to this qubit before,
            # append the node to the node set, that the new node will be connected to
            try:
                node_set.append(current_node_qubits[qb])
            except KeyError:
                pass

            # Update the dictionary
            current_node_qubits[qb] = new_node

        for cb in qc.data[i].clbits:
            # If there is a node which has been connected to this qubit before,
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
                G.add_edge(new_node, node)
        else:
            for node in node_set:
                G.add_edge(node, new_node)

        # Log if the new node is non unitary
        if qc.data[i].op.name in preferential_gates + ["final_op"]:
            non_unitary_nodes.append(new_node)

    # Return result
    if get_non_unitary_nodes:
        return G, non_unitary_nodes

    return G


# Function to reorder a circuit as described above
def nx_reorder_circuit(qc, preferential_gates=[]):
    from qrisp.circuit import Instruction, Operation

    for i in range(len(qc.qubits)):
        qc.append(Operation("final_op", num_qubits=1), [qc.qubits[i]])

    # Acquire causal graph
    G, non_unitary_nodes = nx_get_causal_graph(
        qc,
        inverted=True,
        get_non_unitary_nodes=True,
        preferential_gates=preferential_gates,
    )

    from networkx import descendants, topological_sort, transitive_reduction

    # We now order the non-unitary nodes according to how many decendents they all have
    # Measurements/Resets/Disentanglings with only a few descendants need only a few
    # gates to be simulated until the measurement can be executed
    node_costs = []
    for i in range(len(non_unitary_nodes)):
        if qc.data[hash(non_unitary_nodes[i])].op.name == "final_op":
            node_costs.append((non_unitary_nodes[i], np.inf))
        else:
            node_costs.append(
                (non_unitary_nodes[i], len(descendants(G, non_unitary_nodes[i])))
            )

    node_costs.sort(key=lambda x: x[1])

    # We now determine the topological dictionary of the nodes.
    # This dictionary assigns each node an integer, which indicates its position in a
    # topological ordering. The topological ordering of a graph has the feature that two
    # nodes N1, N2 which stand in a causal relationship to each other (i.e. N1 has to be
    # executed after N2) also have increasing index in the topological ordering i.e.
    # tp_dic[N1] > tp_dic[N2]
    sorted_nodes = list(topological_sort(G))
    tp_dic = {sorted_nodes[i]: i for i in range(len(sorted_nodes))}

    # This function now performs depth first traversal of the given causal graph,
    # starting at node. Each time a node is visited the callback function is called

    # The callback function that this will mainly be used with, is a logger,
    # which notes in which order the nodes have been visited

    # As mentioned in the comment at the beginning of this file,
    # a regular depth first traversal basically allows picking any child
    # to traverse next. In our case, this can mess with the topological ordering
    # Therefore we allways traverse the child with the highest topological index

    def topological_desc_traversal(G, node, tp_dic, callback):
        node_list = [x for x in nx.descendants(G, node)]

        node_list.sort(key=lambda x: -tp_dic[x])

        for n in node_list:
            callback(n)
            G.remove_node(n)

        callback(node)
        G.remove_node(node)
        return

    def topological_df_traversal(G, node, tp_dic, callback):
        # Acquire list of children
        node_list = [x for x in G.neighbors(node)]

        # Sort according to their topological index
        node_list.sort(key=lambda x: -tp_dic[x])

        # Recursively traverse the children
        while node_list:
            topological_df_traversal(G, node_list[0], tp_dic, callback)

            # Since the traversed nodes are removed after each traversal,
            # we need to to update the list of children, in case any of
            # the nodes in node_list have been removed
            node_list = [x for x in G.neighbors(node)]
            node_list.sort(key=lambda x: -tp_dic[x])

        # Call the callback function
        callback(node)
        # Remove the node in order to prevent traversing it again
        # this can happen because the causal graph is not necessarily a tree graph
        G.remove_node(node)

    # The circuits in this list will be the circuits whose execution is the absolute
    # minimum in order to evaluate a certain non-unitary operation
    new_qc_list = []

    # Now we succesively determine the minimal circuit required to execute
    # for each non-unitary operation
    while node_costs:

        # This node contains the non-unitary operation
        evaluation_node = node_costs.pop(0)[0]

        # This list will contain the unitary operations that are necessary
        # in order to perform the operation described by evaluation_node
        evaluation_list = []

        # Create callback function
        def callback(x):
            evaluation_list.append(x)

        # Traverse causal graph
        topological_desc_traversal(G, evaluation_node, tp_dic, callback)
        # topological_df_traversal(G, evaluation_node, tp_dic, callback)

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
    new_qc.data = sum([qc.data for qc in new_qc_list], [])

    # Remove final_op operations
    for i in range(len(qc.qubits)):
        qc.data.pop(-1)

    # Return result
    return new_qc


# Similar function as above but implemented for the C++ based
# graph theory package networkit
def nk_reorder_circuit(qc, preferential_gates=[]):
    from qrisp.circuit import Instruction, Operation

    for i in range(len(qc.qubits)):
        qc.append(Operation("final_op", num_qubits=1), [qc.qubits[i]])

    import networkit as nk

    G = nk.Graph(directed=True)
    current_node_qubits = {qubit: "-" for qubit in qc.qubits}
    current_node_clbits = {clbit: "-" for clbit in qc.clbits}
    non_unitary_nodes = []
    measurement_counter = 0
    for i in range(len(qc.data)):
        new_node = G.addNode()
        node_set = []
        for qb in qc.data[i].qubits:
            if current_node_qubits[qb] != "-":
                node_set.append(current_node_qubits[qb])

            current_node_qubits[qb] = new_node

        for cb in qc.data[i].clbits:
            if current_node_clbits[cb] != "-":
                node_set.append(current_node_clbits[cb])

            current_node_clbits[qb] = new_node

        node_set = list(set(node_set))

        for node in node_set:
            G.addEdge(i, node)

        if qc.data[i].op.name in preferential_gates + ["final_op"]:
            non_unitary_nodes.append(new_node)
            if qc.data[i].op.name == "measure":
                measurement_counter += 1

    from networkit.traversal import Traversal

    new_qc_list = []

    from networkit.graphtools import GraphTools

    sorted_nodes = GraphTools.topologicalSort(G)

    tp_dic = {sorted_nodes[i]: i for i in range(len(sorted_nodes))}

    reach_alg = nk.reachability.ReachableNodes(G, exact=True)

    reach_alg.run()
    node_costs = []

    for i in range(len(non_unitary_nodes)):
        if qc.data[hash(non_unitary_nodes[i])].op.name == "final_op":
            node_costs.append((non_unitary_nodes[i], np.inf))
        else:
            node_costs.append(
                (
                    non_unitary_nodes[i],
                    reach_alg.numberOfReachableNodes(non_unitary_nodes[i]),
                )
            )

    node_costs.sort(key=lambda x: x[1])

    def topological_desc_traversal(G, node, tp_dic, callback):
        node_list = []

        nk.traversal.Traversal.DFSfrom(G, node, node_list.append)

        node_list.sort(key=lambda x: -tp_dic[x])

        for n in node_list:
            callback(n)
            G.removeNode(node)

        # callback(node)
        # G.removeNode(node)

        return

    def topological_df_traversal(G, node, tp_dic, callback):
        node_list = [x for x in G.iterNeighbors(node)]
        node_list.sort(key=lambda x: -tp_dic[x])

        while node_list:
            topological_df_traversal(G, node_list[0], tp_dic, callback)
            node_list = [x for x in G.iterNeighbors(node)]
            node_list.sort(key=lambda x: -tp_dic[x])

        callback(node)
        G.removeNode(node)

    while node_costs:
        evaluation_node = node_costs.pop(0)[0]

        evaluation_list = []

        def callback(x, y=0):
            if len(evaluation_list):
                G.removeNode(evaluation_list[-1])
            evaluation_list.append(x)

        new_qc = qc.clearcopy()

        # topological_df_traversal(G, evaluation_node, tp_dic, callback)
        topological_desc_traversal(G, evaluation_node, tp_dic, callback)

        for node in evaluation_list:
            instr = qc.data[hash(node)]
            if instr.op.name != "final_op":
                new_qc.data.append(instr)

        new_qc_list.append(new_qc)

    for i in range(len(qc.qubits)):
        qc.data.pop(-1)

    new_qc = qc.clearcopy()
    new_qc.data = sum([qc.data for qc in new_qc_list], [])

    return new_qc


try:
    import networkit

    nk_available = True

except:
    nk_available = False
    # print("Install networkit for additional simulator performance.
    # Resorting to networkx.")


def reorder_circuit(qc, preferential_gates=[]):
    if nk_available:
        # return nk_reorder_circuit(qc, preferential_gates)

        return nx_reorder_circuit(qc, preferential_gates)
    else:
        return nx_reorder_circuit(qc, preferential_gates)
