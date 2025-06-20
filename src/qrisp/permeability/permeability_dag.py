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

from builtins import id

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from qrisp.circuit import (
    QuantumCircuit,
    fast_append,
    Instruction,
    Qubit,
    QubitDealloc,
    QubitAlloc,
    ControlledOperation,
)
from qrisp.permeability.type_checker import is_permeable


def get_perm_dic(gate):
    """
    This function takes an Operation object and returns a dictionary indicating the
    permeability status of that Operation on each of it's qubits.

    Parameters
    ----------
    gate : Operation
        The Operation object to investigate.

    Returns
    -------
    dict
        A dictionary of the type {qubit_index : permeability_type}.

    """

    # If the gate is known to be X-permeable return the corresponding dict
    if gate.name in ["x", "rx", "rxx"]:
        return {i: "X" for i in range(gate.num_qubits)}

    # This dictionary will contain the result
    res = {}

    # The strategy is now to determine Z-permeability with the established function
    for i in range(gate.num_qubits):
        if is_permeable(gate, [i]):
            res[i] = "Z"
        else:
            res[i] = "neutral"

    # If the gate is a controlled operation, we can determine based on the permeability of
    # the base operation
    if isinstance(gate, ControlledOperation):
        base_op_perm = get_perm_dic(gate.base_operation)
        for i in range(gate.base_operation.num_qubits):
            res[i + gate.num_qubits - gate.base_operation.num_qubits] = base_op_perm[i]

    # The following code documents a general way of determining X-permeability, however
    # it requires the unitary of the gate, which can be expensive.
    # By not doing this, we miss out on permeability information in some situations
    # but have significantly fast uncomputation.
    return res

    # To check for X-permeability in general, we wrap the gate in H-gates and
    # determine Z permeability.
    qc = QuantumCircuit(gate.num_qubits)
    for qb in qc.qubits:
        qc.h(qb)
    qc.append(gate, qc.qubits)
    for qb in qc.qubits:
        qc.h(qb)

    x_perm_gate = qc.to_gate()

    for i in range(gate.num_qubits):
        if res[i] == "Z":
            continue
        elif is_permeable(x_perm_gate, [i]):
            res[i] = "X"

    return res


# This is the base class of the Nodes that appear in the Permeability DAG
class UnqompNode:
    def __init__(self, name, instr=None):
        self.name = name
        self.uncomputed_node = None
        self.instr = instr
        self.hash = id(self)

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return self.hash == other.hash


# Node which describes allocations.
# The artificial attribute indicates, whether the Allocation node has been added
# by the DAG construction function or whether it has been in the given QuantumCircuit
class AllocNode(UnqompNode):
    def __init__(self, instr, artificial=False):
        name = "alloc_" + instr.qubits[0].identifier
        UnqompNode.__init__(self, name, instr)
        self.artificial = artificial

    def __repr__(self):
        return r"$|0 \rangle$"


# Node describing deallocations.
class DeallocNode(UnqompNode):
    def __init__(self, instr):
        name = "dealloc_" + instr.qubits[0].identifier
        UnqompNode.__init__(self, name, instr)

    def __repr__(self):
        return r"$\langle 0|$"


# Node describing terminations.
class TerminatorNode(UnqompNode):
    def __init__(self, qubit):
        name = "terminate_" + qubit.identifier
        UnqompNode.__init__(self, name)
        self.qubit = qubit

    def __repr__(self):
        return "T"


# Node describing instructions
class InstructionNode(UnqompNode):
    def __init__(self, instr):
        name = instr.op.name
        UnqompNode.__init__(self, name, instr=instr)

    def __repr__(self):
        return self.instr.op.name


# This class represents the Permeability Graph
# As a subclass of networkx's DiGraph, it can be processed by
# many of networkx's algorithms.
class PermeabilityGraph(nx.DiGraph):

    def __init__(self, qc, remove_artificials=False):

        nx.DiGraph.__init__(self)

        # The recent_node_dic attribute is a dictionary of the type
        # {Qubit: UnqompNode}
        # which describes which nodes represent the
        # operations that have been carried ot most
        # recently on the corresponding Qubit.
        self.recent_node_dic = dag_from_qc(
            self, qc, remove_artificials=remove_artificials
        )

    # Visualizes the DAG
    def draw(self, layout_seed=None):
        """
        Visualizes the PermeabilityGraph.

        Parameters
        ----------
        layout_seed : int, optional
            A seed for the RNG which can (slightly) influence the visualisation layout if the Graph is arranged in an inconvenient way. The default is None.

        """
        visualize_dag(self, layout_seed)

    def to_qc(self, topo_sort=None):
        """
        Turns the PermeabilityGraph into a QuantumCircuit, using a given topological sort algorithm.

        Parameters
        ----------
        topo_sort : callable, optional
            The function to perform the topological sorting. By default, networkx.topological_sort is used.

        Returns
        -------
        res_qc : TYPE
            DESCRIPTION.

        """

        if topo_sort is None:
            topo_sort = nx.topological_sort
        res_qc = self.original_qc.clearcopy()

        with fast_append():
            for node in topo_sort(self):
                if node.instr:
                    if isinstance(node, AllocNode) and node.artificial:
                        continue
                    res_qc.append(node.instr.op, node.instr.qubits, node.instr.clbits)

        return res_qc

    def add_edge(self, in_node, out_node, edge_type, qubits=[]):
        """
        Adds an edge (in-place) to the PermeabilityGraph.

        Parameters
        ----------
        in_node : UnqompNode
            The node the edge should start at.
        out_node : UnqompNode
            The node the edge should start at.
        edge_type : str
            A string describing the edge type. Should be either 'X', 'Z', 'neutral' or 'anti_dependency'.
        qubits : list[Qubit], optional
            The list of Qubit the edge connects (only applicable if the edge is not anti-dependency).
            If the same edge is added multiple times with differing qubits, the qubits are accumulated.
            The default is [].

        """

        if in_node == out_node:
            return

        if self.has_edge(in_node, out_node):
            nx.DiGraph.get_edge_data(self, in_node, out_node)["qubits"].extend(qubits)
        else:
            nx.DiGraph.add_edge(
                self, in_node, out_node, edge_type=edge_type, qubits=list(qubits)
            )

    def get_target_qubits(self, node):
        """
        Returns the target qubits of a node, i.e. the qubits that have either neutral or X-permeability.

        Parameters
        ----------
        node : UnqompNode
            The node to be investigated.

        Raises
        ------
        Exception
            Tried to retrieve target qubits of TerminatorNode.

        Returns
        -------
        list[Qubit]
            The list of target qubits.

        """
        target_qubits = []

        if isinstance(node, AllocNode):
            return list(node.instr.qubits)

        in_edges = self.in_edges(node, data=True)
        for in_node, _, data in in_edges:
            if data["edge_type"] != "Z":
                target_qubits.extend(data["qubits"])

        return target_qubits

    def get_control_qubits(self, node):
        """
        Returns the control qubits of a node, i.e. the qubits that have Z-permeability.

        Parameters
        ----------
        node : UnqompNode
            The node to be investigated.

        Raises
        ------
        Exception
            Tried to retrieve control qubits of TerminatorNode.

        Returns
        -------
        list[Qubit]
            The list of control qubits.

        """
        control_qubits = []

        if isinstance(node, TerminatorNode):
            raise Exception("Tried to retrieve target qubits of TerminatorNode")

        if isinstance(node, AllocNode):
            return list(node.instr.qubits)

        in_edges = self.in_edges(node, data=True)
        for in_node, _, data in in_edges:
            if data["edge_type"] == "Z":
                control_qubits.extend(data["qubits"])

        return control_qubits

    def get_control_nodes(self, node):
        """
        Returns the list of control nodes of a node, i.e. the nodes that are connected via an
        outgoing edge with Z-permeability type.

        Parameters
        ----------
        node : UnqompNode
            The node to be investigated.

        Raises
        ------
        Exception
            Tried to retrieve target qubits of TerminatorNode.

        Returns
        -------
        list[UnqompNode]
            The list of control nodes.

        """
        control_nodes = []

        if isinstance(node, TerminatorNode):
            raise Exception("Tried to retrieve control qubits of TerminatorNode")

        in_edges = self.in_edges(node, data=True)
        for in_node, _, data in in_edges:
            if data["edge_type"] == "Z":
                control_nodes.append(in_node)

        return control_nodes

    def get_target_nodes(self, node):
        """
        Returns the list of control nodes of a node, i.e. the nodes that are connected via an
        outgoing edge with X-permeability or neutral type.

        Parameters
        ----------
        node : UnqompNode
            The node to be investigated.

        Raises
        ------
        Exception
            Tried to retrieve target qubits of TerminatorNode.

        Returns
        -------
        list[UnqompNode]
            The list of target nodes.

        """
        target_nodes = []

        if isinstance(node, TerminatorNode):
            raise Exception("Tried to retrieve control qubits of TerminatorNode")

        in_edges = self.in_edges(node, data=True)
        for in_node, _, data in in_edges:
            if data["edge_type"] != "Z":
                target_nodes.append(in_node)

        return target_nodes

    def get_edge_qubits(self, in_node, out_node):
        """
        Returns the list of qubits that a given edge is representing.

        Parameters
        ----------
        in_node : UnqompNode
            The starting node of the edge.
        out_node : UnqompNode
            The end node of the edge.

        Raises
        ------
        Exception
            Tried to retrieve edge qubits from Anti-Dependency edge.

        Returns
        -------
        list[Qubit]
            The list of Qubits.

        """
        return self.get_edge_data(in_node, out_node)["qubits"]

    def get_streak_children(self, node, qb):
        successors = self.successors(node)
        return [
            suc for suc in successors if qb in self.get_edge_data(node, suc)["qubits"]
        ]

    def get_edge_type(self, node_out, node_in):
        return self.get_edge_data(node_out, node_in)["edge_type"]

    def copy(self):
        self.__class__ = nx.DiGraph
        res = self.copy()
        self.__class__ = PermeabilityGraph
        return res


def dag_from_qc(dag, qc, remove_artificials=False):
    """
    This function receives an (empty) PermeabilityGraph and builds up the corresponding
    nodes/edges according to the given QuantumCircuit.

    Parameters
    ----------
    dag : PermeabilityGraph
        The empty PermeabilityGraph instance to operate on.
    qc : QuantumCircuit
        The QuantumCircuit to turn into the PermeabilityGraph.

    Returns
    -------
    recent_node_dic : dict[Qubit: UnqompNode]
        A dictionary describing which nodes represent the last instruction that was
        applied to a given Qubit.

    """

    # This dictionary keeps track of the most recent instruction that was
    # applied to a given Qubit
    recent_node_dic = {}

    # This dictionary tracks what kind of streak each Qubit is currently on.
    streak_dic = {}

    # This dictionary tracks a quantity called value layer. This is important for visualisation.
    # The idea behind it is to group each node into a layer. Nodes that are part of a streak
    # are grouped into the same layer.
    value_layer = {}

    artificial_init_nodes = []

    # We iterate through the QuantumCircuit and process each Instruction
    for i in range(len(qc.data)):

        # Set alias
        instr = qc.data[i]

        # We check whether the relevant Qubit already have an allocation node.
        for qb in instr.qubits:
            if qb not in recent_node_dic:

                # If the qubit has not been allocated yet but the first instruction being
                # executed is not an allocation, we insert and "artificial" allocation node
                is_artificial = instr.op.name != "qb_alloc"

                # Create the allocation node.
                alloc_node = AllocNode(
                    instr=Instruction(QubitAlloc(), [qb]), artificial=is_artificial
                )

                if alloc_node.artificial:
                    artificial_init_nodes.append(alloc_node)
                else:
                    alloc_node.instr = instr

                # Faster version of adding a node
                # dag.add_node(node)
                dag._succ[alloc_node] = {}
                dag._pred[alloc_node] = {}
                dag._node[alloc_node] = {}

                # Set the value layer attribute to 0
                alloc_node.value_layer = 0
                # alloc_node.qubit = qb

                # Set up the entry in the corresponding dictionaries
                recent_node_dic[qb] = alloc_node
                streak_dic[qb] = "neutral"
                value_layer[qb] = 1
                alloc_node.qc_index = 0

        # If the instruction is an allocation, we already processed it with the
        # code above
        if instr.op.name == "qb_alloc" and value_layer[qb] == 1:
            continue

        # This treats the case of a general Instruction
        elif instr.op.name != "qb_dealloc":
            node = InstructionNode(instr)

            # Get the permeability dictionary with the appropriate function
            perm_dic = get_perm_dic(instr.op)

        # This treats the case of a deallocation node
        else:
            node = DeallocNode(instr)
            perm_dic = {0: "neutral"}

        # We add the index of the corresponding gate to the node object,
        # because this information can be used for stable topological ordering:
        # The allocation algorithm performs a topological order based on the ancestors
        # of a certain nodes. The algorithm can subsequently use another topological
        # ordering to sort the ancestors. To make the sorting algorithm "stable" ie.
        # it preserves the previous order where possible, we use the indices of the source
        # circuit as sorting index.
        node.qc_index = i

        # Faster version of adding a node
        # dag.add_node(node)
        dag._succ[node] = {}
        dag._pred[node] = {}
        dag._node[node] = {}

        # To connect the edges, we iterate over each qubit
        for j in range(len(instr.qubits)):

            # Set alias
            qb = instr.qubits[j]

            # Retrieve edge typ
            edge_type = perm_dic[j]

            # Case streak is continued
            if streak_dic[qb] == edge_type and edge_type != "neutral":
                dag.add_edge(
                    recent_node_dic[qb], node, edge_type=edge_type, qubits=[qb]
                )

            # Case streak is terminated or edge_type is neutral
            else:

                # If the streak has more than one member, we insert a terminator node.
                # For that, we get the successors and filter out the edges that don't
                # describe the relevant qubit
                successors = list(dag.successors(recent_node_dic[qb]))
                streak_members = [
                    s
                    for s in successors
                    if qb in dag.get_edge_qubits(recent_node_dic[qb], s)
                ]

                # Insert the terminator if there are more than one streak_member to our qubit
                if len(streak_members) > 1:

                    # Create the TerminatorNode
                    terminator = TerminatorNode(qb)
                    # dag.add_node(terminator)
                    dag._succ[terminator] = {}
                    dag._pred[terminator] = {}
                    dag._node[terminator] = {}

                    # We now insert the anti-depedency edges from all streak members
                    # to the terminator node.

                    # To determine the value layer of the Terminator, we find the
                    # successor with the hight value layer and increase it by one.
                    value_layers = []
                    for s in streak_members:

                        # Add anti-depedency edge
                        dag.add_edge(
                            s, terminator, edge_type="anti_dependency", qubits=[qb]
                        )
                        value_layers.append(s.value_layer)

                    terminator.value_layer = max(value_layers) + 1
                    value_layer[qb] = terminator.value_layer

                    # Insert the terminator into the recen_node_dic
                    recent_node_dic[qb] = terminator

                # If there is only one member of the streak, the new recent_node
                # is that member
                elif len(streak_members) == 1:
                    recent_node_dic[qb] = streak_members[0]

                # Add the edge to the dag
                dag.add_edge(
                    recent_node_dic[qb], node, edge_type=edge_type, qubits=[qb]
                )

                # Increase the value layer
                value_layer[qb] += 1

                # Update the streak dic
                streak_dic[qb] = edge_type

            # The value layer of the node is the highest value layer of all qubits
            node.value_layer = max([value_layer[qb] for qb in instr.qubits])

    # Save the original_qc
    dag.original_qc = qc

    if remove_artificials:
        dag.remove_nodes_from(artificial_init_nodes)

    return recent_node_dic


def visualize_dag(G, layout_seed=None):
    """
    Visualizes a PermeabilityGraph, coloring edges and nodes according to their types.

    Parameters
    ----------
    G : PermeabilityGraph
        The PermeabilityGraph to visualize.
    layout_seed : int, optional
        An integer representing a random seed that can (slightly) modify the plot layout. The default is None.

    """

    # The subset_key dictionary groups the nodes according to their value_layers.
    # This dictionary will then be used to generate the plot layout via the nx.multipartite_layout
    subset_key = {}

    # Fill the dictionary
    for node in G.nodes():
        if node.value_layer in subset_key:
            subset_key[node.value_layer].append(node)
        else:
            subset_key[node.value_layer] = [node]

    # This function is a sort key to sort the nodes in each value layer
    # such that the plot has some resemblance with the QuantumCircuit
    def sort_key(x):
        if isinstance(x, TerminatorNode):
            return G.original_qc.qubits.index(x.qubit)
        else:
            return G.original_qc.qubits.index(x.instr.qubits[-1])

    # Sort the entries
    for key in subset_key.keys():
        subset_key[key].sort(key=sort_key)
        subset_key[key].reverse()

    # Determine the layout
    pos = nx.multipartite_layout(G, subset_key=subset_key)

    # We now modify the vertical positions of each node a bit.
    # We do this because the default layout can make it hard to distinguish
    # between edges crossing multiple layers.

    if layout_seed is not None:
        np.random.seed(layout_seed)

    rnd = np.random.random(1)
    for key in subset_key.keys():
        for node in subset_key[key]:
            # Multiply the y coordinates with some number
            pos[node][1] /= len(subset_key[key]) ** (0.5 + rnd)

    # Now that positions of the nodes are determined we can start plotting the Graph.
    # For this we first classify into (de)allocation nodes, instruction nodes and terminator nodes.
    allocation_nodes = []
    instruction_nodes = []
    terminator_nodes = []
    labels = {}

    for node in G.nodes():
        if isinstance(node, InstructionNode):
            instruction_nodes.append(node)
        elif isinstance(node, TerminatorNode):
            terminator_nodes.append(node)
        elif isinstance(node, DeallocNode):
            allocation_nodes.append(node)
        elif isinstance(node, AllocNode):
            allocation_nodes.append(node)

        # The node label is according to the __repr__ function
        labels[node] = node.__repr__()

    # Draw the nodes
    nx.draw_networkx_nodes(
        G, pos, nodelist=allocation_nodes, node_color="tab:blue", node_size=750
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=instruction_nodes, node_color="grey", node_size=750
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=terminator_nodes, node_color="orange", node_size=750
    )

    # Define node and edge colors
    edge_colors = {
        "Z": "green",
        "neutral": "grey",
        "anti_dependency": "purple",
        "X": "red",
    }
    # Draw edges with colors based on their type
    for edge_type, color in edge_colors.items():
        edges = [
            (u, v)
            for u, v, attr in G.edges(data=True)
            if attr.get("edge_type") == edge_type
        ]

        # Draw the edges
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            width=4,
            alpha=0.6,
            edge_color=color,
            arrowstyle="-|>",
            arrowsize=15,
        )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, labels=labels, font_color="white")

    # Deactivate frame
    plt.box(False)

    # plt.savefig("mm_dag.png", dpi = 300, bbox_inches = "tight")
    # Show plot
    plt.show()
