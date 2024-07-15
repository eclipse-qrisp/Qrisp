"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from qrisp.circuit import QuantumCircuit, fast_append, Instruction, Qubit, QubitDealloc, QubitAlloc
from qrisp.uncomputation.type_checker import is_permeable

def get_perm_dic(gate):
    
    res = {}
    for i in range(gate.num_qubits):
        if is_permeable(gate, [i]):
            res[i] = "Z"
        else:
            res[i] = "neutral"
            
    
    qc = QuantumCircuit(gate.num_qubits)
    
    qc.h(qc.qubits)
    qc.append(gate, qc.qubits)
    qc.h(qc.qubits)
    
    x_perm_gate = qc.to_gate()
    
    for i in range(gate.num_qubits):
        if res[i] == "Z":
            continue
        elif is_permeable(x_perm_gate, [i]):
            res[i] = "X"

    return res        

from builtins import id
class UnqompNode:
    def __init__(self, name, instr=None, node_counter = 0):
        self.name = name
        self.uncomputed_node = None
        self.instr = instr
        self.hash = id(self)

    def __hash__(self):
        return self.hash

    def __str__(self):
        if self.instr is None:
            return self.name
        else:
            return str(self.instr)

    def __eq__(self, other):
        return self.hash == other.hash

class AllocNode(UnqompNode):
    def __init__(self, instr):
        name = "alloc_" + instr.qubits[0].identifier
        UnqompNode.__init__(self, name, instr)
    
    def __repr__(self):
        return r"$\langle 0|$"
        
class DeallocNode(UnqompNode):
    def __init__(self, instr):
        name = "dealloc_" + instr.qubits[0].identifier
        UnqompNode.__init__(self, name, instr)
        
    def __repr__(self):
        return r"$|0 \rangle$"

class TerminatorNode(UnqompNode):
    def __init__(self, qb):
        name = "terminate_" + qb.identifier
        UnqompNode.__init__(self, name)
    
    def __repr__(self):
        return "T"

class InstructionNode(UnqompNode):
    def __init__(self, instr):
        name = instr.op.name
        UnqompNode.__init__(self, name, instr = instr)
        
    def __repr__(self):
        return self.instr.op.name

class PermeabilityGraph(nx.DiGraph):
    
    def __init__(self, qc):
        nx.DiGraph.__init__(self)
        dag_from_qc(self, qc)
        
    def draw(self, layout_seed = None):
        visualize_dag(self, layout_seed)
        
    def to_qc(self, topo_sort = None):
        
        if topo_sort is None:
            topo_sort = nx.topological_sort
        res_qc = self.original_qc.clearcopy()
        
        with fast_append():
            for node in topo_sort(dag):
                if node.instr:
                    res_qc.append(node.instr.op, node.instr.qubits, node.instr.clbits)
                    
        return res_qc


def dag_from_qc(dag, qc):

    recent_node_dic = {}
    streak_dic = {}
    value_layer = {}

    for i in range(len(qc.data)):
        instr = qc.data[i]
        
        for qb in instr.qubits:
            if qb not in recent_node_dic:
                alloc_node = AllocNode(Instruction(QubitAlloc(), [qb]))
                alloc_node.value_layer = 0
                alloc_node.qubit = qb
                dag.add_node(alloc_node)
                recent_node_dic[qb] = alloc_node
                streak_dic[qb] = "neutral"
                value_layer[qb] = 1
        
        if instr.op.name == "qb_alloc":
            continue
        elif instr.op.name != "qb_dealloc":
            node = InstructionNode(instr)
            perm_dic = get_perm_dic(instr.op)
        else:
            node = DeallocNode(instr)
            perm_dic = {0 : "neutral"}
        
        # We add the index of the corresponding gate to the node object,
        # because this information can be used for stable topological ordering:
        # The allocation algorithm performs a topological order based on the ancestors
        # of a certain nodes. The algorithm can subsequently use another topological
        # ordering to sort the ancestors. To make the sorting algorithm "stable" ie.
        # it preserves the previous order where possible, we use the indices of the source
        # circuit as sorting index.
        
        node.qc_index = i

        # dag.add_node(node)
        dag._succ[node] = {}
        dag._pred[node] = {}
        dag._node[node] = {}
        
        
        
        for j in range(len(instr.qubits)):
            
            qb = instr.qubits[j]

            edge_type = perm_dic[j]

            if streak_dic[qb]:
                # Case streak is continued
                if streak_dic[qb] == edge_type and edge_type != "neutral":
                    if (
                        dag.has_edge(recent_node_dic[qb], node)
                        and dag.get_edge_data(recent_node_dic[qb], node)["edge_type"]
                        != "anti_dependency"
                    ):
                        dag.get_edge_data(recent_node_dic[qb], node)["qubits"].append(qb)
                    else:
                        dag.add_edge(
                            recent_node_dic[qb], node, edge_type=edge_type, qubits=[qb]
                        )
                # Case streak is terminated
                else:
                    
                    successors = list(dag.successors(recent_node_dic[qb]))
                    if len(successors) > 1:
                        terminator = TerminatorNode(qb)
                        dag.add_node(terminator)
                        
                        value_layers = []
                        for s in successors:
                            dag.add_edge(s, terminator, edge_type="anti_dependency")
                            value_layers.append(s.value_layer)
                            
                        terminator.value_layer = max(value_layers) + 1
                        terminator.qubit = qb
                        value_layer[qb] += terminator.value_layer
                            
                        recent_node_dic[qb] = terminator
                    elif len(successors) == 1:
                        recent_node_dic[qb] = successors[0]                        
                    
                    dag.add_edge(
                        recent_node_dic[qb], node, edge_type=edge_type, qubits=[qb]
                    )
                    value_layer[qb] += 1
                    streak_dic[qb] = edge_type
            else:
                
                dag.add_edge(
                    recent_node_dic[qb], node, edge_type=edge_type, qubits=[qb]
                )
                recent_node_dic[qb] = node
                    
            node.value_layer = max([value_layer[qb] for qb in instr.qubits])

    dag.original_qc = qc
    return dag

def visualize_dag(G, layout_seed = None):
    """
    Visualize a DAG with NetworkX, coloring edges and nodes according to their types.

    Parameters:
    - G: NetworkX DiGraph object
    - edge_colors: dictionary with edge types as keys and colors as values
    - node_colors: dictionary with node types as keys and colors as values
    - pos: position dictionary for nodes (optional)
    - node_size: size of the nodes
    - font_size: size of the node labels
    """
    
    # Ensure it's a directed acyclic graph
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The input graph is not a directed acyclic graph (DAG).")

    subset_key = {}
    for node in G.nodes():
        if node.value_layer in subset_key:
            subset_key[node.value_layer].append(node)
        else:
            subset_key[node.value_layer] = [node]
    
    def sort_key(x):
        if isinstance(x, TerminatorNode):
            return G.original_qc.qubits.index(x.qubit)
        else:
            return G.original_qc.qubits.index(x.instr.qubits[-1])
    for key in subset_key.keys():
        subset_key[key].sort(key = sort_key)
        subset_key[key].reverse()
            
    pos = nx.multipartite_layout(G, subset_key = subset_key)
    
    if layout_seed is not None:
        np.random.seed(layout_seed)
    
    rnd = np.random.random(1)
    for key in subset_key.keys():
        for node in subset_key[key]:
            pos[node][1] /= len(subset_key[key])**(0.5+rnd)

    
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
        labels[node] = node.__repr__()
    
    
    nx.draw_networkx_nodes(G, pos, nodelist=allocation_nodes, node_color="tab:blue", node_size=750)
    nx.draw_networkx_nodes(G, pos, nodelist=instruction_nodes, node_color="grey", node_size=750)
    nx.draw_networkx_nodes(G, pos, nodelist=terminator_nodes, node_color="orange", node_size=750)

    # Define node and edge colors
    edge_colors = {'Z': 'green', "neutral": 'grey', "anti_dependency" : "purple", "X" : "red"}
    # Draw edges with colors based on their type
    for edge_type, color in edge_colors.items():
        edges = [(u, v) for u, v, attr in G.edges(data=True) if attr.get('edge_type') == edge_type]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            width=4,
            alpha=0.6,
            edge_color=color,
            arrowstyle = "-|>",
            arrowsize = 15
        )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, labels = labels, font_color = "white")

    # Add legend
    plt.box(False)
    # Show plot
    plt.show()

