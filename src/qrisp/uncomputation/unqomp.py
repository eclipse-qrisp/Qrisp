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


import networkx as nx
from networkx import DiGraph

from qrisp.circuit import (
    ControlledOperation,
    PTControlledOperation,
    QuantumCircuit,

)
from qrisp.logic_synthesis import LogicSynthGate
from qrisp.uncomputation.type_checker import is_permeable, is_qfree
from qrisp.circuit import fast_append


class UnqompNode:
    def __init__(self, name, instr=None):
        self.name = name
        self.targets = []
        self.controls = []
        self.uncomputed_node = None

        self.instr = instr
        if instr is None:
            self.hash = hash(self.name)
        else:
            self.hash = hash(instr)

    def __hash__(self):
        return self.hash

    def __str__(self):
        if self.instr is None:
            return self.name
        else:
            return str(self.instr)

    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        return self.hash == other.hash


def qc_from_dag(dag):
    res_qc = dag.original_qc.clearcopy()
    from qrisp.core.compilation import topological_sort
        
    with fast_append():
        for node in nx.topological_sort(dag):
            if node.instr:
                res_qc.append(node.instr.op, node.instr.qubits, node.instr.clbits)

    return res_qc


def dag_from_qc(qc, remove_init_nodes=False):
    dag = DiGraph()

    recent_node_dic = {}
    node_counter = {}
    init_nodes = {}

    for i in range(len(qc.qubits)):
        node = UnqompNode("qubit_" + str(i) + "_0")

        node.qubit = qc.qubits[i]
        
        # dag.add_node(node)
        dag._succ[node] = {}
        dag._pred[node] = {}
        dag._node[node] = {}
        
        recent_node_dic[qc.qubits[i]] = node
        node_counter[qc.qubits[i]] = 1
        init_nodes[qc.qubits[i]] = node
    
    for i in range(len(qc.clbits)):
        
        node = UnqompNode("clbit_" + str(i) + "_0")

        node.clbit = qc.clbits[i]

        # dag.add_node(node)
        dag._succ[node] = {}
        dag._pred[node] = {}
        dag._node[node] = {}
        
        
        recent_node_dic[qc.clbits[i]] = node
        node_counter[qc.clbits[i]] = 1
        init_nodes[qc.clbits[i]] = node
        

    dag.init_nodes = init_nodes

    dealloc_nodes = []

    for i in range(len(qc.data)):
        instr = qc.data[i]

        # if instr.op.name == "barrier":
        # continue

        # node = UnqompNode(str(qc.qubits.index(instr.qubits[-1])) + "_"
        #   + str(node_counter[instr.qubits[-1]]), instr)
        node = UnqompNode(str(node_counter[instr.qubits[-1]]), instr)
        
        node.qc_index = i

        if instr.op.name == "qb_dealloc":
            dealloc_nodes.append(node)

        # dag.add_node(node)
        dag._succ[node] = {}
        dag._pred[node] = {}
        dag._node[node] = {}
        
        
        for j in range(len(instr.qubits)):
            qb = instr.qubits[j]

            if is_permeable(instr.op, [j]):
                edge_type = "control"
                node.controls.append(qb)
            else:
                edge_type = "target"

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

            successors = list(dag.successors(recent_node_dic[qb]))
            if len(successors) > 1:
                if edge_type == "target":
                    for s in successors:
                        if s == node:
                            continue
                        elif not dag.has_edge(s, node):
                            if (
                                dag.get_edge_data(recent_node_dic[qb], s)["edge_type"]
                                == "control"
                            ):
                                dag.add_edge(s, node, edge_type="anti_dependency")

            if edge_type == "target":
                recent_node_dic[qb] = node
                node_counter[qb] += 1
                node.targets.append(qb)
        for j in range(len(instr.clbits)):
            cb = instr.clbits[j]
            dag.add_edge(recent_node_dic[cb], node, edge_type="anti_dependency")
            recent_node_dic[cb] = node

    dag.original_qc = qc
    dag.recent_node_dic = recent_node_dic

    if remove_init_nodes:
        # Remove init nodes
        dag.remove_nodes_from(list(init_nodes.values()))

    return dag


def uncompute_node(dag, uncomp_node, uncomp_qbs, recompute_qubits=[]):
    from qrisp.core import topological_sort    

    a_star_n_list = [dag.recent_node_dic[qb] for qb in set(uncomp_node.targets)]
    a_star_n_list = list(set(a_star_n_list))
    
    if not is_qfree(uncomp_node.instr.op):
        raise Exception(f"Tried to uncompute non-qfree instruction {uncomp_node.instr}")

    ctrls = [
        n
        for n in dag.predecessors(uncomp_node)
        if dag.get_edge_data(n, uncomp_node)["edge_type"] == "control"
    ]

    for i in range(len(ctrls)):
        if not ctrls[i].uncomputed_node is None:
            ctrls[i] = ctrls[i].uncomputed_node

    # Replace controlled operations by phase tolerant controlled operations
    if isinstance(uncomp_node.instr.op, ControlledOperation):
        op = uncomp_node.instr.op

        if op.method == "auto" and len(op.controls) < 5 or op.method == "gray":
            if len(op.controls) != 1:
                uncomp_node.instr.op = PTControlledOperation(
                    op.base_operation,
                    num_ctrl_qubits=len(op.controls),
                    ctrl_state=op.ctrl_state,
                    method = "gray_pt"
                )

    # Replace results of logic synthesis by phase tolerant logic synthesis
    if isinstance(uncomp_node.instr.op, LogicSynthGate):
        if uncomp_node.instr.op.logic_synth_method == "gray":
            from qrisp import QuantumVariable

            tt = uncomp_node.instr.op.tt

            input_qv = QuantumVariable(tt.bit_amount)
            output_qv = QuantumVariable(tt.shape[1], qs=input_qv.qs)

            tt.q_synth(input_qv, output_qv, method="gray_pt")

            temp = input_qv.qs.data[-1].op
            temp.name = uncomp_node.instr.op.name
            
            uncomp_node.instr.op = temp

    new_instr = uncomp_node.instr.inverse()

    reversed_node = UnqompNode(uncomp_node.name + "_*", new_instr)
    reversed_node.controls = uncomp_node.controls
    reversed_node.targets = uncomp_node.targets

    uncomp_node.uncomputed_node = reversed_node

    dag.add_node(reversed_node)
    
    for qb in reversed_node.targets:
        dag.recent_node_dic[qb] = reversed_node

    for i in range(len(ctrls)):
        c = ctrls[i]

        control_edge_qubits = dag.get_edge_data(c, uncomp_node)["qubits"]

        if set(control_edge_qubits).intersection(recompute_qubits):
            for k in dag.successors(c):
                if dag.get_edge_data(c, k)["edge_type"] == "target" and set(
                    dag.get_edge_data(c, k)["qubits"]
                ).intersection(control_edge_qubits):
                    if k.uncomputed_node is None:
                        return True

                    ctrls[i] = k.uncomputed_node
                    break
            else:
                return True

        dag.add_edge(
            ctrls[i], reversed_node, edge_type="control", qubits=control_edge_qubits
        )

    for a_star_n in a_star_n_list:
        dag.add_edge(
            a_star_n,
            reversed_node,
            edge_type="target",
            qubits=list(
                set(uncomp_qbs).intersection(a_star_n.targets).intersection(new_instr.qubits)
            ),
        )

        for v in dag.successors(a_star_n):
            if dag.get_edge_data(a_star_n, v)["edge_type"] == "control":
                dag.add_edge(v, reversed_node, edge_type="anti_dependency")

    for c in ctrls:
        for v in dag.successors(c):
            if dag.get_edge_data(c, v)["edge_type"] == "target":
                target_edge_qubits = dag.get_edge_data(c, v)["qubits"]
                control_edge_qubits = dag.get_edge_data(c, uncomp_node)["qubits"]

                if set(target_edge_qubits).intersection(control_edge_qubits):
                    pass
                    dag.add_edge(reversed_node, v, edge_type="anti_dependency")

    return False


def uncompute_qc(qc, uncomp_qbs, recompute_qubits=[]):
    
    with fast_append():
        qc_new = qc.copy()
        previous_instructions = []
        follow_up_instructions = []
    
        while len(qc_new.data):
            instr = qc_new.data[0]
            if set(uncomp_qbs + recompute_qubits).intersection(instr.qubits):
                break
            previous_instructions.append(qc_new.data.pop(0))
            
    
        while len(qc_new.data):
            instr = qc_new.data[-1]
            if set(uncomp_qbs + recompute_qubits).intersection(instr.qubits):
                break
            follow_up_instructions.append(qc_new.data.pop(-1))
    
        follow_up_instructions = follow_up_instructions[::-1]
    
        dag = dag_from_qc(qc_new)
    
        lin = list(nx.topological_sort(dag))
        lin.reverse()
    
        for i in range(len(lin)):
            # for node in lin:
    
            node = lin[i]
    
            if node.instr is None:
                continue
    
            if not set(node.targets).intersection(uncomp_qbs):
                continue
    
            # if node.instr.op.name == "qb_alloc":# , "qb_alloc", "qb_dealloc"]:
            #     continue
    
            # This function checks if there is more than one allocation gate in
            # the upcoming chain of nodes. This prevents instructions which were
            # created due to previous recomputations to be included in further recomputations
            if detect_double_alloc(lin[i:], node.instr.qubits):
                continue
    
            if set(node.targets).issubset(uncomp_qbs) and node.instr and node.targets:
                recompute = uncompute_node(dag, node, uncomp_qbs, recompute_qubits)
                if recompute:
                    return uncompute_qc(
                        qc,
                        uncomp_qbs
                        + list(set(node.controls).intersection(recompute_qubits)),
                        recompute_qubits,
                    )
                continue
    
            off_target_qubits = list(set(node.targets) - set(uncomp_qbs))
            if off_target_qubits:
                not_uncomputable_qubits = []
                for qb in off_target_qubits:
                    if qb.allocated:
                        not_uncomputable_qubits.append(qb)
    
                if not_uncomputable_qubits:
                    raise Exception(
                        f'Uncomputation failed because gate "{node.instr.op.name}" needs to be uncomputed '
                        f'but is also targeting qubits {not_uncomputable_qubits} which are not up for uncomputation'
                    )
                else:
                    return uncompute_qc(
                        qc,
                        uncomp_qbs + off_target_qubits,
                        recompute_qubits=recompute_qubits,
                    )
    
        try:
            uncomputed_qc = qc_from_dag(dag)
        except nx.NetworkXUnfeasible:
            raise Exception("Cyclic dependency detected in DAG during uncomputation")
    
        uncomputed_qc.data = (
            previous_instructions + uncomputed_qc.data + follow_up_instructions
        )
        
    
        return uncomputed_qc


def detect_double_alloc(lin, qubits):
    qb_allocs = []
    for node in lin:
        if not node.instr:
            continue
        if node.instr.op.name == "qb_alloc" and node.instr.qubits[0] in qubits:
            if node.instr.qubits[0] in qb_allocs:
                return True
            else:
                qb_allocs.append(node.instr.qubits[0])

    return False
