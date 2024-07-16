# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:12:55 2024

@author: sea
"""

import networkx as nx

from qrisp.circuit import fast_append, ControlledOperation, PTControlledOperation
from qrisp.logic_synthesis import LogicSynthGate
from qrisp.uncomputation.type_checker import is_qfree

from qrisp.uncomputation.X_permeability_dag import PermeabilityGraph, InstructionNode, TerminatorNode

def uncompute_qc(qc, uncomp_qbs, recompute_qubits=[]):
    
    with fast_append(0):
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
    
        dag = PermeabilityGraph(qc_new)
    
        lin = list(nx.topological_sort(dag))
        lin.reverse()
    
        for i in range(len(lin)):
    
            node = lin[i]
    
            if isinstance(node, TerminatorNode):
                continue
    
            targets = dag.get_target_qubits(node)
            
            if not set(targets).intersection(uncomp_qbs):
                continue
    
            # This function checks if there is more than one allocation gate in
            # the upcoming chain of nodes. This prevents instructions which were
            # created due to previous recomputations to be included in further recomputations
            if detect_double_alloc(lin[i:], node.instr.qubits):
                continue
    
            if set(targets).issubset(uncomp_qbs) and node.instr and targets:
                recompute = uncompute_node(dag, node, uncomp_qbs, recompute_qubits)
                if recompute:
                    return uncompute_qc(
                        qc,
                        uncomp_qbs
                        + list(set(node.controls).intersection(recompute_qubits)),
                        recompute_qubits,
                    )
                continue
    
            off_target_qubits = list(set(targets) - set(uncomp_qbs))
            
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
            uncomputed_qc = dag.to_qc()
        except nx.NetworkXUnfeasible:
            raise Exception("Cyclic dependency detected in DAG during uncomputation")
    
        uncomputed_qc.data = (
            previous_instructions + uncomputed_qc.data + follow_up_instructions
        )
        
        return uncomputed_qc



def uncompute_node(dag, uncomp_node, uncomp_qbs, recompute_qubits=[]):

    target_qubits = dag.get_target_qubits(uncomp_node)
    
    a_star_n_list = [dag.recent_node_dic[qb] for qb in set(target_qubits)]
    a_star_n_list = list(set(a_star_n_list))
    
    if not is_qfree(uncomp_node.instr.op):
        raise Exception(f"Tried to uncompute non-qfree instruction {uncomp_node.instr}")

    ctrls = dag.get_control_nodes(uncomp_node)

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

    reversed_node = InstructionNode(new_instr)

    uncomp_node.uncomputed_node = reversed_node

    dag.add_node(reversed_node)
    
    for qb in target_qubits:
        dag.recent_node_dic[qb] = reversed_node

    for i in range(len(ctrls)):
        c = ctrls[i]

        control_edge_qubits = dag.get_edge_qubits(c, uncomp_node)

        if set(control_edge_qubits).intersection(recompute_qubits):
            for k in dag.successors(c):
                if dag.get_edge_data(c, k)["edge_type"] in ["X", "neutral"] and set(
                    dag.get_edge_data(c, k)["qubits"]
                ).intersection(control_edge_qubits):
                    if k.uncomputed_node is None:
                        return True
                    ctrls[i] = k.uncomputed_node
                    break
            else:
                return True

        dag.add_edge(
            ctrls[i], reversed_node, edge_type="Z", qubits=control_edge_qubits
        )

    for a_star_n in a_star_n_list:
        dag.add_edge(
            a_star_n,
            reversed_node,
            edge_type="neutral",
            qubits=list(
                set(uncomp_qbs).intersection(dag.get_target_qubits(a_star_n)).intersection(new_instr.qubits)
            ),
        )

        for v in dag.successors(a_star_n):
            if dag.get_edge_data(a_star_n, v)["edge_type"] == "Z":
                dag.add_edge(v, reversed_node, edge_type="anti_dependency")

    for c in ctrls:
        for v in dag.successors(c):
            if dag.get_edge_data(c, v)["edge_type"] in ["X", "neutral"]:
                target_edge_qubits = dag.get_edge_qubits(c, v)
                control_edge_qubits = dag.get_edge_qubits(c, uncomp_node)

                if set(target_edge_qubits).intersection(control_edge_qubits):
                    dag.add_edge(reversed_node, v, edge_type="anti_dependency")

    return False

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