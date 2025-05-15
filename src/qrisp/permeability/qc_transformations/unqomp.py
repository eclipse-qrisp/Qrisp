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

from qrisp.circuit import fast_append, ControlledOperation, PTControlledOperation
from qrisp.permeability.type_checker import is_qfree

from qrisp.permeability.permeability_dag import (
    PermeabilityGraph,
    InstructionNode,
    TerminatorNode,
    AllocNode,
)


def uncompute_qc(qc, uncomp_qbs, recompute_qubits=[]):
    """
    This function applies the Unqomp algorithm to the QuantumCircuit qc, uncomputing
    the qubits uncomp_qbs. It is possible to specify a set of qubits that has been
    uncomputed earlier, which is allowed to be recomputed.

    Parameters
    ----------
    qc : QuantumCircuit
        The QuantumCircuit to uncompute.
    uncomp_qbs : list[Qubit]
        A list of Qubits to uncompute using the Unqomp algorithm.
    recompute_qubits : list[Qubit], optional
        A list of Qubits, that have been uncomputed earlier, which the algorithm
        is allowed to recompute. The default is [].

    Raises
    ------
    Exception
        Cyclic dependency detected in DAG during uncomputation.
    Exception
        Uncomputation failed because gate needs to be uncomputed
        but is also targeting qubits which are not up for uncomputation

    Returns
    -------
    uncomputed_qc
        The result of the Unqomp algorithm.

    """

    with fast_append():

        # To speed up the uncomputation, the first step is to filter
        # out instruction that are guaranteed that they don't need to be uncomputed.

        # The idea is that every instruction before the first instruction that operators
        # on the uncomputation qubits satisfies this property.
        # Same for every instruction after the last instruction.
        qc_new = qc.copy()
        previous_instructions = []
        follow_up_instructions = []

        # Determine the the previous instructions.
        while len(qc_new.data):
            instr = qc_new.data[0]
            if set(uncomp_qbs + recompute_qubits).intersection(instr.qubits):
                break
            previous_instructions.append(qc_new.data.pop(0))

        # Determine the the follow-up instructions.
        while len(qc_new.data):
            instr = qc_new.data[-1]
            if set(uncomp_qbs + recompute_qubits).intersection(instr.qubits):
                break
            follow_up_instructions.append(qc_new.data.pop(-1))

        follow_up_instructions = follow_up_instructions[::-1]

        # Build up the pdag
        pdag = PermeabilityGraph(qc_new)
        # Now come the steps of the Unqomp algorithm

        # We iterate over a reversed topological sort and successively
        # uncompute all relevant nodes.
        lin = list(nx.topological_sort(pdag))
        lin.reverse()

        for i in range(len(lin)):

            # Set an alias
            node = lin[i]

            # If the node is a terminator it needs no uncomputation
            if isinstance(node, TerminatorNode):
                continue

            # Get the target qubits
            # target qubits means the qubits the corresponding instruction is
            # operating either with X-permeability or neutral
            target_qubits = pdag.get_target_qubits(node)

            # If there is no intersection between the target qubits of the instruction
            # and the qubits to uncompute, this instruction needs no uncomputation
            if not set(target_qubits).intersection(uncomp_qbs):
                continue

            # This function checks if there is more than one allocation gate in
            # the upcoming chain of nodes. This prevents instructions which were
            # created due to previous recomputations to be included in further recomputations
            if detect_double_alloc(lin[i:], node.instr.qubits):
                continue

            # This case represents the neccesity to uncompute
            if set(target_qubits).issubset(uncomp_qbs) and node.instr and target_qubits:

                # The uncompute node function inserts an uncomputation node into the pdag
                # If a recomputation is required, no nodes are inserted and it returns True
                recompute = uncompute_node(pdag, node, uncomp_qbs, recompute_qubits)

                # If a recomputation is required, we call the algorithm with the approriate
                # recomputation qubits. Qubits that could require recomputation can only
                # be part of the controls
                if recompute:
                    return uncompute_qc(
                        qc,
                        uncomp_qbs
                        + list(
                            set(pdag.get_control_qubits(node)).intersection(
                                recompute_qubits
                            )
                        ),
                        recompute_qubits,
                    )
                continue

            # This case treats the possibility that the instruction is targeting qubits
            # which are not part of the uncomputation qubits.

            # This would is usually not uncomputable, however it could also imply
            # that these qubits have been uncomputed allready but gate_wrapped.
            # The gate_wrap decorator makes ancilla slots as non-permeable,
            # so these qubits appear as target qubits here.
            off_target_qubits = list(set(target_qubits) - set(uncomp_qbs))

            if off_target_qubits:

                non_uncomputable_qubits = []
                for qb in off_target_qubits:
                    if qb.allocated:
                        non_uncomputable_qubits.append(qb)

                if non_uncomputable_qubits:
                    raise Exception(
                        f'Uncomputation failed because gate "{node.instr.op.name}" needs to be uncomputed '
                        f"but is also targeting qubits {non_uncomputable_qubits} which are not up for uncomputation"
                    )
                else:
                    return uncompute_qc(
                        qc,
                        uncomp_qbs + off_target_qubits,
                        recompute_qubits=recompute_qubits,
                    )

        try:
            # Retrieve the uncomputed QuantumCircuit
            uncomputed_qc = pdag.to_qc()
        except nx.NetworkXUnfeasible:
            raise Exception("Cyclic dependency detected in DAG during uncomputation")

        # Insert the previous and follow_up instructions
        uncomputed_qc.data = (
            previous_instructions + uncomputed_qc.data + follow_up_instructions
        )

        # Return the uncomputed QuantumCircuit
        return uncomputed_qc


def uncompute_node(pdag, node, uncomp_qbs, recompute_qubits=[]):
    """
    Uncomputes a node in a given PermeabilityGraph (in-place) according to the
    Unqomp algorithm.

    Parameters
    ----------
    pdag : PermeabilityGraph
        The PermeabilityGraph to perform the uncomputation in.
    node : UnqompNode
        The node to uncompute.
    uncomp_qbs : list[Qubit]
        A list of Qubits that are supposed to be uncomputed.
    recompute_qubits : list[Qubit], optional
        A list of Qubit, where the algorithm is allowed to recompute them. The default is [].

    Raises
    ------
    Exception
        Tried to uncompute non-qfree instruction

    Returns
    -------
    bool
        A bool indicating whether a recomputation is required.

    """

    # Get the target qubits
    target_qubits = pdag.get_target_qubits(node)

    # In the Unqomp paper there is only one a* node.
    # However since we also allow multiple target operations, our case also
    # includes the possibility of having multiple a*

    # Retrieve the list of a* nodes
    a_star_n_list = [pdag.recent_node_dic[qb] for qb in set(target_qubits)]

    # Remove duplicates
    a_star_n_list = list(set(a_star_n_list))

    if not is_qfree(node.instr.op):
        raise Exception(f"Tried to uncompute non-qfree instruction {node.instr}")

    # Retrieve the list of nodes that are connected to node by a control edges.
    ctrls = pdag.get_control_nodes(node)

    # If the control node is already uncomputed, we use this node instead.
    # for i in range(len(ctrls)):
    #     if not ctrls[i].uncomputed_node is None:
    #         ctrls[i] = ctrls[i].uncomputed_node

    # Replace controlled operations by phase tolerant controlled operations
    if isinstance(node.instr.op, ControlledOperation):
        op = node.instr.op

        if op.method == "auto" and len(op.controls) < 5 or op.method == "gray":
            if len(op.controls) != 1:
                node.instr.op = PTControlledOperation(
                    op.base_operation,
                    num_ctrl_qubits=len(op.controls),
                    ctrl_state=op.ctrl_state,
                    method="gray_pt",
                )

    from qrisp.alg_primitives.logic_synthesis import LogicSynthGate

    # Replace results of logic synthesis by phase tolerant logic synthesis
    if isinstance(node.instr.op, LogicSynthGate):
        if node.instr.op.logic_synth_method == "gray":
            from qrisp import QuantumVariable

            tt = node.instr.op.tt

            input_qv = QuantumVariable(tt.bit_amount)
            output_qv = QuantumVariable(tt.shape[1], qs=input_qv.qs)

            tt.q_synth(input_qv, output_qv, method="gray_pt")

            temp = input_qv.qs.data[-1].op
            temp.name = node.instr.op.name

            node.instr.op = temp

    # Create the instruction for the new UnqompNode
    new_instr = node.instr.inverse()

    # Create the new UnqompNode
    reversed_node = InstructionNode(new_instr)

    # Link the uncomputed node to the original node
    node.uncomputed_node = reversed_node

    # Add the uncomputed node to the pdag
    pdag.add_node(reversed_node)

    reversed_node.value_layer = max([n.value_layer + 1 for n in a_star_n_list + ctrls])

    # Update the recent_node_dic with the reversed node
    for qb in target_qubits:
        pdag.recent_node_dic[qb] = reversed_node

    # We now connect the edges of the reversed node.

    # The first step is to connect the controls.
    # The idea here is to connect the reversed node to the
    # controls of the original node.

    # This dictionary will hold the control qubits of each control edge
    ctrl_qubit_dic = {}
    for c in ctrls:
        ctrl_qubit_dic[c] = pdag.get_edge_qubits(c, node)

    while len(ctrls):

        # Set alias
        c = ctrls.pop(0)

        # Get the qubits of the control edge
        control_edge_qubits = ctrl_qubit_dic[c]

        recomputation_required = set(control_edge_qubits).intersection(recompute_qubits)

        # This treats the case that the control edge is among the recomputable qubits
        # implying this node needs to be recomputed.
        if recomputation_required and pdag.has_edge(c, node):

            # If the required qubits for recomputation are not among the uncomputation
            # qubits, we cancel the function by returning True
            # The parent function will then start a new uncomputation attempt,
            # where the recompute qubits are among the uncomputation qubits.
            if not recomputation_required.issubset(uncomp_qbs):
                return True

            # Instead of connecting to the previos control node, we connect to the
            # "latest" control node that operated on that qubit.
            for qb in recomputation_required:
                ctrls.append(pdag.recent_node_dic[qb])
                ctrl_qubit_dic[pdag.recent_node_dic[qb]] = [qb]

            continue

        # We now add the anti_dependency edges starting at the reversed node

        # For this we iterate over the control qubits and check the streak
        # associated to that qubit
        for qb in control_edge_qubits:

            streak_children = pdag.get_streak_children(c, qb)

            # Iterate over the streak
            for streak_child in streak_children:

                # The streak is cancelled by the streak children of the streak_child
                for cancellation_node in pdag.get_streak_children(streak_child, qb):

                    pdag.add_edge(
                        reversed_node,
                        cancellation_node,
                        edge_type="anti_dependency",
                        qubits=[qb],
                    )

                    cancellation_node.value_layer = max(
                        cancellation_node.value_layer, reversed_node.value_layer + 1
                    )

        # Add the edge from the control node of the original node to the reversed node.
        pdag.add_edge(c, reversed_node, edge_type="Z", qubits=control_edge_qubits)

    # The next step is to connect the targets.
    # For that we use the a_star_n_list which represents all the nodes, that are
    # targetted by the operation
    for a_star_n in a_star_n_list:

        # We first add the edge to the qubits targetted by a_star_n

        # The qubits of this edge are the qubits which are:
        #   1. Part of the uncomputation qubits
        #   2. Targeting a_star_n
        #   3. Part of the instruction

        # Add the edge

        # We need to consider the possibility that we end a streak by adding this node
        # we therefore consider the nodes that use a_star_n as a control node

        # We now iterate over the targets of the node to connect the edges
        for qb in target_qubits:

            # These are the nodes which form a streak on that qubit
            streak_children = pdag.get_streak_children(a_star_n, qb)
            # Depending on how long the streak is, we have ton insert a TerminatorNode

            if len(streak_children) == 0:
                pdag.add_edge(a_star_n, reversed_node, edge_type="neutral", qubits=[qb])

            # If there is only one control, we can safely append a neutral edge to that control node
            elif len(streak_children) == 1:
                pdag.add_edge(
                    streak_children[0], reversed_node, edge_type="neutral", qubits=[qb]
                )

            # If there is a streak, we need to insert a terminator edge
            else:
                t_node = TerminatorNode(qb)
                t_node.value_layer = 0
                for v in streak_children:
                    pdag.add_edge(v, t_node, edge_type="anti_dependency", qubits=[qb])

                    if v.value_layer > t_node.value_layer:
                        t_node.value_layer = v.value_layer + 1

                # Add the edge
                pdag.add_edge(t_node, reversed_node, edge_type="neutral", qubits=[qb])

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
