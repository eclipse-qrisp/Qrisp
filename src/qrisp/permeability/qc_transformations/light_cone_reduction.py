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

import networkx as nx

from qrisp.permeability.qc_transformations.memory_management import topological_sort

# This function reorders the circuit such that the intended measurements can be executed
# as early as possible. Additionally, any instructions that are not needed for the
# intended measurements are removed

# Intended measurements has to be a list of qubits

# The strategy is similar to the one presented in reorder_circuit:
# We bring the circuit in the dag representation and perform a topological sort with
# the intended measurement as prefered instructions


# After that we investigate the circuit for instructions that can be removed
def lightcone_reduction(qc, intended_measurements):
    qc = qc.copy()

    # Insert intended measurements into the circuit
    for qb in intended_measurements:
        qc.measure(qb)

    # Generate dag representation
    from qrisp.permeability import PermeabilityGraph, TerminatorNode

    G = PermeabilityGraph(qc, remove_artificials=True)

    # Create result qc
    qc_new = qc.clearcopy()

    # Define prefered instructions
    measure_identifier = (
        lambda x: x.op.name == "measure" and x.qubits[0] in intended_measurements
    )

    def sub_sort(dag):
        nodes = list(dag.nodes())

        def sort_key(x):
            if isinstance(x, TerminatorNode):
                return 0
            else:
                return x.qc_index

        nodes.sort(key=sort_key)
        return nodes

    # Perform topological sort
    for n in topological_sort(G, prefer=measure_identifier, sub_sort=sub_sort):
        if n.instr:
            qc_new.append(n.instr)

    # Check which instructions come after the final measurement
    for i in range(len(qc_new.data))[::-1]:
        if measure_identifier(qc_new.data[i]):
            break

    redundant_qc = qc_new.clearcopy()

    redundant_qc.data = qc_new.data[i + 1 :]

    G = PermeabilityGraph(redundant_qc, remove_artificials=True)

    # #Now we need to make sure we don't remove deallocation gates from the data
    # #because this would inflate the qubit count of the compiled circuit

    # #The strategy here is that if we find a deallocation gate
    # #we remove any instruction involving the deallocated qubit from the list
    # #redundant instructions.

    for node in G.nodes():
        if node.instr is None:
            continue

        if node.instr.op.name == "qb_dealloc":
            ancs = nx.ancestors(G, node)

            redundant_qc.data.remove(node.instr)
            # print(f"removed {node.instr}")
            for pred in ancs:
                try:
                    redundant_qc.data.remove(pred.instr)
                except ValueError:
                    pass

    redundant_instructions = redundant_qc.data

    # We now remove the redundant instructions and the inserted
    # measurements from the circuit data
    i = 0
    while i < len(qc_new.data):
        if measure_identifier(qc_new.data[i]):
            qc_new.data.pop(i)
            continue
        if qc_new.data[i] in redundant_instructions:
            qc_new.data.pop(i)
            continue
        i += 1

    return qc_new
