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


import numpy as np
import networkx as nx
from numba import njit, prange

from qrisp.circuit import QuantumCircuit, Qubit, PTControlledOperation, ControlledOperation, transpile, QubitAlloc, Instruction, fast_append
from qrisp.misc import get_depth_dic, retarget_instructions, parallelize_qc

# The purpose of this function is to dynamically (de)allocate qubits when they are
# needed or not needed anymore. The qompiler function knows when a qubit is ready to
# deallocate (ie. it is in |0> state) due to a gate called QubitDealloc. After some
# preprocessing steps, this function will iterate through the data and dynamically
# (de)allocate if it is required. Another feature of this function is that it replaces
# mcx gates where the method is set to "auto" with implementations that fit the
# situation i.e. it takes into consideration how many clean/dirty ancilla qubits are
# available and synthesizes and mcx implementation based on this information.

# The parameter workspace can be given to extend the amount of qubits of the resulting
# QuantumCircuit by that amount. This extension can be helpfull because it gives the
# compiler more options to pick when allocating a new qubit. Since the allocation is
# based on a depth-reduction heuristic, giving more workspace results in many case in
# less depth circuits.


# It is however not only depth that can be improved by an increased workspace. Due to
# more clean/dirty ancillae beeing available, in many cases it is also possible to
# generate more efficient mcx implementations, thus also reducing the gate count.
def qompiler(
    qs,
    workspace=0,
    disable_uncomputation=True,
    intended_measurements=[],
    cancel_qfts=True,
    compile_mcm=False,
    gate_speed = None,
    use_dirty_anc_for_mcx_recomp=True
):
    if len(qs.data) == 0:
        return QuantumCircuit(0)
    
    if gate_speed is None:
        gate_speed = lambda x : 1

    with fast_append():
        qc = qs.copy()

        if not disable_uncomputation:
            local_qvs = qs.get_local_qvs()

            from qrisp.uncomputation.unqomp import uncompute_qc

            for qv in local_qvs:
                try:
                    qc = uncompute_qc(qc, qv.reg)
                except:
                    print(f"Warning: Automatic uncomputation for {qv.name} failed")
        
        
        # We now process the circuit in 3 important steps:
        #
        #   1. We use measurement reduction, i.e. we remove all gates that are not
        #   directly relevant to perform the intended measurement
        #
        #   2. We parallelize the circuit using the parallelization algorithm
        #
        #   3. We run reorder the circuit another time such that (de)allocations
        #   are executed in an advantageous order. The topological ordering algorithm
        #   behind the allocation is "stable" in the sense that it preserves the 
        #   previously induced order by the parallelization algorithm.
        
        # To achieve these steps we view the circuit from different levels of abstraction
        # This is achieved by dissolving composite gates until a certain condition is met.
        # Such conditions can be specified using the transpile_predicate keyword.
        # The two relevant levels of abstractions are:
        #
        #   1. The permeability level: Composite gates are dissolved if they have incomplete
        #   permeability information. In turn this implies that the resulting circuit contains
        #   only gates with complete permeability information (such as adders, general arithmetic
        #   or multi controlled X-gates). The resulting DAG can therefore use high-level
        #   features for reordering.
        
        def permeability_transpile_predicate(op):
            for v in op.permeability.values():
                if v is None:
                    return True
            return False
        
        #   2. Allocation level: This level of abstraction leaves only a selected set of low
        #   level gates alive. This comes with the advantage that the allocation algorithm
        #   can also look into reordering the inner structure of high-level functions to also
        #   get a good allocation strategy for this type of functions.
        
        from qrisp.logic_synthesis import LogicSynthGate
        from qrisp.mcx_algs import GidneyLogicalAND, JonesToffoli
        from qrisp.arithmetic import QuasiRZZ
        
        def allocation_level_transpile_predicate(op):
            
            if isinstance(op, PTControlledOperation):
                
                if op.base_operation.name == "x":
                    return False
                if op.base_operation.name == "p" and op.num_qubits == 2:
                    return False
            
            if isinstance(op, (LogicSynthGate, GidneyLogicalAND, JonesToffoli, QuasiRZZ)):
                return False
            
            if "QFT" == op.name[:3]:
                return False
            
            return True
        
        # Transpile to the first level
        
        qc = transpile(qc, transpile_predicate = permeability_transpile_predicate)
        
        if intended_measurements:
            # This function reorders the circuit such that the intended measurements can
            # be executed as early as possible additionally, any instructions that are
            # not needed for the intended measurements are removed
            try:
                qc = measurement_reduction(qc, intended_measurements)
                # pass
            except Exception as e:
                if "Unitary of operation " not in str(e):
                    raise e
        
        # Reorder circuit for parallelization. 
        # For more details check the implementation of this function
        
        qc = parallelize_qc(qc, depth_indicator = gate_speed)
        
        # We now reorder the transpiled circuit to achieve a good (de)allocation order. 
        # Reordering is performed based on the DAG representation of Unqomp. 
        # The advantage of this representation is that
        # it abstracts away non-trivial commutation relations between permeable gates.
        # The actual ordering is performed by performing a topological sort on this dag.
        # Contrary to unqomp, we don't use a modified version of Kahns algorithm here
        # (for more information check the topological sort function)

        # The reordering process aims to find an order that minimizes that maximum
        # amount of qubits that is needed. Losely speaking it tries to reorder the data,
        # that allocations are performed as late as possible and deallocations are
        # performed as early as possible.

        # Note that the more aggressive we transpile, the more optimization is possible
        # through reordering i.e. less qubits. By reordering we however also destroy any
        # previous order which might have been intentionally picked to optimize depth.
        # In summary: Transpiling more aggressively leads to less qubits but more depth.

        transpiled_qc = transpile(
            qc, transpile_predicate=allocation_level_transpile_predicate
        )
        
        reordered_qc = reorder_qc(transpiled_qc)

        if cancel_qfts:
            # Cancel adjacent QFT gates, which are inverse to each
            # other. This can happen alot because of the heavy use of Fourier arithmetic
            reordered_qc = qft_cancellation(reordered_qc)
            
        
        # Transpile logic synthesis
        def logic_synth_transpile_predicate(op):
            
            return (allocation_level_transpile_predicate(op) or
                    isinstance(op, LogicSynthGate))
        
        reordered_qc = transpile(
            reordered_qc,
        )
        
        # We combine adjacent single qubit gates
        if not qs.abstract_params and False:
            reordered_qc = combine_single_qubit_gates(reordered_qc)

        # We now determine the amount of Qubits the circuit will need
        required_qubits = 0
        max_required_qubits = 0


        for instr in reordered_qc.data:
            if instr.op.name == "qb_alloc":
                required_qubits += 1
            elif instr.op.name == "qb_dealloc":
                required_qubits -= 1
            if max_required_qubits < required_qubits:
                max_required_qubits += 1

        # Create a QuantumCircuit. Note that we make sure that the Qubit naming is
        # consistent, since we don't want any non-deterministic elements in the
        # function, as this can hinder bugfixing
        qc = QuantumCircuit()
        for i in range(workspace + max_required_qubits):
            qc.add_qubit(Qubit(str(i)))

        qc.clbits = list(qs.clbits)

        # This dictionary translates between the qubits of the input QuantumSession and
        # the compiled circuit
        translation_dic = {}

        # This list contains the Qubits which are currently not allocated
        free_qb_list = list(qc.qubits)

        allocated_qb_list = []

        if compile_mcm:
            # This list contains the clbits used for mcm mcx compilation
            mcm_clbits = []

        depth_dic = {b: 0 for b in qc.qubits + qc.clbits}
        
        
        # We now iterate through the data of the preprocessed QuantumCircuit
        for i in range(len(reordered_qc.data)):
            instr = reordered_qc.data[i]
            if instr.op.name == "barrier":
                continue

            # Check if any of the involved qubits need an allocation
            for qb in instr.qubits:
                if qb not in translation_dic:
                    # To pick a good allocation, we determine the depth dic and sort the
                    # available Qubits by their corresponding depth. Note that we add
                    # the identifier in order to prevent non-deterministic behavior
                    
                    alloc_cost = lambda x: (str(depth_dic[x]).zfill(20) + x.identifier)
                    
                    free_qb_list.sort(key = alloc_cost)
                    
                    translation_dic[qb] = free_qb_list.pop(0)

                    allocated_qb_list.append(qb)

            if instr.op.name == "qb_dealloc":
                # For the deallocation, we simply remove the qubits from the translation
                # dict and append it to the free_qb_list
                free_qb_list.append(translation_dic[instr.qubits[0]])
                allocated_qb_list.remove(instr.qubits[0])

                qc.append(
                    instr.op, [translation_dic[qb] for qb in instr.qubits], instr.clbits
                )

                del translation_dic[instr.qubits[0]]

                continue

            if (
                isinstance(instr.op, ControlledOperation)
                and instr.op.base_operation.name == "x"
                and instr.op.method == "auto"
                and len(instr.qubits) > 3
            ):
                # This section deals with the automatic recompilation of mcx gates.
                # The reason why we do this is, that it might not be clear how many free
                # ancillae are available for mcx gates. If we set the method to auto in
                # the mcx function, this section is called and replaces the generated
                # mcx gate with an implementation that is better fit to suit the amount
                # of available ancillae. We first determine the free ancillae

                clean_ancillae = list(free_qb_list)
                clean_ancillae.sort(
                    key=lambda x: depth_dic[x] + qc.qubits.index(x) * 1e-5
                )

                if use_dirty_anc_for_mcx_recomp:
                    dirty_ancillae = list(
                        set(translation_dic.values())
                        - set([translation_dic[qb] for qb in instr.qubits])
                    )
                    dirty_ancillae.sort(
                        key=lambda x: depth_dic[x] + qc.qubits.index(x) * 1e-5
                    )
                else:
                    dirty_ancillae = []

                QuantumCircuit.fast_append = False
                with fast_append(0):
                    # This function generates the data for the hybrid implementation
                    compiled_mcx_data = gen_hybrid_mcx_data(
                        instr.qubits[:-1],
                        instr.qubits[-1],
                        instr.op.ctrl_state,
                        clean_ancillae,
                        dirty_ancillae,
                    )
    
                    # We now append the data
                    for qb in clean_ancillae + dirty_ancillae:
                        translation_dic[qb] = qb
    
                    for instr in compiled_mcx_data:
                        qc.append(instr.op, [translation_dic[qb] for qb in instr.qubits])
                        update_depth_dic(qc.data[-1], depth_dic, depth_indicator = gate_speed)
    
                    # And free up the qubits
                    for qb in clean_ancillae + dirty_ancillae:
                        del translation_dic[qb]

            elif (
                isinstance(instr.op, GidneyLogicalAND)
                and compile_mcm
            ):
                
                if not instr.op.inv:
                    qc.append(instr.op.recompile(), 
                              [translation_dic[qb] for qb in instr.qubits])
                else:
                    
                    compile_qubits = [translation_dic[qb] for qb in instr.qubits]
                    qb_depth = max([depth_dic[qb] for qb in compile_qubits])
                    
                    if len(mcm_clbits) == 0:
                        clbit = qc.add_clbit()
                        depth_dic[clbit] = 0
                        mcm_clbits.append(clbit)
                        
                    min_clbit = np.argmin([depth_dic[cb] for cb in mcm_clbits])
                    
                    if depth_dic[mcm_clbits[min_clbit]] > qb_depth:
                        clbit = qc.add_clbit()
                        depth_dic[clbit] = 0
                        mcm_clbits.append(clbit)
                    else:
                        clbit = mcm_clbits[min_clbit]
                                        
                    qc.append(instr.op.recompile(), 
                              [translation_dic[qb] for qb in instr.qubits], 
                              [clbit])
                update_depth_dic(qc.data[-1], depth_dic, depth_indicator = gate_speed)
                    
            
            elif (
                isinstance(instr.op, JonesToffoli)
                and compile_mcm
            ):
                compile_qubits = [translation_dic[qb] for qb in instr.qubits]
                qb_depth = max([depth_dic[qb] for qb in compile_qubits])
                
                if len(mcm_clbits) == 0:
                    clbit = qc.add_clbit()
                    depth_dic[clbit] = 0
                    mcm_clbits.append(clbit)
                min_clbit = np.argmin([depth_dic[cb] for cb in mcm_clbits])
                
                if depth_dic[mcm_clbits[min_clbit]] > qb_depth:
                    clbit = qc.add_clbit()
                    depth_dic[clbit] = 0
                    mcm_clbits.append(clbit)
                else:
                    clbit = mcm_clbits[min_clbit]
                
                qc.append(instr.op.recompile(),
                          [translation_dic[qb] for qb in instr.qubits],
                          [clbit])
                update_depth_dic(qc.data[-1], depth_dic, depth_indicator = gate_speed)
                
            # Finally if all of the above cases are not met, we simply append the
            # operation to the translated qubits
            else:
                try:
                    qc.data.append(
                        Instruction(
                            instr.op,
                            [translation_dic[qb] for qb in instr.qubits],
                            instr.clbits,
                        )
                    )

                except KeyError:
                    raise Exception(
                        "Found operation "
                        + instr.op.name
                        + " on unallocated qubit during compilation."
                    )

                update_depth_dic(qc.data[-1], depth_dic, depth_indicator = gate_speed)

        # The following code is mostly about renaming and ordering the resulting circuit
        # in order to make the compiled circuit still comprehensible

        # We rename the allocated qubits to their name from the quantum session
        for i in range(len(qs.qubits)):
            if qs.qubits[i] in translation_dic:
                translation_dic[qs.qubits[i]].identifier = qs.qubits[i].identifier
                translation_dic[qs.qubits[i]].hash_value = qs.qubits[i].hash_value

        # We also want the qubits to be sorted according the order of the
        # QuantumVariables, i.e. the order of creation
        sorted_qubit_list = []
        for qv in qs.qv_list:
            # Due to the measurement reduction feature, not all qubits of quantum
            # variables that are live, are guaranteed to be represented in the
            # translation dict. Therefore, we need the try - except structure here
            try:
                sorted_qubit_list.extend([translation_dic[qb] for qb in qv.reg])
            except KeyError:
                pass

        # Furthermore, not all qubits that have been deallocated in the QuantumSession
        # are guaranteed to be removed from the translation dic, since the
        # measurement_reduction function might have removed their
        # uncomputation/deallocation gates
        temp = list(set(translation_dic.values()) - set(sorted_qubit_list))
        temp.sort(key=lambda x: x.identifier)
        sorted_qubit_list.extend(temp)

        # Finally, we rename the deallocated qubits to "workspace"
        workspace_naming_counter = 0
        td_values = list(translation_dic.values())
        for i in range(len(qc.qubits)):
            if not qc.qubits[i] in td_values:
                qc.qubits[i].identifier = "workspace_" + str(workspace_naming_counter)
                workspace_naming_counter += 1
                sorted_qubit_list.append(qc.qubits[i])

        qc.qubits = sorted_qubit_list

        reduced_qc = parallelize_qc(qc, depth_indicator = gate_speed)

    if reduced_qc.depth(depth_indicator = gate_speed) > qc.depth(depth_indicator = gate_speed):
        return qc
    else:
        return reduced_qc


def gen_hybrid_mcx_data(controls, target, ctrl_state, clean_ancillae, dirty_ancillae):
    # This function generates the data for the hybrid mcx implementation

    from qrisp.core import QuantumVariable
    from qrisp.mcx_algs import hybrid_mcx
    # Specify QuantumVariables to call mcx function
    control_qv = QuantumVariable(len(controls), name="control")
    target_qv = QuantumVariable(1, name="target")

    # dirty_ancillae = dirty_ancillae + clean_ancillae
    hybrid_mcx(
        control_qv,
        target_qv,
        ctrl_state=ctrl_state,
        num_ancilla=len(clean_ancillae),
        num_dirty_ancilla=len(dirty_ancillae),
    )

    # Get the list of used ancillae
    used_ancillae_set = (
        set(control_qv.qs.qubits) - set(control_qv.reg) - set(target_qv.reg)
    )

    # If we used the list() function to transform the set, this introduces a
    # non-deterministic element in the compilation algorithm, which can hamper bugfixing
    used_clean_ancillae = []
    used_dirty_ancillae = []

    for qb in control_qv.qs.qubits:
        if qb in used_ancillae_set:
            if "dirty" in qb.identifier:
                used_dirty_ancillae.append(qb)
            else:
                used_clean_ancillae.append(qb)

    depth_dic = get_depth_dic(control_qv.qs)

    used_clean_ancillae.sort(key=lambda x: -depth_dic[x])
    used_dirty_ancillae.sort(key=lambda x: -depth_dic[x])

    # ancilla_list = list(clean_ancillae)

    # for i in range(len(used_ancillae)):
    #     if "yong" in used_ancillae[i].identifier:
    #         used_ancillae.insert(0, used_ancillae.pop(i))
    #         ancilla_list.insert(0, dirty_ancillae[0])
    #         break

    # Now retarget the instructions such that they use the appropriate qubits
    data = control_qv.qs.data

    retarget_instructions(data, list(control_qv), controls)
    retarget_instructions(data, list(target_qv), [target])
    retarget_instructions(data, used_clean_ancillae, clean_ancillae)
    retarget_instructions(data, used_dirty_ancillae, dirty_ancillae)

    i = 0
    # Remove (de)allocation gates
    while i < len(data):
        if data[i].op.name in ["qb_dealloc", "qb_alloc"]:
            data.pop(i)
            continue
        i += 1

    return data


def reorder_qc(qc):
    from qrisp.uncomputation import dag_from_qc

    G = dag_from_qc(qc, remove_init_nodes=True)
    qc_new = qc.clearcopy()

    dealloc_identifier = lambda x: x.op.name == "qb_dealloc"
    alloc_identifer = lambda x: x.op.name == "qb_alloc"

    # mcx_identifier = lambda x : isinstance(x.op, PTControlledOperation) and x.op.base_operation.name == "x"
    # nmcx_identifier = lambda x : not mcx_identifier(x)
    # sub_sort = lambda G : topological_sort(G, prefer = mcx_identifier, delay = nmcx_identifier)
    # for n in topological_sort(G, prefer = dealloc_identifier, delay = alloc_identifer, sub_sort = sub_sort):

    def sub_sort(dag):
        nodes = list(dag.nodes())
        nodes.sort(key = lambda x : x.qc_index)
        return nodes

    for n in topological_sort(G, prefer=dealloc_identifier, delay=alloc_identifer, sub_sort = sub_sort):
        qc_new.append(n.instr)

    #The above algorithm does not move allocation gates to their latest possible
    #position (only compared to other deallocation gates)    
    #We therefore manually move the allocation gates to the position right
    #before the first actual instruction on that qubit.
    new_data = []
    delayed_qubit_alloc_dic = {}
    
    for i in range(len(qc_new.data)):
        instr = qc_new.data[i]
        
        if instr.op.name == "qb_alloc":
            delayed_qubit_alloc_dic[instr.qubits[0]] = instr
        else:
            # We sort the qubits in order to prevent non-deterministic compilation behavior
            alloc_qubits = list(set(delayed_qubit_alloc_dic.keys()).intersection(instr.qubits))
            alloc_qubits.sort(key = lambda x : hash(x))
            for qb in alloc_qubits:
                new_data.append(delayed_qubit_alloc_dic[qb])
                del delayed_qubit_alloc_dic[qb]
            
            new_data.append(instr)
    
    for instr in delayed_qubit_alloc_dic.values():
        new_data.append(instr)
    
    qc_new.data = new_data

    return qc_new


# This function performs a topological sort of the graph G where we try to execute any
# deallocation gates as early as possible while still adhering to the topological order.
# We try to perform a depth-first search as described here:
# https://en.wikipedia.org/wiki/Topological_sorting
# According to the Wikipedia page, we are allowed to pick any node as a "starting point"
# of the DF-search, which allows us to modify the algorithm such that it optimizes the
# (de)allocation order. The general idea is to pick the deallocation nodes as
# starting points, where we order them, such that those deallocation nodes that
# "require" the least allocation nodes are executed first. "Require" here means that
# there is a causal relationship between the allocation and deallocation nodes,
# i.e. there is a path from the allocation node to the deallocation node.


# We can thus determine the amount of allocation nodes required for a deallocation node
# n by counting, the amount of allocation nodes in the "ancestors" subgraph of n.
def topological_sort(G, prefer=None, delay=None, sub_sort=nx.topological_sort):
    """
    Function to perform a topological sort on an Unqomp DAG which allows preferring/
    delaying specific types of nodes

    Parameters
    ----------
    G : nx.DiGraph
        The Unqomp DAG.
    prefer : function, optional
        Function which returns True, when presented with an Instruction, that should be
        preferred. The default is the function that returns False on all Operations
    delay : function, optional
        Function which returns True, when presented with an Instruction, that should be
        delayed. The default is the function that returns False on all Operations
    sub_sort : function, optional
        A function which performs a topological sort, which can sorting preferences of
        secondary importance. The default is nx.topological_sort.

    Returns
    -------
    lin : list[UnqompNode]
        The linearized list of UnqompNodes. The init nodes are not included.

    """

    if prefer is None:
        prefer = lambda x: False

    if delay is None:
        delay = lambda x: False

    G = G.copy()
    # Collect the prefered nodes
    prefered_nodes = []

    for n in G.nodes():
        if prefer(n.instr):
            prefered_nodes.append(n)

        n.processed = False

    # For large scales, finding the ancestors is a bottleneck. We therefore use a
    # jitted version
    if len(G) * len(prefered_nodes) > 10000:
        anc_lists = ancestors(G, prefered_nodes)
    else:
        anc_lists = []
        for i in range(len(prefered_nodes)):
            anc_lists.append(list(nx.ancestors(G, prefered_nodes[i])))

    node_ancs = {
        prefered_nodes[i]: anc_lists[i] for i in range(len(prefered_nodes))
    }
    
    # We sort the nodes in order to prevent non-deterministic compilation behavior
    prefered_nodes.sort(key=lambda x: len(node_ancs[x]) + 1/hash(x.instr))
    
    # Determine the required delay nodes for each prefered nodes
    required_delay_nodes = {n: [] for n in prefered_nodes}

    for n in prefered_nodes:
        for k in node_ancs[n]:
            if delay(k.instr):
                required_delay_nodes[n].append(k)

    required_delay_nodes = {n: set(required_delay_nodes[n]) for n in prefered_nodes}

    
    # Generate linearization
    lin = []

    while prefered_nodes:
        # Sort nodes accordingly
        prefered_nodes.sort(key=lambda x: len(required_delay_nodes[x]))

        node = prefered_nodes.pop(0)
        ancs = []

        for n in node_ancs[node] + [node]:
            if n.processed:
                continue
            else:
                n.processed = True
                ancs.append(n)

        sub_graph = G.subgraph(ancs)

        lin += list(sub_sort(sub_graph))

        continue

        for n in prefered_nodes:
            required_delay_nodes[n] = (
                required_delay_nodes[n] - required_delay_nodes[node]
            )

    # Linearize the remainder
    remainder = []
    for n in G.nodes():
        if n.processed:
            continue
        else:
            n.processed = True
            remainder.append(n)

    # lin += list(sub_sort(G))
    lin += list(sub_sort(G.subgraph(remainder)))

    return lin


# Function to combine any sequences of single qubit gates into a single U3
def combine_single_qubit_gates(qc):
    def apply_combined_gates(qc_new, gate_list, qb):
        if not len(gate_list):
            return

        n = len(gate_list)

        m = np.eye(2)
        while gate_list:
            gate = gate_list.pop(-1)
            m = np.dot(m, gate.get_unitary())

        if np.linalg.norm(m - np.eye(2)) < 1e-10:
            return

        if n == 1:
            qc_new.append(gate, [qb])
            return

        qc_new.unitary(m, [qb])

    qb_dic = {qb: [] for qb in qc.qubits}

    qc_new = qc.clearcopy()

    for instr in qc.data:
        if (
            len(instr.qubits) > 1
            or instr.op.name in ["qb_alloc", "qb_dealloc"]
            or len(instr.clbits) > 0
        ):
            for qb in instr.qubits:
                apply_combined_gates(qc_new, qb_dic[qb], qb)
            qc_new.append(instr)
        else:
            qb_dic[instr.qubits[0]].append(instr.op)

    for qb in qc.qubits:
        apply_combined_gates(qc_new, qb_dic[qb], qb)

    return qc_new


# This function reorders the circuit such that the intended measurements can be executed
# as early as possible. Additionally, any instructions that are not needed for the
# intended measurements are removed

# Intended measurements has to be a list of qubits

# The strategy is similar to the one presented in reorder_circuit:
# We bring the circuit in the dag representation and perform a topological sort with
# the intended measurement as prefered instructions


# After that we investigate the circuit for instructions that can be removed
def measurement_reduction(qc, intended_measurements):
    qc = qc.copy()

    # Insert intended measurements into the circuit
    for qb in intended_measurements:
        qc.measure(qb)

    # Generate dag representation
    from qrisp.uncomputation import dag_from_qc

    G = dag_from_qc(qc, remove_init_nodes=True)

    # Create result qc
    qc_new = qc.clearcopy()

    # Define prefered instructions
    measure_identifier = (
        lambda x: x.op.name == "measure" and x.qubits[0] in intended_measurements
    )
    
    def sub_sort(dag):
        nodes = list(dag.nodes())
        nodes.sort(key = lambda x : x.qc_index)
        return nodes

    # Perform topological sort
    for n in topological_sort(G, prefer=measure_identifier, sub_sort = sub_sort):
        qc_new.append(n.instr)

    # Check which instructions come after the final measurement
    for i in range(len(qc_new.data))[::-1]:
        if measure_identifier(qc_new.data[i]):
            break

    redundant_qc = qc_new.clearcopy()

    redundant_qc.data = qc_new.data[i + 1 :]

    G = dag_from_qc(redundant_qc, remove_init_nodes=True)

    # #Now we need to make sure we don't remove deallocation gates from the data
    # #because this would inflate the qubit count of the compiled circuit

    # #The strategy here is that if we find a deallocation gate
    # #we remove any instruction involving the deallocated qubit from the list
    # #redundant instructions.

    for node in G.nodes():
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


def allocation_graph(qc):
    from qrisp.uncomputation.unqomp import dag_from_qc

    dag = dag_from_qc(qc, remove_init_nodes=True)

    res = nx.DiGraph()

    dealloc_nodes = []
    for n in dag.nodes():
        if n.instr.op.name in ["qb_alloc", "qb_dealloc"]:
            res.add_node(n)
        if n.instr.op.name == "qb_dealloc":
            dealloc_nodes.append(n)

    for n in dealloc_nodes:
        for anc in nx.ancestors(n, dag):
            if anc.instr.op.name == "qb_alloc":
                res.add_edge(anc, n)

    return n


def update_depth_dic(instruction, depth_dic, depth_indicator = None):
    
    if depth_indicator is None:
        depth_indicator = lambda x : 1
    
    if instruction.op.definition:
        qc = QuantumCircuit()
        qc.qubits = instruction.qubits
        qc.clbits = instruction.clbits
        qc.append(instruction)
        instr_list = qc.transpile().data
    else:
        instr_list = [instruction]

    # Assign each bit in the circuit a unique integer
    # to index into op_stack.
    # If no bits, return 0

    # Here we are playing a modified version of
    # Tetris where we stack gates, but multi-qubit
    # gates, or measurements have a block for each
    # qubit or cbit that are connected by a virtual
    # line so that they all stacked at the same depth.
    # Conditional gates act on all cbits in the register
    # they are conditioned on.
    # We treat barriers or snapshots different as
    # They are transpiler and simulator directives.
    # The max stack height is the circuit depth.

    for instr in instr_list:
        if instr.op.name in ["qb_alloc", "qb_dealloc", "gphase"]:
            continue
        qargs = instr.qubits
        cargs = instr.clbits

        levels = []
        # If count then add one to stack heights

        for b in qargs + cargs:
            # Add to the stacks of the qubits and
            # cbits used in the gate.
            levels.append(depth_dic[b] + depth_indicator(instr.op))

        max_level = max(levels)

        for b in qargs + cargs:
            depth_dic[b] = max_level


@njit(cache=True)
def ancestors_jitted(start_index, indptr, indices, node_amount):
    to_do_array = np.zeros(node_amount, dtype=np.byte)
    to_do_array[start_index] = 1
    done_array = np.zeros(node_amount, dtype=np.byte)

    stack = 1
    while stack:
        node = np.argmax(to_do_array)
        to_do_array[node] = 0

        for i in range(indptr[node], indptr[node + 1]):
            new_node = indices[i]
            if done_array[new_node] == 0:
                to_do_array[new_node] = 1
                stack += 1

        done_array[node] = 1
        stack -= 1

    return np.nonzero(done_array)[0]


@njit(parallel=True, cache=True)
def ancestors_jitted_wrapper(start_indices, indptr, indices, node_amount):
    res = [np.zeros(1, dtype=np.int64)] * len(start_indices)
    for i in prange(len(start_indices)):
        start_index = start_indices[i]
        res[i] = ancestors_jitted(start_index, indptr, indices, node_amount)

    return res


def ancestors(dag, start_nodes):
    node_list = list(dag.nodes())

    sprs_mat = nx.to_scipy_sparse_array(dag, format="csc")

    start_indices = []
    for i in range(len(dag)):
        if node_list[i] in start_nodes:
            start_indices.append(i)

    res_list_indices = ancestors_jitted_wrapper(
        np.array(start_indices).astype(np.int32),
        sprs_mat.indptr,
        sprs_mat.indices.astype(np.int32),
        len(dag),
    )

    node_list = [
        [node_list[j] for j in anc_indices] for anc_indices in res_list_indices
    ]

    return node_list

# REWORK required: Instead of detecting a QFT by the gate name, a much more robust
# approach is to create a Operation subtype, that describes QFT gates and do type-
# checking here.

# Function to cancel adjacent QFT, which are inverse to each other
# Due to the heavy use of Fourier arithmetic, this can happen alot
# especially if multiple arithmetic operation on a single target are executed
def qft_cancellation(qc):
    # The idea is to iterate through the instructions of the circuit
    # and save the last instruction acting on each qubit.
    # If a QFT is appended, we check if the last instruction was an inverse QFT
    # or an allocation and denote the index if so
    # Afterwards, we remove the QFTs or replace them with H gates

    last_instr_dic = {qb: None for qb in qc.qubits}
    cancellation_indices = []
    h_replacements = []
    dealloc_replacements = []
    from numpy.linalg import norm

    for i in range(len(qc.data)):
        
        instr = qc.data[i]
        if "QFT" == instr.op.name[:3] and "adder" not in instr.op.name:
            previous_instruction_type = []

            for qb in instr.qubits:
                previous_instruction = qc.data[last_instr_dic[qb]]
                
                if previous_instruction.op.name == "qb_alloc":
                    previous_instruction_type.append("alloc")
                    continue
                
                if previous_instruction.op.num_qubits != instr.op.num_qubits:
                    break
                
                if not qb in previous_instruction.qubits:
                    break
                
                if tuple(previous_instruction.qubits) != tuple(instr.qubits):
                    break

                if qc.data.index(previous_instruction) in cancellation_indices:
                    break
                
                if qc.data.index(previous_instruction) in h_replacements:
                    break

                if "QFT" == previous_instruction.op.name[:3]:
                    if instr.op.num_qubits < 8:
                        unitary_self = instr.op.get_unitary()
                        inv_unitary_other = (
                            previous_instruction.op.get_unitary()
                            .transpose()
                            .conjugate()
                        )

                        if bool(norm(unitary_self - inv_unitary_other) < 10**-4):
                            previous_instruction_type.append("QFT")
                        else:
                            break

                    elif hash(instr.op.definition) == hash(
                        previous_instruction.op.definition.inverse()
                    ):
                        previous_instruction_type.append("QFT")
                    else:
                        break

                else:
                    break
                

                
            else:
                if len(set(previous_instruction_type)) == 1:
                    if previous_instruction_type[0] == "QFT":
                        cancellation_indices.append(i)
                        cancellation_indices.append(last_instr_dic[qb])
                    else:
                        h_replacements.append(i)

        if instr.op.name == "qb_dealloc":
            deallocated_qubit = instr.qubits[0]
            previous_instruction = qc.data[last_instr_dic[deallocated_qubit]]

            if "QFT" in previous_instruction.op.name:
                try:
                    previous_instruction.deallocated_qubits[deallocated_qubit] = True
                except AttributeError:
                    previous_instruction.deallocated_qubits = {
                        qb: False for qb in previous_instruction.qubits
                    }
                    previous_instruction.deallocated_qubits[deallocated_qubit] = True
                    dealloc_replacements.append(last_instr_dic[deallocated_qubit])

        for qb in instr.qubits:
            last_instr_dic[qb] = i

    for i in dealloc_replacements:
        instr = qc.data[i]
        for qb in instr.qubits:
            if not instr.deallocated_qubits[qb]:
                break
        else:
            h_replacements.append(i)

    new_qc = qc.clearcopy()

    # print(len(qc.data))
    for i in range(len(qc.data)):
        if i in h_replacements:
            for qb in qc.data[i].qubits:
                new_qc.h(qb)
            # print(qc.data[i].qubits)
            # print("H replacement successfull")
            continue

        if i in cancellation_indices:
            continue
        
        else:
            new_qc.append(qc.data[i])
    # print(len(new_qc.data))
    # print("====")
    return new_qc
