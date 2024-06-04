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

import threading
import sys
# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

from qrisp.circuit import (
    ControlledOperation,
    Instruction,
    Operation,
    QuantumCircuit,
    fast_append,
    transpile,
    ClControlledOperation
)
from qrisp.interface.circuit_converter import convert_circuit
from qrisp.simulator.circuit_preprocessing import (
    circuit_preprocessor,
    count_measurements_and_treat_alloc,
    group_qc,
    extract_measurements,
    insert_multiverse_measurements
)
from qrisp.simulator.quantum_state import QuantumState

# This dictionary contains the qrisp operations as values and their names as keys.
# This is advantageous because during the simulation, we don't want to synthesize them
# every time they are required
# from qrisp.circuit import op_list
# op_dict = {op().name : op for op in op_list}


# This functions determines the quantum state after executing a quantum circuit
# and afterwards extracts the probability of measuring certain bit strings
def run(qc, shots, token="", iqs=None, insert_reset=True):
    
    if len(qc.data) == 0:
        return {"": shots}
    
    progress_bar = tqdm(
        desc=f"Simulating {len(qc.qubits)} qubits..",
        bar_format="{desc} |{bar}| [{percentage:3.0f}%]",
        ncols=85,
        leave=False,
        delay=0.1,
        position=0,
        smoothing=1,
        file=sys.stdout
    )
    
    LINE_CLEAR = "\x1b[2K"
    progress_bar.display()

    # This command enables fast appending. Fast appending means that the .append method
    # of the QuantumCircuit class checks much less validity conditions and is also less
    # tolerant regarding inputs.
    with fast_append(2):

        # We convert the circuit which is given with portable objects to a qrisp circuit
        # qc = convert_circuit(qc, "qrisp", transpile=True)
        qc = qc.transpile()

        # Count the amount of measurements (we can stop the simulation after all
        # measurements are performed)
        measurement_amount = count_measurements_and_treat_alloc(
            qc, insert_reset=insert_reset
        )

        # Apply circuit preprocessing more
        qc = circuit_preprocessor(qc)

        measurement_counter = 0

        for i in range(len(qc.data)):
            if qc.data[i].op.name == "measure":
                measurement_counter += 1
            if measurement_counter == measurement_amount:
                break

        qc.data = qc.data[: i + 1]

        # Counter to track how many measurements have been performed
        measurement_counter = 0

        # Main loop - this loop successively executes operations onto the impure
        # quantum state object
        
        mes_list = []
        
        # if len(qc.qubits) < 30 or True:
            # qc, mes_list = extract_measurements(qc)
        
        qc, new_mes_list = insert_multiverse_measurements(qc)
        
        mes_list = mes_list + new_mes_list
            
        if iqs is None:
            # Create impure quantum state object. This object tracks multiple decoherent
            # quantum states that can appear when applying a non-unitary operation
            # iqs = ImpureQuantumState(len(qc.qubits), clbit_amount=len(qc.clbits))
            iqs = QuantumState(len(qc.qubits))
        
        mes_qubit_indices = []
        mes_clbit_indices = []
        
        total_flops = 0
        for i in range(len(qc.data)):
            total_flops += 2 ** qc.data[i].op.num_qubits
        
        progress_bar.total = total_flops
        for i in range(len(qc.data)):
            # Set alias for the instruction of this operation
            instr = qc.data[i]
            progress_bar.update(2**instr.op.num_qubits)
            # Gather the indices of the qubits from the circuits (i.e. integers instead
            # of the identifier strings)
            qubit_indices = [qc.qubits.index(qb) for qb in instr.qubits]

            # Perform instructions

            # Disentangling describes an operation, which mean that the superposition of
            # two states can be safely treated as two decoherent states. This is
            # advantageous because it might be possible that the amplitude of one
            # state is 0, which means that we halfed the workload. Even if both states
            # have non-zero amplitude, this still yields an improvement because
            # computing two decoherent states is more easily parallelized than the
            # combined coherent state
            if instr.op.name == "disentangle":
                # iqs.reset(qubit_indices[0], True)
                iqs.disentangle(qubit_indices[0])

            # If the operation is unitary, we apply this unitary on to the required
            # qubit indices
            else:

                iqs.apply_operation(instr.op, qubit_indices)

            # If all measurements have been performed, break
            if measurement_counter == measurement_amount:
                break


        mes_list.sort(key = lambda x : -qc.clbits.index(x.clbits[0]))
        
        for instr in mes_list:
            mes_qubit_indices.append(qc.qubits.index(instr.qubits[0]))
            
        if len(mes_qubit_indices):
            outcome_list, cl_prob = iqs.multi_measure(mes_qubit_indices[::-1], return_res_states = False)
            mes_qubit_indices = []
            mes_clbit_indices = []
            

        progress_bar.close()
        print("\r" + 85*" ", end=LINE_CLEAR + "\r")

        # Prepare result dictionary
        # The iqs object contains the outcome bitstrings in the attribute .outcome_list
        # and the probablities in .cl_prob. In order to ensure qiskit compatibility, we
        # reverse the bitstrings
        prob_dict = {}
        
        norm = np.sum(cl_prob)
        cl_prob = cl_prob/norm
        
            
        res = {}
        #If shots >= 1000000, no samples will be drawn and the distribution will
        #be returned instead
        if shots >= 100000:
            
            for j in range(len(outcome_list)):
                
                outcome_str = bin(outcome_list[j])[2:].zfill(len(mes_list))
                
                shot_val = int(np.round(cl_prob[j]*abs(shots)))
                
                try:
                    res[outcome_str] += shot_val
                except KeyError:
                    res[outcome_str] = shot_val

        #Generate samples
        else:
            
            from numpy.random import choice
            # p_array = np.array(list(prob_dict.values()))
            
            # samples = choice(len(p_array), shots, p=p_array)
            
            samples = choice(len(cl_prob), shots, p=cl_prob)
            
            res = {}
            
            for s in samples:
                
                outcome_str = bin(outcome_list[s])[2:].zfill(len(mes_list))
                
                if outcome_str in res:
                    res[outcome_str] += 1
                else:
                    res[outcome_str] = 1
        
        return res


def statevector_sim(qc):
    
    progress_bar = tqdm(
        desc=f"Simulating {len(qc.qubits)} qubits..",
        bar_format="{desc} |{bar}| [{percentage:3.0f}%]",
        ncols=85,
        # ascii = " >#",
        leave=False,
        delay=0.2,
        position=0,
        smoothing=1,
        file=sys.stdout
        # colour = "green"
    )
    
    LINE_CLEAR = "\x1b[2K"
    progress_bar.display()
    # This command enables fast appending. Fast appending means that the .append method
    # of the QuantumCircuit class checks much less validity conditions and is also less
    # tolerant regarding inputs.
    with fast_append():
        # We convert the circuit which is given with portable objects to a qrisp circuit

        if not isinstance(qc, QuantumCircuit):
            qc = convert_circuit(qc, "qrisp", transpile=True)

        qc = qc.copy()

        # TO-DO fix qubit ordering
        # qc.qubits = qc.qubits[::-1]

        # Count the amount of measurements (we can stop the simulation after all
        # measurements are performed)
        measurement_amount = count_measurements_and_treat_alloc(qc, insert_reset=False)

        if measurement_amount != 0:
            raise Exception(
                "Tried to determine the statevector of a circuit containing a "
                "measurement"
            )

        # Apply circuit preprocessing more
        qc = group_qc(qc)

        if len(qc.data) == 0:
            res = np.zeros(2 ** len(qc.qubits), dtype=np.complex64)
            res[0] = 1
            
            progress_bar.close()
            print(LINE_CLEAR, end="\r")
            
            return res

        qs = QuantumState(len(qc.qubits))

        total_flops = 0
        for i in range(len(qc.data)):
            total_flops += 2 ** qc.data[i].op.num_qubits

        progress_bar.total = total_flops

        # Main loop - this loop successively executes operations onto the impure
        # quantum state object
        for i in range(len(qc.data)):
            # Set alias for the instruction of this operation
            instr = qc.data[i]

            progress_bar.update(2**instr.op.num_qubits)
            # Gather the indices of the qubits from the circuits (i.e. integers instead
            # of the identifier strings)
            qubit_indices = [qc.qubits.index(qb) for qb in instr.qubits]

            # Perform instructions
            qs.apply_operation(instr.op, qubit_indices)

        res = qs.eval().tensor_array.to_array()

        progress_bar.close()
        print("\r" + 85*" ", end=LINE_CLEAR + "\r")

        # Deactivate the fast append mode
        QuantumCircuit.fast_append = False

        return res


def single_shot_sim(qc, quantum_state=None):
    if len(qc.data) == 0:
        return "", quantum_state

    # This command enables fast appending. Fast appending means that the .append method
    # of the QuantumCircuit class checks much less validity conditions and is also less
    # tolerant regarding inputs.
    with fast_append():
        # We convert the circuit which is given with portable objects to a qrisp circuit
        qc = convert_circuit(qc, "qrisp", transpile=True)

        # Treat allocation gates (ie. remove them)
        count_measurements_and_treat_alloc(qc, insert_reset=True)

        from qrisp.simulator import reorder_circuit

        qc = group_qc(qc)

        qc = reorder_circuit(qc, ["measure", "reset", "disentangle"])

        if quantum_state is None:
            # Create quantum state object.
            quantum_state = QuantumState(len(qc.qubits))

        result_str = len(qc.clbits) * ["0"]

        import random

        if len(qc.data):
            # Wrapper to pre-calculate the unitaries from the preprocessed circuit in
            # parallel with the main thread
            def pre_calc_unitaries(op):
                try:
                    op.get_unitary()
                except:
                    pass

            # pre_calc_unitaries()

            pre_calc_thr = threading.Thread(
                target=pre_calc_unitaries, args=(qc.data[0].op,)
            )
            pre_calc_thr.start()

        # Main loop - this loop successively executes operations onto the impure
        # quantum state object
        for i in range(len(qc.data)):
            pre_calc_thr.join()

            if i < len(qc.data) - 1:
                pre_calc_thr = threading.Thread(
                    target=pre_calc_unitaries, args=(qc.data[i + 1].op,)
                )
                pre_calc_thr.start()

            # Set alias for the instruction of this operation
            instr = qc.data[i]

            # Gather the indices of the qubits from the circuits (i.e. integers instead
            # of the identifier strings)
            qubit_indices = [qc.qubits.index(qb) for qb in instr.qubits]

            # Perform instructions
            if instr.op.name == "reset":
                quantum_state.measure(qubit_indices[0])

                p_0, state_0, p_1, state_1 = quantum_state.last_mes_outcome

                rng = random.random()

                if rng < p_0:
                    quantum_state = state_0
                else:
                    quantum_state = state_1

            # Disentangling describes an operation, which mean that the superposition of
            # two states can be safely treated as two decoherent states. This is
            # advantageous because it might be possible that the amplitude of one state
            # is 0, which means that we halfed the workload. Even if both states have
            # non-zero amplitude, this still yields an improvement because computing two
            # decoherent states is more easily parallelized than the combined coherent
            # state.

            elif instr.op.name == "measure":
                quantum_state.measure(qubit_indices[0])

                p_0, state_0, p_1, state_1 = quantum_state.last_mes_outcome

                rng = random.random()

                if rng < p_0:
                    quantum_state = state_0
                else:
                    quantum_state = state_1
                    result_str[qc.clbits.index(instr.clbits[0])] = "1"

            elif instr.op.name[:4] == "c_if":
                if result_str[qc.clbits.index(instr.clbits[0])] == "1":
                    quantum_state.apply_operation(instr.op, qubit_indices)
            # If the operation is unitary, we apply this unitary on to the required qubit
            # indices
            else:
                quantum_state.apply_operation(instr.op, qubit_indices)

        return "".join(result_str)[::-1], quantum_state
