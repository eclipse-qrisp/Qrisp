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

================================================================================
Qrisp Simulator Circuit Preprocessor
================================================================================

This module is responsible for optimizing and preprocessing quantum circuits 
before they are dispatched to the Qrisp statevector simulator backend. 
Because simulating large statevectors is exponentially expensive in both time 
and memory, this module applies several advanced heuristic transformations to 
significantly reduce computational overhead.

The preprocessor acts as a compiler pass, modifying the circuit's structure
without altering its mathematical outcome.

Core Functionalities:
---------------------

1. Gate Grouping (Instruction Merging)
   Applying many small unitary matrices (e.g., 1-qubit or 2-qubit gates) to a 
   massive 2^n statevector is inefficient due to high memory bandwidth usage.
   - The `GroupedInstruction` class and `group_qc` function recursively search 
     the circuit for sets of small, adjacent, or commuting gates.
   - These gates are grouped together so their combined "medium-sized" unitary 
     can be pre-calculated. Applying one medium unitary saves millions of 
     floating-point operations (FLOPs) compared to applying many small ones.
   - To make this search fast, `IntegerCircuit` translates the circuit into a 
     bitwise representation, allowing the Numba-jitted search functions 
     (`binary_get_circuit_block_jitted`, `binary_get_circuit_block_jitted_chunked`) 
     to evaluate gate commutativity using ultra-fast bitwise logic. 
     It features a dual-path that vectorizes qubit bitmasks into chunks to bypass 
     64-bit memory limitations on massive statevector simulations.

2. State Disentangling
   Simulating a 50+ qubit statevector is practically impossible if fully entangled. 
   However, many algorithms naturally "disentangle" certain qubits during execution 
   (e.g., via measurements, resets, or specific uncomputations).
   - `insert_disentangling` identifies points in the circuit where wave-function 
     branches no longer interact.
   - It inserts a custom `disentangle` instruction. The simulator catches this and 
     splits the massive simulation into smaller, separate, parallelizable wave-functions, 
     effectively turning an intractable problem into a solvable one.

3. Measurement and Allocation Management
   - `extract_measurements` and `count_measurements_and_treat_alloc` optimize 
     how classical measurements and temporary qubit allocations are handled.
   - `insert_multiverse_measurements` handles deferred measurement patterns by 
     introducing ancilla qubits and CNOT gates, ensuring probability distributions 
     are correctly captured without breaking coherence prematurely.

4. The Main Wrapper: `circuit_preprocessor(qc)`
   The main entry point for this module. It evaluates the incoming circuit, 
   applies disentangling (if the circuit is dangerously wide, e.g., >45 qubits), 
   groups the gates for performance, and finally reorders the circuit to safely 
   push measurements, resets, and disentanglers to the end of execution blocks.
================================================================================
"""

import numpy as np
from numba import njit
import math

from qrisp.circuit import (
    ClControlledOperation,
    CXGate,
    Instruction,
    Measurement,
    Operation,
    QuantumCircuit,
)
from qrisp.permeability.type_checker import is_permeable

memory_bandwidth_penalty = 2


class GroupedInstruction:
    def __init__(self, int_qc, indices, qubits=None):
        self.gate_signature_list = []

        if qubits is None:
            if int_qc.use_chunks:
                qubit_set = np.zeros(int_qc.num_chunks, dtype=np.int64)
                for i in range(len(indices)):
                    qubit_set |= int_qc.data[indices[i]]
            else:
                qubit_set = 0
                for i in range(len(indices)):
                    qubit_set |= int_qc.data[indices[i]]
            
            self.qubits = int_to_qb_set_generic(qubit_set, int_qc.source)
        else:
            self.qubits = list(set(qubits))

        self.instr_list = int_qc.source.data
        self.indices = indices

        self.gain = 0
        for i in range(len(indices)):
            self.gain += 1 << self.instr_list[indices[i]].op.num_qubits

        self.gain = self.gain - 2 ** len(self.qubits) * 0.45

    def get_instruction(self):
        temp_qc = QuantumCircuit()
        temp_qc.qubits = self.qubits
        
        # Eliminated the slow try...except block with an O(1) set lookup
        added_clbits = set(temp_qc.clbits)

        for i in range(len(self.indices)):
            for cb in self.instr_list[self.indices[i]].clbits:
                if cb not in added_clbits:
                    temp_qc.add_clbit(cb)
                    added_clbits.add(cb)

            temp_qc.append(self.instr_list[self.indices[i]])
            
        self.instruction = Instruction(temp_qc.to_op(), temp_qc.qubits, temp_qc.clbits)
        return self.instruction


def group_qc(qc):
    print("Grouping circuit with " + str(len(qc.qubits)) + " qubits and " + str(len(qc.data)) + " instructions")
    max_recursion_depth = optimal_grouping_recursion_parameter(len(qc.qubits)) + 12
    
    int_qc = IntegerCircuit(qc)
    num_instructions = len(int_qc.data)
    processed = np.zeros(num_instructions, dtype=bool)
    
    # NEW: Create a new list for the final circuit
    final_data = [] 
    
    current_idx = 0 
    while current_idx < num_instructions:
        if processed[current_idx]:
            current_idx += 1
            continue

        if not int_qc.is_unitary[current_idx]: # Ensure your IntegerCircuit has the is_unitary array
            final_data.append(int_qc.source.data[current_idx])
            processed[current_idx] = True
            current_idx += 1
            continue

        group = find_group(int_qc, max_recursion_depth, current_idx, processed)
        
        # Append the new grouped instruction
        final_data.append(group.get_instruction())

        for idx in group.indices:
            processed[idx] = True
            
        current_idx += 1

    # Replace the circuit data with the newly constructed list
    grouped_qc = qc.clearcopy()
    grouped_qc.data = final_data # Ensure this is assigned directly
    
    print("Grouping complete. Resulting circuit has " + str(len(grouped_qc.data)) + " instructions")
    return grouped_qc


def find_group(int_qc, max_recursion_depth, current_idx, processed):
    traversed_qb_sets = []
    # For the chunked path int_qc.data[i] is a row-view of the 2D array; copy
    # it so that any in-place modifications inside the jitted helper do not
    # corrupt int_qc.data for subsequent instructions.
    initial_qubits = int_qc.data[current_idx].copy() if int_qc.use_chunks else int_qc.data[current_idx]
    
    options = find_grouping_options(
        int_qc=int_qc, 
        traversed_qb_sets=traversed_qb_sets, 
        max_recursion_depth=max_recursion_depth, 
        qubits=initial_qubits, 
        established_indices=[current_idx],
        processed=processed,
        current_idx=current_idx
    )
    
    best_gain = -float("inf")
    best_group = options[0]
    for opt in options:
        if opt.gain > best_gain:
            best_gain = opt.gain
            best_group = opt
            
    return best_group


def find_grouping_options(int_qc, traversed_qb_sets, max_recursion_depth, qubits, established_indices, processed, current_idx):
    hashable_qubits = tuple(qubits) if isinstance(qubits, np.ndarray) else qubits
    traversed_qb_sets.append(hashable_qubits)

    instruction_indices, expansion_options = get_circuit_block_(
        int_qc, qubits, established_indices, processed, current_idx
    )

    options = [GroupedInstruction(int_qc, instruction_indices, int_to_qb_set_generic(qubits, int_qc.source))]

    if len(expansion_options) == 0 or max_recursion_depth == 0 or len(int_to_qb_set_generic(qubits, int_qc.source)) >= 7:
        return options

    for i in range(len(expansion_options)):
        opt_int = qb_set_to_int_generic([expansion_options[i]], int_qc)
        proposed_set = qubits | opt_int

        prop_hashable = tuple(proposed_set) if isinstance(proposed_set, np.ndarray) else proposed_set

        if prop_hashable not in traversed_qb_sets:
            options += find_grouping_options(
                int_qc=int_qc,
                traversed_qb_sets=traversed_qb_sets,
                max_recursion_depth=max_recursion_depth - 1,
                qubits=proposed_set,
                established_indices=instruction_indices,
                processed=processed,
                current_idx=current_idx
            )

    return options


def get_circuit_block_(int_qc, qubits, established_indices, processed, current_idx):
    n = int_qc.n

    if int_qc.use_chunks:
        # Pass a copy: the chunked jitted function modifies the qubits array
        # in-place (qubits[c] = ...) and Numba propagates that back to the
        # caller.  Without a copy the caller's qubits variable would be
        # corrupted, causing GroupedInstruction to miss qubit entries.
        instruction_indices, expansion_options = binary_get_circuit_block_jitted_chunked(
            int_qc.data, int_qc.is_unitary, qubits.copy(), np.array(established_indices, dtype=np.int64), processed, current_idx, int_qc.num_chunks
        )
    else:
        instruction_indices, expansion_options = binary_get_circuit_block_jitted(
            int_qc.data, int_qc.is_unitary, qubits, np.array(established_indices, dtype=np.int64), processed, current_idx
        )

    return instruction_indices, int_to_qb_set_generic(expansion_options, int_qc.source)


# ==============================================================================
# JITTED CORE FUNCTIONS (DUAL-PATH ARCHITECTURE)
# ==============================================================================

@njit(cache=True)
def binary_get_circuit_block_jitted(int_qc_data, is_unitary, qubits, established_indices, processed, current_idx):
    """Path A: Ultra-fast scalar bitwise logic for circuits < 63 qubits"""
    expansion_options = 0
    instruction_indices = []
    ee_counter = 0
    window_size = 100
    end_idx = min(current_idx + window_size, len(int_qc_data))

    for i in range(current_idx, end_idx):
        is_established = False
        if ee_counter < len(established_indices):
            if established_indices[ee_counter] == i:
                is_established = True

        if processed[i] and not is_established:
            continue
        if qubits == 0:
            break
        if is_established:
            ee_counter += 1
            instruction_indices.append(i)
            continue

        instr_qubits = int_qc_data[i]
        intersection = qubits & instr_qubits

        if not intersection:
            continue

        # If it is a non-unitary gate, mask out the affected qubits
        if not is_unitary[i]:
            qubits = qubits & (~instr_qubits)
            continue

        if (~qubits & instr_qubits) == 0:
            instruction_indices.append(i)
        else:
            qubits = qubits & (~intersection)
            expansion_options = expansion_options | (instr_qubits & (~intersection))

    return instruction_indices, expansion_options


@njit(cache=True)
def binary_get_circuit_block_jitted_chunked(int_qc_data, is_unitary, qubits, established_indices, processed, current_idx, num_chunks):
    """Path B: Chunked-Vector logic for massive circuits >= 63 qubits"""
    expansion_options = np.zeros(num_chunks, dtype=np.int64)
    instruction_indices = []
    ee_counter = 0
    window_size = 100
    end_idx = min(current_idx + window_size, len(int_qc_data))

    for i in range(current_idx, end_idx):
        is_established = False
        if ee_counter < len(established_indices):
            if established_indices[ee_counter] == i:
                is_established = True

        if processed[i] and not is_established:
            continue

        is_zero = True
        for c in range(num_chunks):
            if qubits[c] != 0:
                is_zero = False
                break
        if is_zero:
            break

        if is_established:
            ee_counter += 1
            instruction_indices.append(i)
            continue

        instr_qubits = int_qc_data[i]
        
        has_intersection = False
        for c in range(num_chunks):
            if (qubits[c] & instr_qubits[c]) > 0:
                has_intersection = True
                break
        if not has_intersection:
            continue

        if not is_unitary[i]:
            for c in range(num_chunks):
                qubits[c] = qubits[c] & (~instr_qubits[c])
            continue

        is_subset = True
        for c in range(num_chunks):
            if (~qubits[c] & instr_qubits[c]) > 0:
                is_subset = False
                break

        if is_subset:
            instruction_indices.append(i)
        else:
            for c in range(num_chunks):
                intersection_c = qubits[c] & instr_qubits[c]
                qubits[c] = qubits[c] & (~intersection_c)
                expansion_options[c] = expansion_options[c] | (instr_qubits[c] & (~intersection_c))

    return instruction_indices, expansion_options


# ==============================================================================
# HELPER AND TRANSLATION FUNCTIONS
# ==============================================================================

def int_to_qb_set_generic(data, qc):
    res = []
    if isinstance(data, np.ndarray):
        chunk_size = 62
        for c, chunk in enumerate(data):
            temp = int(chunk)
            bit_idx = 0
            while temp > 0:
                if temp & 1:
                    qb_idx = c * chunk_size + bit_idx
                    if qb_idx < len(qc.qubits):
                        res.append(qc.qubits[qb_idx])
                temp = temp >> 1
                bit_idx += 1
    else:
        temp = int(data)
        for i in range(len(qc.qubits)):
            if temp & 1:
                res.append(qc.qubits[i])
            temp = temp >> 1
            if temp == 0:
                break
    return res


def qb_set_to_int_generic(qubits, int_qc):
    if int_qc.use_chunks:
        res = np.zeros(int_qc.num_chunks, dtype=np.int64)
        for qb in qubits:
            idx = int_qc.qb_to_index[qb]
            c_idx = idx // int_qc.chunk_size
            b_idx = idx % int_qc.chunk_size
            res[c_idx] |= (1 << b_idx)
        return res
    else:
        res = 0
        for qb in qubits:
            res |= 1 << int_qc.qb_to_index[qb]
        return res


class IntegerCircuit:
    def __init__(self, qc):
        self.source = qc
        self.qb_to_index = {qc.qubits[i]: i for i in range(len(qc.qubits))}
        self.n = len(qc.qubits)
        self.chunk_size = 62
        
        is_unitary_list = []
        for instr in qc.data:
            is_u = not (instr.op.name in ["measure", "reset", "disentangle"] or isinstance(instr.op, ClControlledOperation))
            is_unitary_list.append(is_u)
            
        self.is_unitary = np.array(is_unitary_list, dtype=np.bool_)
        
        if self.n < 63:
            self.use_chunks = False
            res_list = []
            for instr in qc.data:
                res = 0
                for qb in instr.qubits:
                    res |= 1 << self.qb_to_index[qb]
                res_list.append(res)
            self.data = np.array(res_list, dtype=np.int64)
        else:
            self.use_chunks = True
            self.num_chunks = int(math.ceil(self.n / self.chunk_size))
            res_list = []
            for instr in qc.data:
                res = np.zeros(self.num_chunks, dtype=np.int64)
                for qb in instr.qubits:
                    idx = self.qb_to_index[qb]
                    res[idx // self.chunk_size] |= (1 << (idx % self.chunk_size))
                res_list.append(res)
            self.data = np.array(res_list, dtype=np.int64)


def optimal_grouping_recursion_parameter(qubit_amount):
    if qubit_amount <= 16: return 2
    elif 16 < qubit_amount <= 20: return 3
    elif 20 < qubit_amount <= 24: return 4
    elif 24 < qubit_amount <= 28: return 6
    elif 28 < qubit_amount <= 32: return 7
    elif 32 < qubit_amount < 35: return 8
    else: return 2


# ==============================================================================
# DISENTANGLING AND MEASUREMENT OPERATIONS
# ==============================================================================

disentangler = Operation("disentangle", num_qubits=1)
disentangler.definition = QuantumCircuit(1)
disentangler.permeability = {0: False}
disentangler.warning = False

def Disentangler(warning=False):
    d = Operation("disentangle", num_qubits=1)
    d.definition = QuantumCircuit(1)
    d.permeability = {0: False}
    d.warning = warning
    return d

def insert_disentangling(qc):
    for i in range(len(qc.qubits)):
        qc.reset(qc.qubits[i])

    reversed_data = list(qc.data)[::-1]
    disentangling_counter = 0
    i = 0
    
    while i < len(reversed_data):
        if reversed_data[i].op.name not in ["measure", "reset"]:
            i += 1
            continue

        qubit = reversed_data[i].qubits[0]
        j = int(i)

        while j < len(reversed_data):
            instr = reversed_data[j]
            if qubit in instr.qubits:
                if instr.op.name in ["measure", "reset", "disentangle"]:
                    j += 1
                    continue

                qubit_index = instr.qubits.index(qubit)

                if is_permeable(instr.op, [qubit_index]):
                    reversed_data.insert(j + 1, Instruction(disentangler, [qubit]))
                    disentangling_counter += 1
                    j += 1
                else:
                    break
            j += 1
        i += 1

    new_qc = qc.clearcopy()
    new_qc.data = reversed_data[::-1]
    return new_qc

def count_measurements_and_treat_alloc(qc, insert_reset=True):
    counter = 0
    i = 0
    while i < len(qc.data):
        instr = qc.data[i]
        if instr.op.name == "barrier":
            qc.data.pop(i)
            continue
        if instr.op.name == "measure":
            counter += 1
        elif instr.op.name == "qb_alloc":
            qc.data.pop(i)
            continue
        elif instr.op.name == "qb_dealloc":
            qc.data.pop(i)
            if insert_reset:
                qc.data.insert(i, Instruction(Disentangler(True), qubits=instr.qubits))
            else:
                continue
        i += 1
    return counter


def extract_measurements(qc):
    qubits = list(qc.qubits)
    clbits = list(qc.clbits)
    mes_list = []
    data = []
    for instr in qc.data[::-1]:
        if instr.op.name == "measure" and instr.qubits[0] in qubits and instr.clbits[0] in clbits:
            mes_list.append(instr)
        else:
            data.append(instr)

        for qb in instr.qubits:
            try: qubits.remove(qb)
            except: pass
        for cb in instr.clbits:
            try: clbits.remove(cb)
            except: pass

    new_qc = qc.clearcopy()
    new_qc.data = data[::-1]
    return new_qc, mes_list


def insert_multiverse_measurements(qc):
    new_data = []
    new_measurements = []
    cb_to_qb_dic = {}
    data = list(qc.data)
    
    while data:
        instr = data.pop(0)
        
        if instr.op.name == "measure":
            meas_qubit = instr.qubits[0]
            meas_clbit = instr.clbits[0]

            next_instr_is_reset = False
            for j in range(len(data)):
                if meas_qubit in data[j].qubits:
                    if data[j].op.name == "reset":
                        next_instr_is_reset = True
                        data.pop(j)
                        break
                    elif not is_permeable(data[j].op, [data[j].qubits.index(meas_qubit)]):
                        break
                if meas_clbit in data[j].clbits and not isinstance(data[j], ClControlledOperation):
                    break
                if data[j].op.name == "measure" and data[j].qubits[0] == meas_qubit:
                    break
            else:
                new_data.append(Instruction(disentangler, [meas_qubit]))
                new_measurements.append((instr.qubits[0], instr.clbits[0]))
                continue

            qb = qc.add_qubit()
            new_data.append(Instruction(CXGate(), instr.qubits + [qb]))

            if next_instr_is_reset:
                new_data.append(Instruction(CXGate(), [qb] + instr.qubits))
                new_data.append(Instruction(disentangler, [meas_qubit]))

            cb_to_qb_dic[instr.clbits[0]] = qb
            mes_instr = instr.copy()
            mes_instr.qubits = [qb]

        elif instr.op.name == "reset":
            meas_qubit = instr.qubits[0]
            new_data.append(Instruction(Disentangler(warning=False), [meas_qubit]))

            for j in range(len(data)):
                if meas_qubit in data[j].qubits:
                    if not is_permeable(data[j].op, [data[j].qubits.index(meas_qubit)]):
                        break
            else:
                continue

            qb = qc.add_qubit()
            new_data.append(Instruction(CXGate(), instr.qubits + [qb]))
            new_data.append(Instruction(CXGate(), [qb] + instr.qubits))
            new_data.append(Instruction(Disentangler(), [qb]))

        elif isinstance(instr.op, ClControlledOperation):
            new_qubits = []
            ctrl_state = instr.op.ctrl_state
            control_qubits = []
            for j in range(len(instr.clbits)):
                cb = instr.clbits[j]

                if cb not in cb_to_qb_dic:
                    if ctrl_state[j] == "1":
                        break
                    else:
                        qb = qc.add_qubit()
                        new_qubits.append(qb)
                        control_qubits.append(qb)
                else:
                    control_qubits.append(cb_to_qb_dic[cb])
            else:
                new_data.append(
                    Instruction(
                        instr.op.base_op.control(len(control_qubits), ctrl_state=ctrl_state),
                        control_qubits + instr.qubits,
                    )
                )

            for qb in control_qubits:
                new_data.append(Instruction(disentangler, [qb]))
        else:
            new_data.append(instr)

    for cb, qb in cb_to_qb_dic.items():
        new_measurements.append((qb, cb))

    new_measurements = list(set(new_measurements))
    measurements = []
    for qb, cb in new_measurements:
        measurements.append(Instruction(Measurement(), [qb], [cb]))
        
    qc.data = new_data
    return qc, measurements


def circuit_preprocessor(qc):
    from qrisp.simulator import reorder_circuit
    if len(qc.data) == 0:
        return qc.copy()

    if len(qc.qubits) > 45:
        qc = insert_disentangling(qc)
        
    qc = group_qc(qc)
    return reorder_circuit(qc, ["measure", "reset", "disentangle"])