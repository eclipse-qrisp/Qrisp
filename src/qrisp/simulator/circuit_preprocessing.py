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
before they are dispatched to a backend (primarily the Qrisp statevector simulator). 
Because simulating large statevectors is exponentially expensive in both time 
and memory, this module applies several advanced heuristic transformations to 
significantly reduce computational overhead.

The preprocessor acts as a compiler pass, modifying the circuit's abstract 
syntax tree (AST) without altering its mathematical outcome.

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
     (`binary_get_circuit_block_jitted`) to evaluate gate commutativity using 
     ultra-fast bitwise logic.

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


# This class is supposed to describe a group of instructions
# The idea behind the grouping is that grouping instructions together allows
# to precalculate their unitary. This saves alot of time because applying
# a medium size unitary on a large statevector is more efficient than applying
# many small unitaries. This estimation is elaborated in the calc_gain method.
class GroupedInstruction:
    # The constructor takes a list of instruction (for instance from a quantum circuit)
    # and a list of indices, which describe which instruction to include in the group
    # Using the qubits argument, is possible to provide a list of qubits, where the
    # instruction are acting on. Otherwise, the (rather slow) .merge method is used
    def __init__(self, int_qc, indices, qubits=None):
        self.gate_signature_list = []

        if qubits == None:
            qubit_set = 0

            for i in range(len(indices)):
                qubit_set |= int_qc.data[indices[i]]

            qubit_set = int_to_qb_set(int(qubit_set), int_qc.source)
        else:
            qubit_set = set(qubits)

        self.qubits = list(qubit_set)

        self.instr_list = int_qc.source.data

        self.indices = indices

        # We now calculate the gain by estimating how many floating point operations are
        # performed for the grouped circuit vs the non-grouped circuit

        # Consider a statevector s of size 2**n and k small unitaries U_i of shape
        # (2**l, 2**l). Applying one of the unitaries to the large statevector can be
        # understood as a block matrix application
        # (U_i, 0, 0, 0)
        # (0, U_i, 0, 0)
        # (0, 0, U_i, 0)
        # (0, 0, 0, U_i)
        # i.e. 2**(n-l) applications of a (2**l, 2**l) matrix.

        # Counting the floating point operations we have 2**(n-l)*(2**l)**2 = 2**(n+l)
        # for a single matrix multiplication. Therefore, accounting for all k unitaries,
        # we have FLOPS = k*2**(n+l). If we now group the k unitaries into a medium
        # sized unitary of shape (2**L, 2**L) we therefore get a FLOP count of
        # FLOPS = 2**(n+L). Note that this assumes that calculating the medium-sized
        # unitary can be calculated for free. As it turns out, for the scales where the
        # pythonic slowdown of this algorithm does not decrease the speed, this
        # assumption is largely valid. (Check the calc_circuit_unitary function of
        # unitary_management.py)

        # Since both FLOP counts scale with 2**n, we simply calculate k*2**l
        # (or better the SUM instead of just taking the product) and 2**L

        self.gain = 0
        for i in range(len(indices)):
            self.gain += 1 << self.instr_list[indices[i]].op.num_qubits  # + memory_bandwidth_penalty

        self.gain = self.gain - 2 ** len(self.qubits) * 0.45

    def get_instruction(self):
        # Create the QuantumCircuit which describes this group of instructions

        # Else create a quantum circuit from the given qubits
        temp_qc = QuantumCircuit()
        temp_qc.qubits = self.qubits

        for i in range(len(self.indices)):
            for cb in self.instr_list[self.indices[i]].clbits:
                try:
                    temp_qc.add_clbit(cb)
                except:
                    pass

            temp_qc.append(self.instr_list[self.indices[i]])
        # Create instruction
        self.instruction = Instruction(temp_qc.to_op(), temp_qc.qubits, temp_qc.clbits)

        return self.instruction


# The idea is now to iterate through different groupings and find the one with the most
# gain.
def group_qc(qc):
    max_recursion_depth = optimal_grouping_recursion_parameter(len(qc.qubits)) + 12
    grouped_instr_list = []
    
    int_qc = IntegerCircuit(qc)
    num_instructions = len(int_qc.data)
    
    # NEW: Keep track of which instructions have been grouped
    processed = np.zeros(num_instructions, dtype=bool)
    
    # NEW: Index pointer instead of while qc.data:
    current_idx = 0 
    
    while current_idx < num_instructions:
        # Skip this instruction if it was already swallowed by a previous group
        if processed[current_idx]:
            current_idx += 1
            continue

        # If the instruction is non-unitary (less than 0 in the bitwise representation)
        if int_qc.data[current_idx] < 0:
            grouped_instr_list.append(int_qc.source.data[current_idx])
            processed[current_idx] = True
            current_idx += 1
            continue

        # Find the best grouping
        # Note: We now need to pass current_idx and the processed mask down the chain
        group = find_group(int_qc, max_recursion_depth, current_idx, processed)

        # Append the grouped instruction to the result list
        grouped_instr_list.append(group.get_instruction())

        # NEW: Mark all indices in this group as processed instead of deleting them
        for idx in group.indices:
            processed[idx] = True
            
        current_idx += 1

    # Create resulting circuit
    grouped_qc = qc.clearcopy()
    for i in range(len(grouped_instr_list)):
        grouped_qc.data.append(grouped_instr_list[i])

    return grouped_qc


# This function determines a set of grouping options and chooses the best option
def find_group(int_qc, max_recursion_depth, current_idx, processed):
    traversed_qb_sets = []
    initial_qubits = int_qc.data[current_idx]
    
    # Get all valid grouping combinations recursively
    options = find_grouping_options(
        int_qc=int_qc, 
        traversed_qb_sets=traversed_qb_sets, 
        max_recursion_depth=max_recursion_depth, 
        qubits=initial_qubits, 
        established_indices=[current_idx],
        processed=processed,
        current_idx=current_idx
    )
    
    # Find the grouping option with the highest computational gain
    best_gain = -float("inf")
    best_group = options[0]
    for opt in options:
        if opt.gain > best_gain:
            best_gain = opt.gain
            best_group = opt
            
    return best_group


# The groupings are determined by choosing a set of qubits and then trying which
# instructions can be executed on these qubits without "leaving" this set of qubits.
def find_grouping_options(int_qc, traversed_qb_sets, max_recursion_depth, qubits, established_indices, processed, current_idx):
    traversed_qb_sets.append(qubits)

    # Pass processed and current_idx down to the scanning function
    instruction_indices, expansion_options = get_circuit_block_(
        int_qc, qubits, established_indices, processed, current_idx
    )

    options = [GroupedInstruction(int_qc, instruction_indices, int_to_qb_set(qubits, int_qc.source))]

    if len(expansion_options) == 0 or max_recursion_depth == 0 or len(int_to_qb_set(qubits, int_qc.source)) >= 7:
        return options

    for i in range(len(expansion_options)):
        proposed_set = qubits | qb_set_to_int([expansion_options[i]], int_qc.qb_to_index)

        if proposed_set not in traversed_qb_sets:
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


# This function returns the indices of the instructions of a circuit,
# where the returned can be executed by only "staying inside" a given set of qubits
def get_circuit_block(qc_data, qubits, established_indices=[]):
    # Copy set in order to prevent modification
    qubits = set(qubits)

    # Set up set of expansion options
    expansion_options = set([])

    # Set up result list
    instruction_indices = []

    # Now iterate through all the given instructions
    for i in range(len(qc_data)):
        if len(qubits) == 0:
            break

        # If the instruction has been identified as part of the group
        # in a previous recursion, skip the checking and add to the list
        # of instructions
        if len(established_indices):
            if established_indices[0] == i:
                instruction_indices.append(established_indices.pop(0))
                continue

        # Set some aliases
        instr = qc_data[i]
        instr_qubits = set(instr.qubits)

        # Determine the intersection between the qubits of the instruction
        # and the qubits of the group
        intersection = instr_qubits.intersection(qubits)

        # If the intersection is empty, this instruction is not part of the group
        # and no further action needs to be taken
        if not intersection:
            continue

        # If the instruction is non-unitary, no further instruction
        # on this qubit can be part of the group
        if instr.op.name in ["measure", "reset", "disentangle"]:
            qubits = qubits - instr_qubits
            continue

        # If the instruction qubits are part of the group qubits,
        # add the instruction to the group
        if instr_qubits.issubset(qubits):
            instruction_indices.append(i)

        # Otherwise, the instruction happens partly on the group qubits,
        # partly outside. Therefore, we need to remove the qubits
        # that interact with the outside.
        # Nevertheless, we add the "outside" qubits to the set of expansion options
        else:
            qubits = qubits - intersection
            expansion_options = expansion_options.union(instr_qubits - intersection)

    # Return result
    return instruction_indices, list(expansion_options)


# Empirically determined parameters that seem to work best
def optimal_grouping_recursion_parameter(qubit_amount):
    if qubit_amount <= 16:
        return 2
    elif 16 < qubit_amount <= 20:
        return 3
    elif 20 < qubit_amount <= 24:
        return 4
    elif 24 < qubit_amount <= 28:
        return 6
    elif 28 < qubit_amount <= 32:
        return 7
    elif 32 < qubit_amount < 35:
        return 8
    else:
        return 9


# This function inserts disentangling operations into the circuit, if suited
# Consider the following circuit
#             ┌───┐
# qubit_3362: ┤ H ├──■──────────────────────
#             └───┘┌─┴─┐┌───┐
# qubit_3363: ─────┤ x ├┤ H ├──■────────────
#                  └───┘└───┘┌─┴─┐┌────────┐
# qubit_3364: ───────────────┤ x ├┤ P(0.3) ├
#                            └───┘└────────┘
# If we don't the gates after the first CNOT with U,
# the resulting state is
# 2**-0.5*(|0>U|0> + |1>U|1>)
# Since we are not interested in the wave-function but in the resulting
# probabilities, it is now safe to evaluate both summands of the wave function
# separately, ie. evaluate |0>U|0> and |1>U|1> and multiply the resulting,
# probabilities with 2**-0.5.
# This is possible because there is not going to be any interaction between the to
# states.
# (TO-DO better proof required)
# Therefore the workflow for performing the disentangling is
# 1. Identify position to insert disentangler (here: directly behind the first CNOT)
# 2. Perform measurement (to acquire classical probabilities)
# 3. Continue the simulation on the two decoherent states

# Simulating both states separately has two advantages
# 1. If one of the branches has probability zero, this will show up in the measurement
#   and we don't have to continue the simulation of this branch
# 2. Simulating two states that don't interact is easier to parallelize

# As it turns out, it is even possible to disentangle this circuit right
# after the first H gate on qubit zero


#              ┌───┐     ┌────────┐
# qubit_10119: ┤ H ├──■──┤ P(0.2) ├─────────────────■──
#              └───┘┌─┴─┐└─┬───┬──┘                 │
# qubit_10120: ─────┤ x ├──┤ H ├─────■──────────────┼──
#                   └───┘  └───┘   ┌─┴─┐┌────────┐┌─┴─┐
# qubit_10121: ────────────────────┤ x ├┤ P(0.3) ├┤ x ├
#                                  └───┘└────────┘└───┘


# This is because all the gates after the H-gate are permeable on this qubit
# (for an elaboration on permeability check the uncomputation module)
# Roughly said, a gate U which is permeable on the first qubit behaves like this
# U|0>|a> = exp(i*phi_0) |0> U_0 |a>
# U|1>|b> = exp(i*phi_1) |0> U_1 |b>

# Create disentangling operation
disentangler = Operation("disentangle", num_qubits=1)
disentangler.definition = QuantumCircuit(1)
disentangler.permeability = {0: False}
disentangler.warning = False


def Disentangler(warning=False):
    disentangler = Operation("disentangle", num_qubits=1)
    disentangler.definition = QuantumCircuit(1)
    disentangler.permeability = {0: False}
    disentangler.warning = warning
    return disentangler


def insert_disentangling(qc):
    # This function checks for permeability on a given qubit
    from qrisp.permeability import is_permeable

    # After all the operations have been performed on a qubit,
    # it can be reset without changing the statistics
    for i in range(len(qc.qubits)):
        qc.reset(qc.qubits[i])
        # pass
        # qc.append(Disentangler(), [qc.qubits[i]])

    # return qc

    # We now insert the disentanglers at every position which
    # is only separated by permeable operations to the final operation
    # of that qubit
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

                # Find qubit index
                qubit_index = instr.qubits.index(qubit)

                if is_permeable(instr.op, [qubit_index]):
                    # If so, insert disentangler before this instruction
                    reversed_data.insert(j + 1, Instruction(disentangler, [qubit]))
                    disentangling_counter += 1
                    j += 1

                else:
                    break
            j += 1

        i += 1

    # #For every qubit, we traverse the circuit from the back,
    # #until we find an operation which is not permeable on that qubit
    # for i in range(len(qc.qubits)):

    #     #Set qubit alias
    #     qb = qc.qubits[i]
    #     j = -1
    #     while j < len(reversed_data)-1:
    #         j += 1

    #         #Set alias for instruction that is to be checked
    #         instr = reversed_data[j]

    #         #If the instruction is a reset, measurement or disentangling,
    #         #it performs the same operation as a disentangler, therefore
    #         #we can safely insert further disentanlger before it
    #         if instr.op.name in ["measure", "reset", "disentangle"]:
    #             continue

    #         #If the instruction acts on the qubit in question,
    #         #check if the instruction is permeable on this qubit
    #         if qb in instr.qubits:

    #             #Find qubit index
    #             qubit_index = instr.qubits.index(qb)

    #             #Check for permeability
    #             if is_permeable(instr.op, [qubit_index]):

    #                 #If so, insert disentangler before this instruction
    #                 reversed_data.insert(j+1, Instruction(disentangler, [qb]))
    #                 disentangling_counter += 1
    #             #Otherwise, we can no longer insert disentanglers
    #             else:
    #                 break

    # Create output quantum circuit
    new_qc = qc.clearcopy()
    new_qc.data = reversed_data[::-1]

    return new_qc


# Simple function to count the amount of measurements in a circuit,
# this is helpfull, because we don't need to continue the simulation,
# once all measurements have been performed
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
            pass
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


def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


def qb_set_to_int(qubits, qb_to_index):
    res = 0
    for qb in qubits:
        res |= 1 << qb_to_index[qb]
    return res


def qc_to_int_list(qc, qb_to_index):
    res_list = []
    for instr in qc.data:
        res_list.append(qb_set_to_int(instr.qubits, qb_to_index))
        if instr.op.name in ["measure", "reset", "disentangle"] or (isinstance(instr.op, ClControlledOperation)):
            res_list[-1] *= -1

    return res_list


# Copy set in order to prevent modification
# qubits = set(qubits)
def get_circuit_block_(int_qc, qubits, established_indices, processed, current_idx):
    n = int_qc.n

    if isinstance(int_qc.data, np.ndarray):
        instruction_indices, expansion_options = binary_get_circuit_block_jitted(
            int_qc.data, qubits, n, np.array(established_indices, dtype=np.int64), processed, current_idx
        )
    else:
        instruction_indices, expansion_options = binary_get_circuit_block(
            int_qc.data, qubits, n, np.array(established_indices, dtype=np.int64), processed, current_idx
        )

    return instruction_indices, int_to_qb_set(expansion_options, int_qc.source)


@njit(cache=True)
def binary_get_circuit_block_jitted(int_qc_list, qubits, n, established_indices, processed, current_idx):
    expansion_options = 0
    instruction_indices = []
    ee_counter = 0
    
    # Lookahead window to prevent massive deep-circuit overhead
    window_size = 1000
    end_idx = min(current_idx + window_size, len(int_qc_list))

    for i in range(current_idx, end_idx):
        # Determine if this index is one of the explicitly established ones
        is_established = False
        if ee_counter < len(established_indices):
            if established_indices[ee_counter] == i:
                is_established = True

        # If it was already processed in a previous group, ignore it!
        if processed[i] and not is_established:
            continue

        if qubits == 0:
            break

        if is_established:
            ee_counter += 1
            instruction_indices.append(i)
            continue

        instr_qubits = int_qc_list[i]
        intersection = qubits & instr_qubits

        if not intersection:
            continue

        if instr_qubits < 0:
            qubits = qubits & (((1 << n) - 1) ^ instr_qubits)
            continue

        if not bool((((1 << n) - 1) ^ qubits) & instr_qubits):
            instruction_indices.append(i)
        else:
            qubits = qubits & (((1 << n) - 1) ^ intersection)
            expansion_options = expansion_options | (instr_qubits & (((1 << n) - 1) ^ intersection))

    return instruction_indices, expansion_options


def binary_get_circuit_block(int_qc_list, qubits, n, established_indices, processed, current_idx):
    # Identical logic to the jitted version, for cases where np arrays aren't used
    expansion_options = 0
    instruction_indices = []
    ee_counter = 0
    
    window_size = 1000
    end_idx = min(current_idx + window_size, len(int_qc_list))

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

        instr_qubits = int_qc_list[i]
        intersection = qubits & instr_qubits

        if not intersection:
            continue

        if instr_qubits < 0:
            qubits = qubits & (((1 << n) - 1) ^ instr_qubits)
            continue

        if not bool((((1 << n) - 1) ^ qubits) & instr_qubits):
            instruction_indices.append(i)
        else:
            qubits = qubits & (((1 << n) - 1) ^ intersection)
            expansion_options = expansion_options | (instr_qubits & (((1 << n) - 1) ^ intersection))

    return instruction_indices, expansion_options


def int_to_qb_set(integer, qc):
    res = []
    temp = int(integer)
    for i in range(len(qc.qubits)):
        if temp & 1:
            res.append(qc.qubits[i])
        temp = temp >> 1
        if temp == 0:
            break

    return res


class IntegerCircuit:
    def __init__(self, qc):
        self.source = qc
        self.qb_to_index = {qc.qubits[i]: i for i in range(len(qc.qubits))}
        self.data = qc_to_int_list(qc, self.qb_to_index)
        self.n = len(qc.qubits)


def average_group_size(qc):
    average_group_size = 0

    for i in range(len(qc.data)):
        if qc.data[i].op.definition:
            average_group_size += len(qc.data[i].op.definition.data)
        else:
            average_group_size += 1

    return average_group_size / len(qc.data)


def extract_measurements(qc):

    qubits = list(qc.qubits)
    clbits = list(qc.clbits)
    mes_list = []
    data = []
    mes_dic = {}
    for instr in qc.data[::-1]:
        if instr.op.name == "measure" and instr.qubits[0] in qubits and instr.clbits[0] in clbits:
            mes_list.append(instr)
        else:
            data.append(instr)

        for qb in instr.qubits:
            try:
                qubits.remove(qb)
            except:
                pass

        for cb in instr.clbits:
            try:
                clbits.remove(cb)
            except:
                pass

    new_qc = qc.clearcopy()

    new_qc.data = data[::-1]

    return new_qc, mes_list


def insert_multiverse_measurements(qc):

    new_data = []
    new_measurements = []

    cb_to_qb_dic = {}

    data = list(qc.data)
    while data:
        # for i in range(len(qc.data)):

        instr = data.pop(0)
        # instr = qc.data[i]
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

                # This treats the case that two measurements with the same outcome are performed
                # in this case we break the loop to make the first measurement appear as a
                # separate qubit.
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


# Wrapping function for all preproccessing operations
def circuit_preprocessor(qc):

    from qrisp.simulator import reorder_circuit

    if len(qc.data) == 0:
        return qc.copy()

    # TO-DO find reliable classifiaction when automatic disentangling works best
    if len(qc.qubits) > 45:
        qc = insert_disentangling(qc)
    qc = group_qc(qc)

    return reorder_circuit(qc, ["measure", "reset", "disentangle"])
