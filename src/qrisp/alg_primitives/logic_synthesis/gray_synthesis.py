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

import numpy as np
import sympy as sp

from qrisp.misc import array_as_int, gate_wrap, int_as_array
from qrisp.circuit import QuantumCircuit, Operation, PTControlledOperation, ControlledOperation, fast_append

use_gray_code = False
try:
    from ffht import fht as fwht

    fwht(np.array([0, 1, 0, 1]))
    # print("FALCONN fwht import successfull")
except:
    # print("FALCONN fwht import failed")

    def fwht(a):
        """In-place Fast Walsh–Hadamard Transform of array a."""
        h = 1
        while h < len(a):
            for i in range(0, len(a), h * 2):
                for j in range(i, i + h):
                    x = a[j]
                    y = a[j + h]
                    a[j] = x + y
                    a[j + h] = x - y
            h *= 2


def hamming_d(i, j, bit):
    return sum((int_as_array(i, bit) + int_as_array(j, bit)) % 2)


# Solution of the Hamming TSP
# Based on having two salesmen who always visit the place which is closest to them
# and which are reunified at the end of their travel
# The solution is now the path of the first one + the reversed path of the second
def hamming_tsp(location_list, bit_amount):
    if use_gray_code or len(location_list) == 2 ** (bit_amount - 1) + 1:
        return gray_code(max(bit_amount - 1, 1))

    # print(location_list)
    if len(location_list) == 1:
        return list(location_list)

    if len(location_list) == 2:
        temp = list(location_list)
        temp.sort()
        # print([temp[1]] + hamming_movement(temp[1], temp[0], bit_amount))
        return [temp[1]] + hamming_movement(temp[1], temp[0], bit_amount)

    # Copy list in order not to change the input
    location_list = list(location_list)

    # Create log-lists for the salesman pathes
    salesman_1 = [min(location_list)]
    location_list.pop(np.argmin(location_list))

    salesman_2 = [min(location_list)]
    location_list.pop(np.argmin(location_list))

    # Keep track of some stats
    empty_steps = 0

    while len(location_list):
        # Move first salesman

        # Find closest location
        best_distance = np.inf
        for i in range(len(location_list)):
            trial_distance = hamming_d(
                salesman_1[-1], location_list[i], bit_amount
            )  # *len(location_list)
            # trial_distance += hamming_d(salesman_1[-1],
            # salesman_2[-1], n)/len(location_list)

            if trial_distance < best_distance:
                best_distance = trial_distance
                best_candidate_index = i

        # Count steps which do not visit a relevant parity operator
        empty_steps += (
            hamming_d(salesman_1[-1], location_list[best_candidate_index], bit_amount)
            - 1
        )

        # Remove target location from the location list
        target_location = location_list.pop(best_candidate_index)

        # Move salesman
        salesman_1 += hamming_movement(salesman_1[-1], target_location, bit_amount)

        # End procedure if the location list is empty now
        if len(location_list) == 0:
            break

        # Move second salesman (same steps as first salesman)

        best_distance = np.inf
        for i in range(len(location_list)):
            trial_distance = hamming_d(
                salesman_2[-1], location_list[i], bit_amount
            )  # *len(location_list)
            # trial_distance += hamming_d(salesman_1[-1],
            # location_list[i], n)#/len(location_list)

            if trial_distance < best_distance:
                best_distance = trial_distance
                best_candidate_index = i

        empty_steps += (
            hamming_d(salesman_2[-1], location_list[best_candidate_index], bit_amount)
            - 1
        )
        target_location = location_list.pop(best_candidate_index)

        salesman_2 += hamming_movement(salesman_2[-1], target_location, bit_amount)

    # Unify salesmen
    salesman_1 += hamming_movement(salesman_1[-1], salesman_2[-1], bit_amount)

    salesman_2.pop(-1)

    # Construct solution (first salesman + second salesman reversed)
    solution = salesman_1 + salesman_2[::-1]

    # print("Reunion cost:")
    # print(hamming_d(salesman_1[-1], salesman_2[-1], n))
    # empty_steps += hamming_d(salesman_1[-1], salesman_2[-1], n) - 1
    # print("Percentage of empty steps")
    # print(empty_steps/parity_amount*100)
    # print(solution)

    if 2 ** max(bit_amount - 1, 1) < len(solution):
        return gray_code(max(bit_amount - 1, 1))

    return solution


def hamming_movement(init, target, bit_amount):
    salesman = [init]
    distance = hamming_d(init, target, bit_amount)
    while distance:
        distance_array = (
            int_as_array(salesman[-1], bit_amount) + int_as_array(target, bit_amount)
        ) % 2
        movement_index = np.argmax(distance_array)
        movement_array = np.zeros(distance_array.shape)
        movement_array[movement_index] = 1

        salesman.append(
            array_as_int((int_as_array(salesman[-1], bit_amount) + movement_array) % 2)
        )
        distance = hamming_d(salesman[-1], target, bit_amount)
    # print((init, target))
    # print(salesman[1:])
    return salesman[1:]


# Solves the hamming tsp problem i.e. finds a path which visits each
# corner of the n-D unit cube exactly once and ends up in the origin again
def gray_code(n):
    # The key observation is that we can solve the problem for the n-D cube, if we solve
    # it for the (n-1)-D cube such that the path starts at bin(0) and ends in
    # bin(2**(n-2)). We then concantenate this solution in the "lower half" of the n-D
    # cube with the reversed solution in the upper half applied

    # This concantenation will again give a solution which starts at 0 and ends in
    # bin(2**((n+1)-2)). We thus have a recursion

    # Example for the 3D cube:
    # A possible solution for the 2D cube is [0,1,3,2]

    # The reversal is [2,3,1,0] which is [6,7,5,4] on the upper half
    # Concantenating gives [0,1,3,2,6,7,5,4] ie. a solution which starts in 0
    # and ends in 2**(3-1) = 4

    if n < 1:
        return []
    result_list = []

    if n == 1:
        return [1, 0]

    temp_list = gray_code(n - 1)

    result_list += [2 ** (n - 1) + k for k in temp_list[::-1]]
    result_list += temp_list

    return result_list


# This function calculates the phases to put after each parity operator.
# A given state will recieve then recieve the phase phi_i if it is true under the parity
# operator i. As each state is true on a unique combination of parity operators each
# state will recieve a unique phase. The total phase of any state is the sum of all the
# parity operators it is True on. If we now want to get a certain constellation of total
# phases. This gives us a system of linear equations, where the matrix is given by the
# parity matrix


# Target_phases is a list of phases , where the i-th entry describes the phase the
# i-state should pick up. Note that the phases should be given as numbers between 0 and
# 2 where 2 describes a phase of 2pi. This helps to reduce the amount of float errors.
def gen_phase_shifts(target_phases):
    try:
        temp = np.array(target_phases, dtype=np.longdouble) / (len(target_phases) / 2)
    except TypeError:
        temp = [x / (len(target_phases) / 2) for x in target_phases]

    # Transform
    fwht(temp)

    spectrum = temp

    return spectrum


# Solves the hamming tsp and turns it into a sequence of CNOT gates,
# all acting on the same qubit
# Returns a list of tuples indicating CNOT gates and the sequence of
# parity operators traversed
def single_qb_traversal(locations, bit_amount):
    # Solve Hamming TSP to get traversal sequence
    tr_seq = hamming_tsp(locations, bit_amount)

    # Make sure 0 appears last
    if tr_seq[-1] != 0:
        tr_seq = tr_seq + [0]

    if tr_seq[0] != 0:
        tr_seq = [0] + tr_seq

    # Turn into binaries
    tr_seq_bin = [int_as_array(x, bit_amount) for x in tr_seq]

    # Sequence of control bits.
    # The first control bit is the operation bit
    # This indicates later in the synthesis function, that
    # the unmodified parity operator has been loaded into the operation bit
    control_seq = [bit_amount - 1]

    # Now successively calculate Hamming distance vector
    # and determine the non-zero coordinate
    temp = tr_seq[0]
    for i in range(1, len(tr_seq)):
        temp = (tr_seq_bin[i - 1] + tr_seq_bin[i]) % 2
        temp = temp[::-1]

        control_seq.append(np.argmax(temp))

    return control_seq, tr_seq


# Similar to single_qb_traversal but this function also allows traversal
# of Parity operators in more than one qubit
# Returns a list of tuples indicating the applied CNOT gates
# and a list of integers indicating the sequence of traversed parity operators
def multi_qb_traversal(locations, bit_amount):
    # Copy locations list (in oder to prevent modification)

    locations = list(locations)

    # Sort list
    locations.sort()
    locations = locations[::-1]

    # Sort into a dictionary, where the keys are the operation qubits
    # and the values are list of parity operators that have to be traversed
    # on this qubit.
    single_qb_traversal_locations = {len(bin(x)) - 3: [] for x in locations}

    for x in locations:
        single_qb_traversal_locations[len(bin(x)) - 3].append(x)

    # Set up lists to be filled
    cnot_seq = []
    p_op_seq = []

    # Iterate through the qubits which are operated on
    for operation_bit in single_qb_traversal_locations.keys():
        # Convert the traversal locations into single qubit traversal routes
        # (for instance 4,5,7, is the traversal 0,1,3 on the qubit with significance 4)
        if operation_bit != 0:
            single_qb_locations = [
                x % 2 ** (operation_bit)
                for x in single_qb_traversal_locations[operation_bit]
            ]
        else:
            single_qb_locations = single_qb_traversal_locations[operation_bit]

        # Find single qubit traversal routes
        control_seq, p_op_temp = single_qb_traversal(
            [0] + single_qb_locations, operation_bit + 1
        )

        # Append to CNOT list
        cnot_seq += [(x, operation_bit) for x in control_seq]

        # Append to traversed parity operator list
        if operation_bit != 0:
            p_op_seq += [x + 2 ** (operation_bit) for x in p_op_temp]
        else:
            p_op_seq += p_op_temp

    return cnot_seq, p_op_seq


compiled_gates = {}
compiled_pt_gates = {}

import time
def gray_synth_qc(target_phases, phase_tolerant=False, flip_bit_order = False):
    start_time = time.time()
    bit_amount = int(np.log2(len(target_phases)))

    if not flip_bit_order:
        target_phases = np.array(target_phases)
        target_phases = target_phases.reshape(bit_amount*[2])
        for i in range(bit_amount//2):
            target_phases = np.swapaxes(target_phases, i, bit_amount-i-1)
        target_phases = target_phases.reshape(2**bit_amount)

    target_phases_id = str(list(target_phases))
    if not phase_tolerant:
        if target_phases_id in compiled_gates:
            temp = compiled_gates[target_phases_id]
            return temp
    else:
        if target_phases_id in compiled_pt_gates:
            temp = compiled_pt_gates[target_phases_id]
            return temp

    

    qc = QuantumCircuit(bit_amount)

    # Generate phase shifts
    phase_shifts = gen_phase_shifts(target_phases)

    # Set global phase to 0
    phase_shifts[0] = 0

    # Find out about non-zero parity operators that need to be traversed
    locations = []
    for i in range(len(phase_shifts)):
        try:
            if abs(phase_shifts[i]) < 1e-16:
                continue
        except TypeError:
            pass

        locations.append(i)

    # Generate CNOT list
    cnot_list, p_op_seq = multi_qb_traversal(locations, bit_amount)
    qb_list = qc.qubits
    with fast_append():
        # Apply CNOT list and add phases
        for i in range(len(cnot_list)):
            if phase_tolerant is True and cnot_list[i][1] != bit_amount - 1:
                # If we are in the phase tolerant mode we don't need to apply a phase here,
                # because we are only interested in the phase DIFFERENCE of the 0 and 1
                # state of the output qubit. As these phase shifts dont act on the output
                # qubit they do not change the phase difference and can therefore be ignored
                break
    
            # Set some aliases to keep the code readable
            operation_qubit = qb_list[cnot_list[i][1]]
            control_qubit = qb_list[cnot_list[i][0]]
    
            # Apply CNOT gate
            if cnot_list[i][0] != cnot_list[i][1]:
                control_qubit = qb_list[cnot_list[i][0]]
                qc.cx(control_qubit, operation_qubit)
    
            # Apply phase shifts
            if phase_shifts[p_op_seq[i]] == 0:
                continue
            qc.p(-phase_shifts[p_op_seq[i]], operation_qubit)
            phase_shifts[p_op_seq[i]] = 0

    from qrisp.core import parallelize_qc

    if len(qc.qubits) < 8:
        try:
            qc = parallelize_qc(qc)
        except TypeError:
            pass

    if not phase_tolerant and not flip_bit_order and len(qc.qubits) < 9:
        definition_reversed_bit = gray_synth_qc(target_phases, phase_tolerant, flip_bit_order = True)
        if qc.depth() > definition_reversed_bit.depth():
            definition_reversed_bit.qubits.reverse()
            qc = definition_reversed_bit    
    
    if not phase_tolerant:
        compiled_gates[target_phases_id] = qc
    else:
        compiled_pt_gates[target_phases_id] = qc

    return qc


class GraySynthGate(Operation):
    
    def __init__(self, target_phases, phase_tolerant = False):
        
        definition = gray_synth_qc(target_phases, phase_tolerant)
        
        Operation.__init__(self, num_qubits = len(definition.qubits), 
                           definition = definition, 
                           name = "gray_synth_gate")
        
        self.target_phases = target_phases
        self.phase_tolerant = phase_tolerant
        
        self.abstract_params = set()
        for i in range(len(target_phases)):
            if isinstance(target_phases[i], sp.Expr):
                self.abstract_params = self.abstract_params.union(target_phases[i].free_symbols)
        
    def control(self, num_ctrl_qubits=1, ctrl_state=-1, method=None):
        

        ctrl_state = int(ctrl_state, 2)

        target_phases_new = []

        for i in range(2**num_ctrl_qubits):
            if i != ctrl_state:
                target_phases_new += [0] * len(self.target_phases)
            else:
                target_phases_new += list(self.target_phases)

        tmp = GraySynthGate(target_phases_new, phase_tolerant="pt" in str(method))
        
        if "pt" in str(method):
            res = PTControlledOperation(self,
                                        num_ctrl_qubits=num_ctrl_qubits, 
                                        ctrl_state=ctrl_state, 
                                        method=method)
        else:
            res = ControlledOperation(self,
                                      num_ctrl_qubits=num_ctrl_qubits, 
                                      ctrl_state=ctrl_state, 
                                      method=method)
        
        res.definition = tmp.definition
            
        return res
    
    def inverse(self):
        res = GraySynthGate(self.target_phases, self.phase_tolerant)
        res.definition = res.definition.inverse()
        res.target_phases = [-x for x in self.target_phases]
        return res
        
        
# Function apply gray synthesis to quantum variable qv


# Target_phases is a list of phases , where the i-th entry describes the phase the
# i-state should pick up. Note that the phases should be given as numbers between 0 and
# 2 where 2 describes a phase of 2pi. This helps to reduce the amount of float errors.
@gate_wrap(permeability="full", is_qfree=True)
def gray_phase_synth(qv, target_phases, phase_tolerant=False):
    # qv.qs.append(gray_synth_gate(target_phases, phase_tolerant), qv.reg)
    qv.qs.append(GraySynthGate(target_phases, phase_tolerant), qv.reg)


# Similar function as the above but takes a QuantumCircuit and a qubit list as arguments
# instead
def gray_phase_synth_qb_list(qc, qb_list, target_phases, phase_tolerant=False):
    # qc.append(gray_synth_gate(target_phases, phase_tolerant), qb_list)
    qc.append(GraySynthGate(target_phases, phase_tolerant), qb_list)


# Function to use gray synthesis for logic synthesis
# The general idea is to apply gray synthesis to the input variable and the output
# variables. The output variable however is enclosed H gates which means that it will be
# |1> if we synthesized the |-> state and |0> if we synthesized the |+> state.
# We choose the phases for the states which have a 0 in the output variable to be 0
# and the phases for the states which have a 1 in the output variable according to
# wether the truth table to be synthesized requires a 1 or a 0.

# An additional perk of this function is, that it support phase tolerant synthesis
# This means that we are tolerant regarding the phase the out will have, i.e. if the
# truth table for a given state says 1 and we activate the phase_tolerant option
# we will get the state exp(i*phi)*|1> so a one but with a phase which might vary
# for different input states. This tolerance regarding the phase allows us to skip half
# of the synthesis procedure making it very cheap regarding the gate count


# input_var is the quantum variable which contains the input
# output_var is the quantum variable where the output should be synthesized
# Note that output_var doesn't need to be of size one - the next parameter
# qb_nr indicates on which qubit of output_var the synthesis should take place
# Finally tt is the truth table to be synthesized
@gate_wrap(is_qfree=True, permeability=[0])
def gray_logic_synth_single_qb(input_var, output_var, qb_nr, tt, phase_tolerant=False):
    if input_var.size != tt.bit_amount:
        raise Exception(
            "Input variable does not contain enough qubits to encode truth table"
        )

    if tt.shape[1] != 1:
        raise Exception("This function can only encode single column truth tables")

    # Generate target phases such that all the states with a 0 in the output qubit
    # get phase 0 and the states with a 1 in the input qubit get a phases of pi
    # if the truth table says 1 or a phase of 0 otherwise
    target_phases = np.array(tt.shape[0] * [0] + list(tt.n_rep[:, 0])) * np.pi
    
    bit_amount = int(np.log2(len(target_phases)))
    
    target_phases = np.array(target_phases)
    target_phases = target_phases.reshape(bit_amount*[2])
    for i in range(bit_amount//2):
        target_phases = np.swapaxes(target_phases, i, bit_amount-i-1)
    target_phases = target_phases.reshape(2**bit_amount)

    # Apply h gate to the output qubit in order to get logic states from phases
    from qrisp import h

    h(output_var[qb_nr])
    input_var.qs.append(
        # gray_synth_gate(target_phases, phase_tolerant),
        GraySynthGate(target_phases, phase_tolerant),
        (input_var.reg + [output_var[qb_nr]]),
    )
    h(output_var[qb_nr])
    return


# This function uses it's single qubit version iteratively to synthesize
# truth tables with more than one column
@gate_wrap(is_qfree=True, permeability=[0])
def gray_logic_synth(input_var, output_var, tt, phase_tolerant=False, lin_solve=False):
    if len(input_var) != tt.bit_amount:
        raise Exception(
            "Input variable does not contain enough qubits to encode truth table"
        )

    if len(output_var) != tt.shape[1]:
        raise Exception(
            "output variable does not contain enough qubits to encode truth table"
        )


    if len(output_var) == 1:
        gray_logic_synth_single_qb(
            input_var, output_var, 0, tt.sub_table(0), phase_tolerant=False
        )
        return
        
        
    input_var_dupl = input_var.duplicate()
    output_var_dupl = output_var.duplicate(qs=input_var_dupl.qs)


    residual_phases = np.zeros(tt.shape[0])

    for i in range(tt.shape[1]):
        gray_logic_synth_single_qb(
            input_var_dupl, output_var_dupl, i, tt.sub_table(i), phase_tolerant=True
        )

        temp = (tt.sub_table(i).n_rep - 1 / 2) * np.pi / 2
        temp = temp.transpose()[0]

        residual_phases += temp

    from qrisp.core import parallelize_qc

    qc = input_var_dupl.qs.transpile()

    if len(qc.qubits) < 10 and input_var.size != 1:
        qc = parallelize_qc(qc)

    input_var_dupl.qs.data = []

    if not phase_tolerant:
        gray_phase_synth(input_var_dupl, residual_phases)

    qc.data.extend(input_var_dupl.qs.data)

    input_var.qs.append(qc.to_gate(), input_var.reg + output_var.reg)



# This function uses it's single qubit version iteratively to synthesize
# truth tables with more than one column
@gate_wrap
def gray_logic_synth_qb_list(
    input_qb_list,
    output_qb,
    qs,
    tt,
    phase_tolerant=False,
    inverse=False,
    gate_name=None,
):
    from qrisp import QuantumSession, QuantumVariable

    qs_temp = QuantumSession()

    input_var = QuantumVariable(len(input_qb_list), qs_temp)
    output_var = QuantumVariable(1, qs_temp)

    gray_logic_synth(input_var, output_var, tt, phase_tolerant)

    temp_gate = qs_temp.to_gate()

    if gate_name is not None:
        temp_gate.name = gate_name

    if inverse:
        qs.append(temp_gate.inverse(), input_qb_list + [output_qb])
    else:
        qs.append(temp_gate, input_qb_list + [output_qb])
