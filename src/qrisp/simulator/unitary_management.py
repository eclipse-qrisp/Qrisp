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

# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp
from numba import njit

from qrisp import fast_append
from qrisp.simulator.bi_arrays import BiArray, DenseBiArray, SparseBiArray, tensordot


np_dtype = np.complex64

id_matrix = np.eye(2, dtype=np_dtype)

pauli_x = np.asarray([[0, 1], [1, 0]], dtype=np_dtype)
pauli_y = np.asarray([[0, 0], [0, 0]]) + 1j * np.asarray([[0, -1], [1, 0]], dtype = np_dtype)
pauli_z = np.asarray([[1, 0], [0, -1]], dtype=np_dtype)
from sympy.core.expr import Expr
import numpy
import sympy

# Function which returns the unitary of a u3 gate
def u3matrix(theta, phi, lam, global_phase, use_sympy = False):
    if not use_sympy:
        module = numpy
        I = 1j
        res = numpy.empty(shape=(2, 2), dtype=numpy.complex64)
    else:
        module = sympy
        I = sympy.I
        res = numpy.empty(shape=(2, 2), dtype=numpy.dtype("O"))
    
    
    
    # for par in [theta, phi, lam, global_phase]:
    #     if isinstance(par, Expr):
    #         res = module.zeros(shape=(2, 2), dtype="object")
    #         import sympy as module

    #         
    #         break

    res[0, 0] = module.cos(theta / 2)
    res[0, 1] = -module.exp(I * lam) * module.sin(theta / 2)
    res[1, 0] = module.exp(I * phi) * module.sin(theta / 2)
    res[1, 1] = module.exp(I * (phi + lam)) * module.cos(theta / 2)

    return res * module.exp(I * global_phase)


# Efficient function to generate the unitary of a controlled gate
def controlled_unitary(controlled_gate):
    m = controlled_gate.base_operation.num_qubits
    try:
        temp = controlled_gate.base_operation.unitary_array
    except AttributeError:
        temp = (
            controlled_gate.base_operation.unitary_array
        ) = controlled_gate.base_operation.get_unitary()

    n = controlled_gate.num_qubits
    res = np.eye(2**n, dtype=temp.dtype)

    res[-(2**m) :, -(2**m) :] = temp

    controlled_gate.unitary = temp

    return res


# Calculates the unitary of a gate embeddded into a circuit of n qubits
# For instance consider this circuit
# Here we have gate = CXGate(), n = 4, destination_qubits = [2,3]
# q80_0: ─────


# q80_1: ─────
#        ┌───┐
# q80_2: ┤ X ├
#        └─┬─┘
# q80_3: ──■──
# This function then returns the overall unitary matrix in the form of either
# a BiArray or a numpy array depending what is more efficient
def calc_embedded_unitary(gate, n, destination_qubits):
    # First we generate the array corresponding to
    # 1 (x) 1 (x) ... gate
    # where 1 denotes the identity mapping on a qubit,
    # (x) the tensor product,
    # and gate the gate unitary

    # In principle, we could use the numpy function np.kron
    # tensor_array = np.kron(np.eye(2**(n-m)), gate.get_unitary())

    # However, this performs alot of unnecessary multiplications since the structure of
    # this matrix is simply the unitary of the gate on the diagonal 2**n/2**dim(gate)
    # times repeated
    if gate.definition is not None:
        gate_array = __calc_circuit_unitary(gate.definition)
        gate.unitary = gate_array
    else:
        gate_array = gate.get_unitary()

    # Numpy arrays are only fast on a low scale than BiArrays. Therefore,  we generate
    # a Numpy Arrays
    if n < 10 and not gate_array.dtype == np.dtype("O"):
        tensor_array = generate_id_kron_jitted(gate_array, n)
    else:
        tensor_array = generate_id_kron(gate_array, n)

    # Reshape into an 2n-dimensional array where each axis represents an
    # index of the tensor factors
    tensor_array = tensor_array.reshape(n * [2, 2])

    # #Generate the nec<essary swaps to put the destination qubits at the right place
    swaps = generate_swaps(destination_qubits, n)

    # Perform swaps
    for swap in swaps:
        tensor_array = swap_tensor_factors(tensor_array, *swap)

    # Shape back into the 2**n x 2**n matrix
    tensor_array = tensor_array.reshape([1 << n, 1 << n])

    if tensor_array.dtype == np.dtype("O") and n < 10:
        tensor_array = tensor_array.to_array()

    return tensor_array


# In order to generate the swaps we follow the strategy that
# each destination qubit can be guaranteed to be puth in the correct place
# with a single swap. Therefore, we iterate through the destination qubits and
# successively put them in the right place
def generate_swaps(destination_qubits, n):
    # This list represents the current state of the tensor
    # The list element represents an identity tensor
    tensor_state = (n - len(destination_qubits)) * ["id"] + list(destination_qubits)
    swaps = []

    # Function to check if the tensor state reached its desired state
    def check_if_reached(tensor_state):
        for i in range(len(tensor_state)):
            if tensor_state[i] != "id":
                if tensor_state[i] != i:
                    return False
        return True

    # Repat until the tensor reached its final state
    while not check_if_reached(tensor_state):
        # Iterate through the tensor state and find a qubit which is not
        # in the correct position.
        for i in range(len(tensor_state)):
            if tensor_state[i] != "id":
                # If tensor_state[i] == i the qubit in question already has reached the
                # final state
                if tensor_state[i] != i:
                    # Swap the list elements
                    index = int(i)
                    value = int(tensor_state[i])
                    tensor_state[index], tensor_state[value] = (
                        tensor_state[value],
                        tensor_state[index],
                    )

                    # Log the swaps
                    swaps.append((index, value))

                    # Restart search for qubit in an incorrect position
                    break

    return list(swaps)


# Swaps the axis of a kroneker product (tensor product of linear mappings)
# ie. if we insert tensor = 1 (x) A (x) 1 (x) 1
# and apply this function with i = 0, j = 1
# we get A (x) 1 (x) 1 (x) 1
def swap_tensor_factors(tensor, i, j):
    n = len(tensor.shape) // 2

    # Note that we have to swap two axes because we are
    # dealing with the kroneker product and not the tensor product

    tensor = tensor.swapaxes(i, j)
    tensor = tensor.swapaxes(n + i, n + j)

    return tensor


# In principle we could use the numpy function np.kron
# tensor_array = np.kron(np.eye(2**(n-m)), gate.get_unitary())


# However this performs alot of unnecessary multiplications
# since the structure of this matrix is simply the unitary
# of the gate on the diagonal 2**n/2**dim(gate) times repeated
def generate_id_kron(input_tensor, n):
    if isinstance(input_tensor, np.ndarray):
        if input_tensor.dtype == np.dtype("O"):
            input_tensor = DenseBiArray(input_tensor)
        else:
            input_tensor = SparseBiArray(input_tensor)

    k = int(np.log2(input_tensor.shape[0]))

    data = np.ones(2 ** (n - k))

    nz_indices = np.arange(2 ** (n - k)).astype(np.int64)
    nz_indices = nz_indices * 2 ** (n - k) + nz_indices

    id_tensor = SparseBiArray((nz_indices, data), shape=(2 ** (n - k), 2 ** (n - k)))

    id_tensor.reshape((n - k) * [2, 2])
    input_tensor.reshape(k * [2, 2])

    res = tensordot(id_tensor, input_tensor, ((), ()))

    for i in range(n - k):
        for j in range(k):
            res.swapaxes(2 * n - 2 * k - 1 + j - i, 2 * n - 2 * k + j - i)

    res.reshape([2**n, 2**n])

    return res


@njit(cache=True)
def generate_id_kron_jitted(input_tensor, n):
    res = np.zeros((2**n, 2**n), dtype=input_tensor.dtype)

    m = int(input_tensor.shape[0])
    for i in range(2**n // m):
        res[i * m : (i + 1) * m, i * m : (i + 1) * m] = input_tensor

    return res


# This functions calculates the unitary of a given circuit

# The idea to make this more efficient is that not all matrix multiplication have to be
# perforned in the 2**n dimensional basis

# For instance in this circuit:

#      ┌───┐
# q_0: ┤ X ├──■──
#      └───┘┌─┴─┐
# q_1: ─────┤ X ├
#           └───┘
# q_2: ──────────

# q_3: ──────────

# We can calculate the unitary on the first two qubits on the 4 d basis
# and then embedd this unitary into the 2**4 = 16d basis


# In order to harness the effciency gain we use a divide and conquer strategy:
# We merge the instruction of the circuit into pairs of elementary gates, calculate
# this unitary and the again merge this unitary with the neighbouring pair of elementary
# gate. Applying this recursively requires only a single 2**n d matrix multiplication.


def __calc_circuit_unitary(qc):
    n = len(qc.qubits)

    # If the circuit is empty, return the identity matrix
    if len(qc.data) == 0:
        return np.eye(2**n, dtype = np_dtype)

    # If the circuit contains only a single insturction,
    # calculate this instructions unitary and embedd it
    if len(qc.data) == 1:
        instr_0 = qc.data[0]
        unitary_0 = calc_embedded_unitary(
            qc.data[0].op, n, [qc.qubits.index(qb) for qb in instr_0.qubits]
        )

        return unitary_0

    # If the circuit is a pair of instructions and uses all qubits,
    # we calculate their product by matrix multiplication

    # If the circuit contained unused qubits, the next "else" statement will turn
    # this circuit into it's version without any idle qubits
    # and this case is then called at a later recursion
    elif len(qc.data) == 2 and len(qc.qubits) == len(
        set(qc.data[0].qubits).union(qc.data[1].qubits)
    ):
        instr_0 = qc.data[0]
        instr_1 = qc.data[1]

        unitary_0 = calc_embedded_unitary(
            instr_0.op, n, [qc.qubits.index(qb) for qb in instr_0.qubits]
        )
        unitary_1 = calc_embedded_unitary(
            instr_1.op, n, [qc.qubits.index(qb) for qb in instr_1.qubits]
        )

        res = tensordot(unitary_1, unitary_0, (1, 0), contract_sparsity_threshold=0.01)

    # This is the case where we generate the pairs of instructions
    else:
        # Set up a new circuit where the merged instructions are appended
        recursion_qc = qc.clearcopy()

        # Merge every pair of neighbouring instructions into one instruction
        for i in range(0, len(qc.data), 2):
            if i + 1 < len(qc.data):
                recursion_qc.append(qc.data[i].merge(qc.data[i + 1]))
            else:
                recursion_qc.append(qc.data[i])
        # The recursion circuit now has about half of the instructions of the original
        # quantum circuit. Therefore, the recursion is finite
        res = __calc_circuit_unitary(recursion_qc)

    return res


# The result of __calc_circuit_unitary can either be a numpy array or a BiArray.
# This is important in order for the recursion structure to work out
# This function is a wrapper that converts to the desired output
def calc_circuit_unitary(qc, res_type="numpy"):
    with fast_append():
        res = __calc_circuit_unitary(qc)

    if res_type == "numpy":
        if isinstance(res, BiArray):
            res = res.to_array()

        return res

    elif res_type == "bi_array":
        if isinstance(res, BiArray):
            return res

        return DenseBiArray(res)

    else:
        raise Exception("Don't know result type " + res_type)
