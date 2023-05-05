"""
/*********************************************************************
* Copyright (c) 2023 the Qrisp Authors
*
* This program and the accompanying materials are made
* available under the terms of the Eclipse Public License 2.0
* which is available at https://www.eclipse.org/legal/epl-2.0/
*
* SPDX-License-Identifier: EPL-2.0
**********************************************************************/
"""

# -*- coding: utf-8 -*-

from qrisp.simulator.bi_arrays import DenseBiArray, SparseBiArray, tensordot
from qrisp.simulator.numerics_config import float_tresh, xp

# tensordot = xp.tensordot

# Class to describe TensorFactors


class TensorFactor:
    def __init__(self, qubits, init_tensor_array=None):
        # This list contains a list of integers, which describe the current permutation
        # of qubit indices inside this TensorFactor, i.e. swapping two elements of this
        # list is the same as applying a swap gate. Using this structure is advantageous
        # because applying matrices on specific qubit indices requires alot of swapping.
        # Swapping the indices of an array is ressource costly because the array needs
        # to be copied. As it turns out memory speed is a bottleneck.
        self.qubits = qubits

        self.n = len(qubits)

        if init_tensor_array is None:
            self.tensor_array = xp.zeros(2**self.n, dtype=xp.complex64)
            self.tensor_array[0] = 1

        else:
            self.tensor_array = init_tensor_array

        if isinstance(self.tensor_array, xp.ndarray):
            self.tensor_array = SparseBiArray(self.tensor_array)

    def copy(self):
        return TensorFactor(list(self.qubits), self.tensor_array.copy())

    # Swaps two qubit indices
    def swap(self, i, j):
        if i == j:
            return

        self.qubits[i], self.qubits[j] = self.qubits[j], self.qubits[i]

        self.tensor_array = self.tensor_array.reshape(self.n * [2])
        self.tensor_array = self.tensor_array.swapaxes(i, j)
        self.tensor_array = self.tensor_array.reshape(2**self.n)

    # Applies a matrix onto the given state
    def apply_matrix(self, matrix, qubits):
        # Remove the non-zero entries caused by floating point errors to get
        # the most out of the sparse matrix representation
        import numpy as np

        if matrix.size >= 2**6:
            if matrix.dtype == np.complex64:
                matrix = matrix * (np.abs(matrix) > 1e-7)
            if matrix.dtype == np.complex128:
                matrix = matrix * (np.abs(matrix) > 1e-15)

        # Convert matrix to BiArray
        matrix = DenseBiArray(matrix)

        # Swap the qubits which the matrix is supposed to be applied on to the front
        for i in range(len(qubits)):
            qubit_index = self.qubits.index(qubits[i])
            self.swap(qubit_index, i)

        # Reshape tensor array such that the qubits form a single index
        reshaped_tensor_array = self.tensor_array.reshape(
            (matrix.shape[0], 2**self.n // matrix.shape[0])
        )

        # Apply matrix
        self.tensor_array = tensordot(
            matrix, reshaped_tensor_array, axes=(1, 0), contract_sparsity_threshold=0.05
        ).reshape(2**self.n)
        # print(type(self.tensor_array))

    # Entangles two TensorFactors.
    def entangle(self, other):
        if self == other:
            return self
        if set(self.qubits).intersection(other.qubits):
            raise Exception(
                "Tried to entangle two tensor factors with overlapping qubits"
            )

        # The tensordot function with axes = ((),()) performs a,b -> a (x) b, where
        # (x) is the tensor product sign
        init_array = tensordot(
            other.tensor_array,
            self.tensor_array,
            axes=((), ()),
            contract_sparsity_threshold=0.01,
        ).ravel()

        return TensorFactor(other.qubits + self.qubits, init_array)

    # Perform measurement
    def measure(self, i):
        # Swap the index that is supposed to be measured to the front
        index = self.qubits.index(i)
        self.swap(0, index)

        # Split the array - the lower half corresponds to the state with outcome 0,
        # the upper half to 1
        lower_half, upper_half = self.tensor_array.split()

        # Determine probabilities
        p_0 = lower_half.squared_norm()
        p_1 = upper_half.squared_norm()

        # p_1 = abs(p_0 - 1) #This doesn't work due to floating point errors
        # accumulating in the normalization

        # Prepare the qubit list of the resulting TensorFactor
        # After the measurement, the measured qubit is disentangled from the rest,
        # so we can put it into it's own TensorFactor object
        new_qubit_list = list(self.qubits)
        new_qubit_list.pop(0)

        # This treats the case that both measurement probabilities are non-zero
        if p_0 > float_tresh and p_1 > float_tresh:
            # Normalize the new statevector arrays
            normalization = 1 / (p_0**0.5)
            lower_half.data *= normalization

            # Create tensor factor corresponding to outcome 0
            tensor_factor_0 = TensorFactor(new_qubit_list, lower_half)

            # Perform the same operation for outcome 1
            normalization = 1 / (p_1**0.5)
            upper_half.data *= normalization

            tensor_factor_1 = TensorFactor(list(new_qubit_list), upper_half)

        # This treats the cases that one of the outcomes has 0 probability
        elif p_0 < float_tresh:
            p_0 = 0
            p_1 = 1
            tensor_factor_0 = None
            tensor_factor_1 = TensorFactor(new_qubit_list, upper_half)

        else:
            p_0 = 1
            p_1 = 0
            tensor_factor_0 = TensorFactor(new_qubit_list, lower_half)
            tensor_factor_1 = None

        return p_0, tensor_factor_0, p_1, tensor_factor_1

    def unravel(self):
        sorted_qubit_list = list(self.qubits)
        sorted_qubit_list.sort()
        for i in range(len(self.qubits)):
            if sorted_qubit_list[i] == self.qubits[i]:
                continue
            self.swap(i, self.qubits.index(sorted_qubit_list[i]))


# This function allow entangling more than two TensorFactors.
# Since we are free to choose a contraction order, we pick one
# that synergizes with the parallelization features of the BiArrays,
# i.e. we entangle two factors, and put the entangled result at
# the back of a queue. The processing of the entangling is hopefully
# done once this factor is at the front of the queue.
def multi_entangle(input_factor_list):
    # If there is only one factor, there is nothing to be done
    if len(input_factor_list) == 1:
        return input_factor_list[0]

    else:
        # In the case of multiple factors, entangle the first two, and put the result
        # at the back

        input_factor_list.sort(key=lambda x: len(x.qubits))
        input_factor_list.append(input_factor_list[0].entangle(input_factor_list[1]))

        # Remove entangling factors from the queue
        input_factor_list.pop(0)
        input_factor_list.pop(0)

        # Recurse on remaining factors
        return multi_entangle(input_factor_list)
