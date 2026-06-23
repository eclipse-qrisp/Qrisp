"""
********************************************************************************
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
********************************************************************************
"""

# -*- coding: utf-8 -*-

from qrisp.simulator.bi_arrays import DenseBiArray, SparseBiArray, tensordot, BiArray
from qrisp.simulator.numerics_config import float_tresh, xp

# tensordot = xp.tensordot

# Class to describe TensorFactors


class TensorFactor:

    index_init = xp.zeros(1, dtype=xp.int64)
    data_init = xp.ones(1, dtype=xp.complex64)

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

            self.tensor_array = SparseBiArray(
                (self.index_init, self.data_init), shape=(2**self.n,)
            )

        else:
            self.tensor_array = init_tensor_array

            if not isinstance(init_tensor_array, BiArray):
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

        if matrix.size >= 2**6 and matrix.dtype != np.dtype("O"):
            matrix = matrix * (np.abs(matrix) > float_tresh)

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
        if id(self.tensor_array) == id(other.tensor_array):
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

    def multi_measure(self, mes_qubits, return_res_tf=True):

        for i in range(len(mes_qubits)):
            # Swap the index that is supposed to be measured to the front
            index = self.qubits.index(mes_qubits[i])
            self.swap(i, index)

        # Split the array - the lower half corresponds to the state with outcome 0,
        # the upper half to 1
        new_bi_arrays, p_list, outcome_index_list = self.tensor_array.multi_measure(
            list(range(len(mes_qubits))), return_new_arrays=return_res_tf
        )

        p_list = xp.array(p_list) / xp.sum(p_list)

        new_qubit_list = list(self.qubits[len(mes_qubits) :])

        if return_res_tf:
            tf_list = []
            for i in range(len(p_list)):

                new_bi_arrays[i].data *= 1 / p_list[i] ** 0.5
                tf_list.append(TensorFactor(list(new_qubit_list), new_bi_arrays[i]))
        else:
            tf_list = len(p_list) * [None]

        return p_list, tf_list, outcome_index_list

    def disentangle(self, qubit, warning=False):
        if len(self.qubits) == 1:
            return self, self

        # Swap the index that is supposed to be measured to the front
        index = self.qubits.index(qubit)
        self.swap(0, index)

        # Split the array - the lower half corresponds to the state with outcome 0,
        # the upper half to 1
        new_bi_arrays, p_list, outcome_index_list = self.tensor_array.multi_measure(
            [0], return_new_arrays=True
        )

        new_qubits = list(self.qubits)
        new_qubits.remove(qubit)

        if len(outcome_index_list) == 1:
            temp = xp.zeros(2, dtype=self.tensor_array.data.dtype)
            if outcome_index_list[0] == 1:
                if warning:
                    print(
                        "\r"
                        + 85 * " "
                        + "\rWARNING: Faulty uncomputation found during simulation."
                    )
                temp[1] = 1
            else:
                temp[0] = 1

            new_bi_arrays[0].data *= 1 / p_list[0] ** 0.5
            # print("disentangling successfull")
            return TensorFactor([qubit], temp), TensorFactor(
                new_qubits, new_bi_arrays[0]
            )

        if not new_bi_arrays[0].exclude_linear_indpendence(new_bi_arrays[1]):
            if warning:
                print(
                    "\r"
                    + 85 * " "
                    + "\rWARNING: Faulty uncomputation found during simulation."
                )
            return self, self

        vdot_value = new_bi_arrays[0].vdot(new_bi_arrays[1])

        if xp.abs(xp.abs(vdot_value) - (p_list[0] * p_list[1]) ** 0.5) > 1e-7:
            if warning:
                print(
                    "\r"
                    + 85 * " "
                    + "\rWARNING: Faulty uncomputation found during simulation."
                )
            # print("disentangling failed")
            # print(vdot_value)
            # print(xp.abs(xp.abs(vdot_value) - (p_list[0]*p_list[1])**0.5))
            return self, self
        temp = xp.zeros(2, dtype=self.tensor_array.data.dtype)
        temp[0] = (p_list[0]) ** 0.5
        temp[1] = (p_list[1]) ** 0.5 * vdot_value / xp.abs(vdot_value)

        new_bi_arrays[0].data *= 1 / p_list[0] ** 0.5
        # print("disentangling successfull")
        return TensorFactor([qubit], temp), TensorFactor(new_qubits, new_bi_arrays[0])


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
