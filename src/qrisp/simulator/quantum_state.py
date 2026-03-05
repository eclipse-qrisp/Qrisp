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

from itertools import product

from qrisp.simulator.numerics_config import xp
from qrisp.simulator.tensor_factor import TensorFactor, multi_entangle
from qrisp.simulator.bi_array_helper import permute_axes, invert_permutation
import numpy as np

from qrisp.misc.utility import bin_rep


# This class describes a quantum state
# This is achieved by another class, the TensorFactor.
# Each qubit initially has its own tensor factor. When an entangling operation is
# executed, the TensorFactors can be entangled. We then have the situation that two
# qubits are described by the same TensorFactor


# This structure allows the simulation to perform entangling operations only when they
# are really needed. If a circuit consists of two blocks which are never entangled,
# after the simulation, the QuantumState will consist of two TensorFactors.
class QuantumState:
    def __init__(self, n):
        # The index of the tensor factors refers to what qubit they are describing
        self.tensor_factors = [TensorFactor([i]) for i in range(n)]
        self.n = n

    # This functions applies a (unitary) operation onto the tensor factor
    def apply_operation(self, operation, qubits):
        # Determine the involved tensor factors and the involved qubits
        involved_factors = list(
            set(sum([[self.tensor_factors[qubits[i]]] for i in range(len(qubits))], []))
        )
        involved_qubits = list(set(sum([tf.qubits for tf in involved_factors], [])))

        # If the combined tensor factor has more than 63 qubits,
        # even the sparse representation will result in an error because numpy can
        # only process 64 bit integers. Therefore the entangling function would result
        # in an error. As a last resort, we can try to disentangle every single involved qubit
        if sum([len(tf.qubits) for tf in involved_factors]) > 63:
            for tf in involved_factors:
                for qb in tf.qubits:
                    self.disentangle(qb)

            # Determine the new involved factors
            involved_factors = list(
                set(
                    sum(
                        [[self.tensor_factors[qubits[i]]] for i in range(len(qubits))],
                        [],
                    )
                )
            )

            involved_qubits = list(set(sum([tf.qubits for tf in involved_factors], [])))

        # Entangle all the factors involved in the operation
        entangled_factor = multi_entangle(involved_factors)

        # Update the entangled factor on the involved qubits
        for i in range(len(involved_qubits)):
            self.tensor_factors[involved_qubits[i]] = entangled_factor

        # Get the unitary of the operation
        unitary = operation.get_unitary()

        # Apply the matrix
        entangled_factor.apply_matrix(unitary, qubits)

        if len(
            entangled_factor.qubits
        ) < 25 or entangled_factor.tensor_array.data.dtype == xp.dtype("O"):
            return

        p_list, tf_list, outcome_index_list = entangled_factor.multi_measure(
            qubits[::-1], False
        )

        disentangling_qubits = []

        for i in range(len(qubits))[::-1]:
            for j in outcome_index_list:
                if j & (1 << i) == 0:
                    break
            else:
                disentangling_qubits.append(qubits[i])
                continue

            for j in outcome_index_list:
                if (j ^ (1 << i)) & (1 << i) == 0:
                    break
            else:
                disentangling_qubits.append(qubits[i])

        for qb in disentangling_qubits:
            self.disentangle(qb)

    # Method to retrieve the statevector in the form of an array of complex numbers
    def eval(self):
        [factor.unravel() for factor in self.tensor_factors]

        entangled_factor = multi_entangle(list(set(self.tensor_factors)))

        entangled_factor.unravel()
        return entangled_factor

    # Copying method
    def copy(self):
        factor_list = list(set(self.tensor_factors))
        new_factor_list = [factor.copy() for factor in factor_list]

        res = QuantumState(self.n)

        for i in range(len(new_factor_list)):
            factor_qubits = new_factor_list[i].qubits
            for j in range(len(factor_qubits)):
                res.tensor_factors[factor_qubits[j]] = new_factor_list[i]

        return res

    # Method to perform a measurement on this quantum state
    def measure(self, i, keep_res=True):
        # Determine the TensorFactor to be measured
        tensor_factor = self.tensor_factors[i]

        # Perform measurement on this TensorFactor
        # p_0 is the measurement probability of measuring p_0,
        # outcome_tensor_f_0 is the TensorFactor that is left after a measurement with
        # outcome 0 has been performed, i.e. if the state descibes
        # |psi> = 2**-0.5 |0>|a> + 2**-0.5|1>|b>
        # we have p_0 = 0.5 and outcome_tensor_f_0 = |a>
        p_0, outcome_tensor_f_0, p_1, outcome_tensor_f_1 = tensor_factor.measure(i)

        if outcome_tensor_f_0 is None:
            outcome_tensor = outcome_tensor_f_1
            meas_res = True
        elif outcome_tensor_f_1 is None:
            outcome_tensor = outcome_tensor_f_0
            meas_res = False
        else:
            rnd = np.random.random(1)
            if rnd < p_0:
                outcome_tensor = outcome_tensor_f_0
                meas_res = False
            else:
                outcome_tensor = outcome_tensor_f_1
                meas_res = True

        outcome_state = self
        # Update the tensor factors of the qubits that have been entangled to the
        # measured qubit
        outcome_qubits = outcome_tensor.qubits
        for j in range(len(outcome_qubits)):
            outcome_state.tensor_factors[outcome_qubits[j]] = outcome_tensor

        # Update the qubit that has been measured (now it's own tensor factor again)
        outcome_state.tensor_factors[i] = TensorFactor([i])

        if keep_res and meas_res:
            # This sets the measured qubit to the |1> state (described by the
            # array [0,1] instead of [1,0] which descibes the |0> state)
            outcome_state.tensor_factors[i] = TensorFactor(
                [i], xp.array([0, 1], dtype=xp.complex64)
            )
        else:
            outcome_state.tensor_factors[i] = TensorFactor([i])

        return meas_res, outcome_state

    def multi_measure(self, mes_qubits, return_res_states=True):

        involved_factors = []
        involved_factors_dic = {}

        for i in range(len(mes_qubits)):
            involved_factors.append(self.tensor_factors[mes_qubits[i]])
            involved_factors_dic[mes_qubits[i]] = self.tensor_factors[mes_qubits[i]]

        involved_factors = list(set(involved_factors))

        # involved_factors.sort(key = lambda x : len(x.qubits))

        grouped_qubits = []

        prob_array = np.ones(1)
        ind_array = np.zeros(1, dtype=np.int64)

        for tf in involved_factors:

            tf_mes_qubits = []
            for qb in mes_qubits:
                if hash(involved_factors_dic[qb]) == hash(tf):
                    tf_mes_qubits.append(qb)

            tf.unravel()
            p_list, tf_list, outcome_list = tf.multi_measure(
                tf_mes_qubits, return_res_tf=False
            )

            prob_array = np.tensordot(prob_array, np.array(p_list), ((), ())).ravel()
            grouped_qubits.extend(tf_mes_qubits)

            if len(grouped_qubits) < 63:
                ind_array = tensorize_indices_jitted(
                    np.array(outcome_list, dtype=np.int64),
                    ind_array,
                    len(tf_mes_qubits),
                )
            else:
                if not isinstance(ind_array, list):
                    ind_array = [int(i) for i in ind_array]

                ind_array = tensorize_indices(
                    [int(i) for i in outcome_list], ind_array, len(tf_mes_qubits)
                )

        index_permutation = [mes_qubits.index(i) for i in grouped_qubits]
        index_permutation = invert_permutation(np.array(index_permutation))[::-1]

        if len(grouped_qubits) < 63:
            ind_array = permute_axes(ind_array, index_permutation)
        else:
            ind_array = permute_axes(ind_array, index_permutation, jit=False)

        prob_array = prob_array / np.sum(prob_array)

        return ind_array, prob_array

    def add_qubit(self):
        self.tensor_factors.append(TensorFactor([self.n]))
        self.n += 1

    def disentangle(self, qb, warning=False):

        tensor_factor = self.tensor_factors[qb]

        disent_tf, remain_tf = tensor_factor.disentangle(qb, warning=warning)

        for i in tensor_factor.qubits:
            self.tensor_factors[i] = remain_tf

        self.tensor_factors[tensor_factor.qubits[0]] = disent_tf


from numba import njit


@njit(cache=True)
def tensorize_indices_jitted(ind_array_0, ind_array_1, ind_0_size):

    n = len(ind_array_0)
    m = len(ind_array_1)

    res = np.zeros((m, n), dtype=np.int64)

    for i in range(m):
        for j in range(n):
            res[i, j] = (ind_array_1[i] << ind_0_size) | ind_array_0[j]

    return res.ravel()


def tensorize_indices(ind_array_0, ind_array_1, ind_0_size):

    n = len(ind_array_0)
    m = len(ind_array_1)

    res = []

    for i in range(m):
        for j in range(n):
            res.append((ind_array_1[i] << ind_0_size) | ind_array_0[j])

    return res
