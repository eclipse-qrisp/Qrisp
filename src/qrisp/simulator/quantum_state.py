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

from itertools import product

from qrisp.simulator.numerics_config import xp
from qrisp.simulator.tensor_factor import TensorFactor, multi_entangle

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
                set(sum([[self.tensor_factors[qubits[i]]] for i in range(len(qubits))], []))
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
        
        if entangled_factor.tensor_array.data.dtype == xp.dtype("O"):
            return
        
        p_list, tf_list, outcome_index_list = entangled_factor.multi_measure(qubits, False)
        
        disentangling_qubits = []
        
        for i in range(len(qubits)):
            for j in outcome_index_list:
                if j & (1<<i) == 0:
                    break
            else:
                disentangling_qubits.append(qubits[i])
                continue
            
            for j in outcome_index_list:
                if (j^(1<<i)) & (1<<i) == 0:
                    break
            else:
                disentangling_qubits.append(qubits[i])
        
        for qb in disentangling_qubits:
            self.disentangle(qb)
        

    # Method to retrieve the statevector in the form of an array of complex numbers
    def eval(self):
        [factor.unravel() for factor in self.tensor_factors]

        # print(len(list(set(self.tensor_factors))))

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

        # This code block handles several cases that mainly serve to prevent simulation
        # of QuantumStates with a probablity of 0 to be measured.
        if p_0 != 0:
            # If both measurement results have non-zero probability, we copy self
            # In case only one state has non-zero probability we can continue using self
            if p_1 != 0:
                outcome_state_0 = self.copy()
            else:
                outcome_state_0 = self

            # Update the tensor factors of the qubits that have been entangled to the
            # measured qubit
            outcome_qubits = outcome_tensor_f_0.qubits
            for j in range(len(outcome_qubits)):
                outcome_state_0.tensor_factors[outcome_qubits[j]] = outcome_tensor_f_0

            # Update the qubit that has been measured (now it's own tensor factor again)
            outcome_state_0.tensor_factors[i] = TensorFactor([i])

        else:
            outcome_state_0 = None

        # In case the probability to measure a 1 is non-zero, we can basically proceed
        # as above, but we need to set the measured qubit to the |1> state if this is
        # required
        if p_1 != 0:
            outcome_state_1 = self
            temp = outcome_tensor_f_1.qubits
            for j in range(len(temp)):
                outcome_state_1.tensor_factors[temp[j]] = outcome_tensor_f_1

            if keep_res:
                # This set the measured qubit to the |1> state (described by the
                # array [0,1] instead of [1,0] which descibes the |0> state)
                outcome_state_1.tensor_factors[i] = TensorFactor(
                    [i], xp.array([0, 1], dtype=xp.complex64)
                )
            else:
                outcome_state_1.tensor_factors[i] = TensorFactor([i])
        else:
            outcome_state_1 = None

        # log measurement outcome. Note that we can't return here, because this function
        # is called as a thread, implying it does not keep the returned result
        self.last_mes_outcome = (p_0, outcome_state_0, p_1, outcome_state_1)

        return self.last_mes_outcome
    
    def multi_measure(self, mes_qubits, return_res_states = True):
        
        involved_factors = []
        involved_factors_dic = {}
        
        for i in range(len(mes_qubits)):
            involved_factors.append(self.tensor_factors[mes_qubits[i]])
            involved_factors_dic[mes_qubits[i]] = self.tensor_factors[mes_qubits[i]]
            
            
        involved_factors = list(set(involved_factors))
        
        outcome_dic = {}
        
        outcome_dic_list = []
        
        for tf in involved_factors:
            
            tf_mes_qubits = []
            for qb in mes_qubits:
                if hash(involved_factors_dic[qb]) == hash(tf):
                    tf_mes_qubits.append(qb)
            
            p_list, tf_list, outcome_list = tf.multi_measure(tf_mes_qubits, return_res_tf = False)
            
            
            dic_list = []
            
            for i in range(len(p_list)):
                dic_list.append({"p" : p_list[i], 
                                 "tf" : tf_list[i], 
                                 "outcome" : bin_rep(outcome_list[i], len(tf_mes_qubits)),
                                 "qubits" : tf_mes_qubits})
            
            outcome_dic_list.append(dic_list)
            outcome_dic[tf] = (p_list, tf_list, outcome_list)
            
        
        outcome_states = []
        p_list = []
        outcome_list = []
        
        for dic_tuple in product(*outcome_dic_list):
            
            p = 1
            if return_res_states:
                outcome_state = self.copy()
            outcome_value = [0]*len(mes_qubits)
            
            for i in range(len(dic_tuple)):
                
                p *= dic_tuple[i]["p"]
                
                for j in range(len(dic_tuple[i]["qubits"])):
                    qb = dic_tuple[i]["qubits"][j]
                    
                    outcome_value[mes_qubits.index(qb)] = dic_tuple[i]["outcome"][j]
                
                    if return_res_states:
                        outcome_state.tensor_factors[qb] = dic_tuple[i]["tf"]
                    
            
            outcome = "".join(outcome_value)
            outcome = int(outcome, 2)
            outcome_list.append(outcome)
            
            if return_res_states:
                outcome_states.append(outcome_state)
            else:
                outcome_states.append(None)
                
            p_list.append(p)
            
        
        if not return_res_states:
            return outcome_list, p_list
            
        
        
    # def multi_measure(self, mes_qubits, return_res_states = True):
            
        
    #     tf = multi_entangle(list(set([self.tensor_factors[i] for i in mes_qubits])))
        
    #     p_list, tf_list, outcome_list = tf.multi_measure(mes_qubits, return_res_tf = return_res_states)
        
    #     outcome_states = []
        
    #     for i in range(len(p_list)):
            
    #         if p_list[i] == 0:
    #             outcome_states.append(None)
    #             continue
            
    #         if return_res_states:
    #             new_outcome_state = self.copy()
                
    #             for j in tf_list[i].qubits:
    #                 new_outcome_state.tensor_factors[j] = tf_list[i]
                    
    #             for j in mes_qubits:
    #                 new_outcome_state.tensor_factors[j] = TensorFactor([j])
                    
    #             outcome_states.append(new_outcome_state)
                
    #         else:
                    
    #             outcome_states.append(None)
        
    #     return p_list, outcome_states, outcome_list
            

    def add_qubit(self):
        self.tensor_factors.append(TensorFactor([self.n]))
        self.n += 1
    
    def disentangle(self, qb):
        
        tensor_factor = self.tensor_factors[qb]
        
        disent_tf, remain_tf = tensor_factor.disentangle(qb)
        
        for i in tensor_factor.qubits:
            if qb == i:
                self.tensor_factors[i] = disent_tf
            else:
                self.tensor_factors[i] = remain_tf
        
        
        
