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

import threading

import numpy as np

from qrisp.simulator.quantum_state import QuantumState

thread_limit = 1


# This class describes a set of decoherent quantum states
class ImpureQuantumState:
    # Initially we start with a single quantum state
    def __init__(self, n, clbit_amount=0):
        self.states = [QuantumState(n)]
        self.cl_prob = [1]
        self.outcome_list = [np.zeros(clbit_amount, dtype=np.byte)]
        self.collected_measurements = []

    def apply_operation(self, operation, qubits):
        # Perform operation application in parallel. This is possible because many
        # functions in here release the GIL (such as numpy's np.dot)
        if thread_limit > 1 and len(self.states) > 1:
            threads = []

            for i in range(len(self.states)):
                mp_wrapper = lambda: self.states[i].apply_operation(operation, qubits)

                thr = threading.Thread(target=mp_wrapper)

                if len(threads) >= thread_limit:
                    threads[0].join()
                    threads.pop(0)

                thr.start()
                threads.append(thr)

            [thr.join() for thr in threads]

        else:
            for i in range(len(self.states)):
                self.states[i].apply_operation(operation, qubits)

    def conditionally_apply_operation(self, operation, qubits, clbit):
        # Perform operation application in parallel. This is possible because many
        # functions in here release the GIL (such as numpy's np.dot)

        if thread_limit > 1:
            threads = []

            for i in range(len(self.states)):
                if self.outcome_list[clbit] == "0":
                    continue

                mp_wrapper = lambda: self.states[i].apply_operation(operation, qubits)

                if len(threads) >= thread_limit:
                    threads[0].join()
                    threads.pop(0)

                thr = threading.Thread(target=mp_wrapper)
                thr.start()

                threads.append(thr)

            [thr.join() for thr in threads]

        else:
            for i in range(len(self.states)):
                if self.outcome_list[clbit] == "0":
                    continue

                self.states[i].apply_operation(operation, qubits)

    # Perform reset operation
    # We perform the reset operation by measuring but not tracking the measurement
    # result.
    def reset(self, qubit_index, keep_outcome=False):
        
        if thread_limit > 1:
            threads = []

            for i in range(len(self.states)):
                mp_wrapper = lambda: self.states[i].measure(qubit_index, keep_outcome)
                thr = threading.Thread(target=mp_wrapper)

                if len(threads) > thread_limit:
                    threads[0].join()
                    threads.pop(0)

                thr.start()
                threads.append(thr)

        else:
            for i in range(len(self.states)):
                self.states[i].measure(qubit_index, keep_outcome)

        # Generate new list of states and the corresponding classical propabilities +
        # the outcome bitstrings
        new_states = []
        new_cl_prob = []
        new_outcome_list = []

        for i in range(len(self.states)):
            # Wait for measurement threads to finish
            if thread_limit > 1:
                if len(threads):
                    threads[0].join()
                    threads.pop(0)

            # Get measurement outcome
            # p_0 represents the probability to measure a 0,
            # state_0 represents the state after measuring a 0
            p_0, state_0, p_1, state_1 = self.states[i].last_mes_outcome
            # Log results
            if p_0 != 0:
                # Add the new decoherent state to the list of states
                new_states.append(state_0)

                # Update its classical probability
                new_cl_prob.append(p_0 * self.cl_prob[i])

                # The bitstring stays the same
                new_outcome = self.outcome_list[i].copy()
                new_outcome_list.append(new_outcome)

            # Perform the same thing for the outcome 1
            if p_1 != 0:
                new_states.append(state_1)
                new_cl_prob.append(p_1 * self.cl_prob[i])
                new_outcome = self.outcome_list[i].copy()
                new_outcome_list.append(new_outcome)

        # Update attributes
        self.states = new_states
        self.cl_prob = new_cl_prob
        self.outcome_list = new_outcome_list

    # This functions performs measurements
    # The logic behind it is rather similar to reset, but this time we log the
    # measurement outcome.
    def measure(self, qubit_index, clbit_index):
        
        # self.multi_measure([qubit_index], [clbit_index])
        # return
        
        if thread_limit > 1:
            threads = []

            for i in range(len(self.states)):
                mp_wrapper = lambda: self.states[i].measure(qubit_index)
                thr = threading.Thread(target=mp_wrapper)

                if len(threads) > thread_limit:
                    threads[0].join()
                    threads.pop(0)

                thr.start()
                threads.append(thr)

        else:
            for i in range(len(self.states)):
                self.states[i].measure(qubit_index, True)

        new_states = []
        new_cl_prob = []
        new_outcome_list = []

        for i in range(len(self.states)):
            # Wait for measurement threads to finish
            if thread_limit > 1:
                if len(threads):
                    threads[0].join()
                    threads.pop(0)

            p_0, state_0, p_1, state_1 = self.states[i].last_mes_outcome

            if p_0 != 0:
                new_states.append(state_0)
                new_cl_prob.append(p_0 * self.cl_prob[i])
                new_outcome = self.outcome_list[i].copy()

                # Here is the point, where we log the measurement outcome
                new_outcome[clbit_index] = 0
                new_outcome_list.append(new_outcome)
            if p_1 != 0:
                new_states.append(state_1)
                new_cl_prob.append(p_1 * self.cl_prob[i])
                new_outcome = self.outcome_list[i].copy()
                new_outcome[clbit_index] = 1
                new_outcome_list.append(new_outcome)

        self.states = new_states
        self.cl_prob = new_cl_prob
        self.outcome_list = new_outcome_list
        
        
    # This functions performs measurements
    # The logic behind it is rather similar to reset, but this time we log the
    # measurement outcome.
    def multi_measure(self, qubit_indices, clbit_indices, gen_new_states = True):
        
        from qrisp.misc import int_as_array
        
        new_states = []
        new_cl_prob = []
        new_outcome_list = []

        for i in range(len(self.states)):
            # Wait for measurement threads to finish

            p_list, mes_states, outcome_list = self.states[i].multi_measure(qubit_indices, return_res_states = gen_new_states)
            
            for j in range(len(p_list)):
                
                if p_list[j] == 0:
                    continue
                
                new_states.append(mes_states[j])
                
                new_outcome = self.outcome_list[i].copy()
                new_outcome[clbit_indices] = int_as_array(outcome_list[j], len(qubit_indices))
                new_outcome_list.append(new_outcome)
                
                new_cl_prob.append(p_list[j] * self.cl_prob[i])
                
        self.states = new_states
        self.cl_prob = new_cl_prob
        self.outcome_list = new_outcome_list

    def add_qubit(self):
        for state in self.states:
            state.add_qubit()

    def add_clbit(self):
        for i in range(len(self.outcome_list)):
            self.outcome_list[i] = np.array(
                [0] + list(self.outcome_list[i]), dtype=np.byte
            )
