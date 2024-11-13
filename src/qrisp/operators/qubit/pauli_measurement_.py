"""
\********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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
from numba import njit

class QubitOperatorMeasurement:
    
    def __init__(self, qubit_op):
        
        hamiltonian = qubit_op.hermitize()
        hamiltonian = hamiltonian.eliminate_ladder_conjugates()
        
        self.groups = hamiltonian.commuting_qw_groups()
        
        self.n = len(self.groups)
        
        # Collect standard deviation
        self.stds = []
        for group in self.groups:
            self.stds.append(np.sqrt(group.to_pauli().hermitize().get_operator_variance()))
            
        N = sum(self.stds)
        self.shots_list = [N*s for s in self.stds]
        
        self.conjugation_circuits = []
        self.measurement_operators = []
        
        for group in self.groups:
            
            conj_circuit, meas_op = group.get_conjugation_circuit()
            
            self.conjugation_circuits.append(conj_circuit)
            self.measurement_operators.append(meas_op)
            
    
    def get_measurement(self, qc, qubit_list, precision, backend):
        
        from qrisp.misc import get_measurement_from_qc
        results = []
        meas_ops = []
        meas_coeffs = []
        
        for i in range(self.n):
            
            group = self.groups[i]
            conjugation_circuit = self.conjugation_circuits[i]
            shots = int(self.shots_list[i]/precision**2)
            
            curr = qc.copy()
            qubits = [qubit_list[i] for i in range(conjugation_circuit.num_qubits())]
            curr.append(conjugation_circuit.to_gate(), qubits)
            
            res = get_measurement_from_qc(curr.transpile(), list(qubit_list), backend, shots)
            results.append(res)
            
            temp_meas_ops = []
            temp_coeff = []
            for term, coeff in group.terms_dict.items():
                temp_meas_ops.append(term.serialize())
                temp_coeff.append(coeff)
                
            meas_coeffs.append(temp_coeff)
            meas_ops.append(temp_meas_ops)
            
        
        samples = create_padded_array([list(res.keys()) for res in results]).astype(np.int64)
        probs = create_padded_array([list(res.values()) for res in results])
        meas_ops = create_padded_array(meas_ops, use_tuples = True).astype(np.int64)
        meas_coeffs = create_padded_array(meas_coeffs)
        
        return evaluate_expectation_jitted(samples, probs, meas_ops, meas_coeffs)
        


def create_padded_array(list_of_lists, use_tuples = False):
    """
    Create a padded numpy array from a list of lists with varying lengths.
    
    Parameters:
    list_of_lists (list): A list of lists with potentially different lengths.
    
    Returns:
    numpy.ndarray: A 2D numpy array with padded rows.
    """
    # Find the maximum length of any list in the input
    max_length = max(len(lst) for lst in list_of_lists)
    
    # Create a padded list of lists
    if not use_tuples:
        padded_lists = [
            lst + [0] * (max_length - len(lst))
            for lst in list_of_lists
        ]
    else:
        padded_lists = [
            lst + [(0,0,0,0)] * (max_length - len(lst))
            for lst in list_of_lists
        ]
        
    
    # Convert to numpy array
    return np.array(padded_lists)


#
# Evaluate expectation
#


def evaluate_expectation(samples, probs, operators, coefficients):
    """
    Evaluate the expectation.
    
    """
    # print(results)
    # print(operators)
    # print(coefficients)
    # raise

    expectation = 0

    for index1,ops in enumerate(operators):
        for index2,op in enumerate(ops):
            for i in range(len(samples[index1])):
                outcome,probability = samples[index1, i], probs[index1, i]
                expectation += probability*evaluate_observable(op,outcome)*np.real(coefficients[index1][index2])
    
    return expectation


def evaluate_observable(observable: tuple, x: int):
    # This function evaluates how to compute the energy of a measurement sample x.
    # Since we are also considering ladder operators, this energy can either be
    # 0, -1 or 1. For more details check out the comments of QubitOperator.get_conjugation_circuit
    
    # The observable is given as tuple, containing for integers and a boolean.
    # To understand the meaning of these integers check QubitTerm.serialize.
    
    # Unwrap the tuple
    z_int, AND_bits, AND_ctrl_state, contains_ladder = observable

    # Compute whether the sign should be sign flipped based on the Z operators
    sign_flip_int = z_int & x
    sign_flip = 0
    while sign_flip_int:
        sign_flip += sign_flip_int & 1
        sign_flip_int >>= 1
    
    # If there is a ladder operator in the term, we need to half the energy 
    # because we want to measure (|110><110| - |111><111|)/2
    if contains_ladder:
        prefactor = 0.5
    else:
        prefactor = 1
    
    # If there are no and bits, we return the result
    if AND_bits == 0:
        return prefactor*(-1)**sign_flip

    # Otherwise we apply the AND_ctrl_state to flip the appropriate bits.
    corrected_x = (x ^ AND_ctrl_state)
    
    # If all bits are in the 0 state the AND is true.
    if corrected_x & AND_bits == 0:
        return prefactor*(-1)**sign_flip
    else:
        return 0
    

evaluate_observable_jitted = njit(cache = True)(evaluate_observable)

@njit(cache = True)
def evaluate_expectation_jitted(samples, probs, operators, coefficients):
    """
    Evaluate the expectation.
    
    """
    # print(results)
    # print(operators)
    # print(coefficients)
    # raise

    expectation = 0

    for index1,ops in enumerate(operators):
        for index2,op in enumerate(ops):
            for i in range(len(samples[index1])):
                outcome,probability = samples[index1, i], probs[index1, i]
                expectation += probability*evaluate_observable_jitted(op,outcome)*np.real(coefficients[index1][index2])
    
    return expectation

