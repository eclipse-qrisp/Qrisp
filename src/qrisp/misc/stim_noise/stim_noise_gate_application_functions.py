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

import stim
from qrisp.core import append_operation
from qrisp.misc.stim_noise.error_class import StimError

def stim_noise(stim_name, *parameters_and_qubits, pauli_string = None):
    
    error_data = stim.gate_data("X_ERROR")
    
    if pauli_string is not None:
        # Check for compatibility
        if not (stim_name in ["E", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR"]):
             raise Exception(f"Stim error {stim_name} does not support Pauli strings. Supported gates are E, CORRELATED_ERROR, ELSE_CORRELATED_ERROR")

        num_qubits = len(pauli_string)

    elif error_data.is_single_qubit_gate:
        num_qubits = 1
    elif error_data.is_two_qubit_gate:
        num_qubits = 2
    else:
        raise Exception(f"Could not determine qubit amount for Stim error {stim_name}. Please check if the error is supported.")
    
    params = parameters_and_qubits[:-num_qubits]
    qubits = parameters_and_qubits[-num_qubits:]
    
    error_op = StimError(stim_name, *params, pauli_string=pauli_string)
    
    append_operation(error_op, qubits = qubits)
    
    
    