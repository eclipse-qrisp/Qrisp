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
from qrisp.circuit import Operation

class StimError(Operation):
    
    def __init__(self, stim_name, *params):
        
        # We need to handle correlated errors separately from standard errors
        # because for correlated errors, the user supplies a string like E_XYZ
        # where XYZ specifies the Pauli targets.
        
        self.pauli_string = None
        
        if stim_name.startswith("E_"):
            self.stim_name = "E"
            self.pauli_string = stim_name[2:]
            
        elif stim_name.startswith("CORRELATED_ERROR_"):
            self.stim_name = "CORRELATED_ERROR"
            self.pauli_string = stim_name[17:]
            
        elif stim_name.startswith("ELSE_CORRELATED_ERROR_"):
            self.stim_name = "ELSE_CORRELATED_ERROR"
            self.pauli_string = stim_name[22:]
        else:
            self.stim_name = stim_name
        
        error_data = stim.gate_data(self.stim_name)
        
        parameter_amounts = list(error_data.num_parens_arguments_range)
        
        if not error_data.is_noisy_gate:
            raise Exception("Non-noisy stim gates are not supported via the error interface. Please use the default Qrisp alternatives.")
        
        if len(params) not in parameter_amounts:
            raise Exception(f"Stim error of type {self.stim_name} can take parameter amounts {parameter_amounts} but not received {len(params)} parameters instead")
            
        if self.pauli_string is not None:
             num_qubits = len(self.pauli_string)
        elif error_data.is_single_qubit_gate:
            num_qubits = 1
        elif error_data.is_two_qubit_gate:
            num_qubits = 2
        else:
            raise Exception(f"Could not determine qubit amount for Stim error {stim_name}. Please check if the error is supported.")
        
        name = "stim." + stim_name
        
        Operation.__init__(self, name = name, 
                           num_qubits = num_qubits,
                           params = params)

