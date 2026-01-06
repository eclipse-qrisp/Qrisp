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
        
        self.stim_name = stim_name
        
        error_data = stim.gate_data(stim_name)
        
        parameter_amounts = list(error_data.num_parens_arguments_range)
        
        if not error_data.is_noisy_gate:
            raise Exception("Non-noisy stim gates are not supported via the error interface. Please use the default Qrisp alternatives.")
        
        if len(params) not in parameter_amounts:
            raise Exception(f"Stim error of type {stim_name} can take parameter amounts {parameter_amounts} but not received {len(params)} parameters instead")
            
        if error_data.is_single_qubit_gate:
            num_qubits = 1
        elif error_data.is_two_qubit_gate:
            num_qubits = 2
        
        name = "stim." + stim_name
        
        Operation.__init__(self, name = name, 
                           num_qubits = num_qubits,
                           params = params)

