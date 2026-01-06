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
from qrisp.circuit import Operation, QuantumCircuit

class StimError(Operation):
    """
    Class for representing Stim errors in Qrisp circuits.
    
    This class is used to wrap Stim instructions into Qrisp Operations. These operations are effectively identity gates (they have an empty definition) but carry the information about the Stim noise channel. When converted to Stim circuits, these operations are replaced by the corresponding Stim instruction.
    
    Parameters
    ----------
    stim_name : str
        The name of the Stim error gate (e.g. ``DEPOLARIZE1``).
    *params : float
        The parameters of the error channel (e.g. error probability). Further 
        details about the semantics of the parameters can be found in the
        `Stims gate reference <https://github.com/quantumlib/Stim/blob/main/doc/gates.md#noise-channels>`_
    pauli_string : str, optional
        A string of Pauli operators (e.g. ``XX``) for correlated errors.
    
    Examples
    --------

    We construct a simple circuit that contains both: quantum gates and error instructions.

    ::
        
        from qrisp import QuantumCircuit
        from qrisp.misc.stim_tools import StimError
        qc = QuantumCircuit(1)
        qc.x(0)
        # Apply a depolarization error with probability 0.1
        qc.append(StimError("DEPOLARIZE1", 0.1), qc.qubits)
        print(qc)
        # Yields:
        #       ┌───┐┌──────────────────┐
        # qb_0: ┤ X ├┤ stim.DEPOLARIZE1 ├
        #       └───┘└──────────────────┘
        print(qc.to_stim())
        # Yields:
        # X 0
        # DEPOLARIZE1(0.1) 0
        
    """

    def __init__(self, stim_name, *params, pauli_string = None):
        
        self.stim_name = stim_name
        self.pauli_string = pauli_string
        
        error_data = stim.gate_data(stim_name)
        
        parameter_amounts = list(error_data.num_parens_arguments_range)
        
        if not error_data.is_noisy_gate:
            raise Exception("Non-noisy stim gates are not supported via the error interface. Please use the default Qrisp alternatives.")
        
        if len(params) not in parameter_amounts:
            raise Exception(f"Stim error of type {stim_name} can take parameter amounts {parameter_amounts} but not received {len(params)} parameters instead")
            
        if self.pauli_string is not None:
            # Check for compatibility
            if not (stim_name in ["E", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR"]):
                 raise Exception(f"Stim error {stim_name} does not support Pauli strings. Supported gates are E, CORRELATED_ERROR, ELSE_CORRELATED_ERROR")

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
                           params = params,
                           definition = QuantumCircuit(num_qubits))

