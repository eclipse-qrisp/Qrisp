"""
\********************************************************************************
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
********************************************************************************/
"""

from qrisp import *

@RUS(static_argnums=[1, 2]) 
def LCU(state_prep, unitaries, num_qubits):
    """
    Implements the Linear Combination of Unitaries (LCU) protocol 
    https://arxiv.org/abs/1202.5822 utilizing the RUS (Repeat-Until-Success) 
    Jasp feature.

    The terminal_sampling decorator is utilized to evaluate the LCU.

    Parameters
    ----------
    state_prep : callable
        Quantum circuit preparing the initial state for LCU. 
    unitaries : list of callables
        List of unitary operations to combine. Each unitary should be a callable
        acting on a QuantumVariable of size num_qubits.
    num_qubits : int
        Number of qubits required for the target QuantumVariable that unitaries
        act upon.

    Returns
    -------
    success_bool : QuantumBool
        QuantumBool indicating successful LCU execution (true when case indicator
        measures 0)
    qv : QuantumVariable
        The state to which we successfully applied a linear combination of unitaries
        represented as a QuantumVariable.

    """
    qv = QuantumFloat(num_qubits)
    
    # Specify the QunatumVariable that indicates which case to execute
    n = int(np.ceil(np.log2(len(unitaries))))
    case_indicator = QuantumFloat(n)
    
    # Turn into a list of qubits
    case_indicator_qubits = [case_indicator[i] for i in range(n)]
    
    # LCU protocol with conjugate preparation
    with conjugate(state_prep)(case_indicator):
        # SELECT
        for i in range(len(unitaries)):
            with control(case_indicator_qubits, ctrl_state=i):
                unitaries[i](qv)  # Apply i-th unitary from unitary list
    
    # Success condition
    success_bool = (measure(case_indicator) == 0)
    
    return success_bool, qv
