
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

def test_SQPE():
    from qrisp import QuantumVariable
    from qrisp.operators import X, Z

    H_1q = Z(0)


    def prepare_ansatz():
        qv = QuantumVariable(1)  
        X(qv[0])  
        return qv

    ansatz_state_1q = prepare_ansatz()

    # Parameters
    precision_energy_ground_state = 0.5    
    tau = 1.0                             
    energy_threshold = 0               
    energy_resolution_cdf = 0.2           
    overlap_H_ansatz = 0.9                
    total_error = 0.2                    
    failure_probability = 0.1              
    n_max = 10

    # Run 
    is_below_threshold, prob_estimate = SQPE(
        H=H_1q,
        ansatz_state=ansatz_state_1q,
        precision_energy_ground_state=precision_energy_ground_state,
        tau=tau,
        energy_threshold=energy_threshold,
        energy_resolution_cdf=energy_resolution_cdf,
        overlap_H_ansatz=overlap_H_ansatz,
        total_error=total_error,
        failure_probability=failure_probability,
        n_max=n_max
    )

    assert is_below_threshold