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

import random
from qrisp.operators import X, Y, Z, A, C, P0, P1
from qrisp import *
from qrisp.interface import VirtualBackend
from qrisp.simulator import run

def test_measurement_method(sample_size=100, seed=42, exhaustive = False):
    
    non_sampling_backend = VirtualBackend(lambda qasm_string, shots, token : run(QuantumCircuit.from_qasm_str(qasm_string), None, ""))    

    def testing_helper(qv, operator_combinations):
        for H in operator_combinations:
            if isinstance(H, int):
                continue
            
            print(H)
            assert abs(H.get_measurement(qv, precision=0.0005, backend =non_sampling_backend) - 
                       H.to_pauli().get_measurement(qv, precision=0.0005, backend = non_sampling_backend)) < 1E-1
            assert abs(H.get_measurement(qv, precision=0.0005, 
                       diagonalisation_method="commuting", backend = non_sampling_backend) - 
                       H.to_pauli().get_measurement(qv, precision=0.0005, 
                       diagonalisation_method="commuting", backend = non_sampling_backend)) < 1E-1

    # Set the random seed for reproducibility
    random.seed(seed)

    # Define the full list of operators
    operator_list = [lambda x: 1, X, Y, Z, A, C, P0, P1]

    # Generate all possible combinations of operators
    all_combinations = []
    
    if exhaustive:
        for op1 in operator_list:
            for op2 in operator_list:
                for op3 in operator_list:
                    for op4 in operator_list:
                        
                        H = op1(0)*op2(1)*op3(2)*op4(3)
                        
                        if H is 1:
                            continue
                        
                        all_combinations.append(H)
    else:
        for _ in range(sample_size):
            combination = [random.choice(operator_list) for _ in range(4)]  # Choose 4 operators
            H = combination[0](0) * combination[1](1) * combination[2](2) * combination[3](3)
            all_combinations.append(H)

    qv = QuantumVariable(4)

    # Perform tests with the randomly generated operator combinations
    testing_helper(qv, all_combinations)

    h(qv[0])

    testing_helper(qv, all_combinations)

    cx(qv[0], qv[1])

    testing_helper(qv, all_combinations)

    cx(qv[0], qv[2])

    testing_helper(qv, all_combinations)

    h(qv[0])
    

    # Perform test for issue #165
    qv = QuantumVariable(4)
    x(qv[0])
    x(qv[1])

    H = A(0)*C(1)*C(2)*A(3) + P1(0)*P1(2) + P1(1)*P1(3)

    assert H.get_measurement(qv,diagonalisation_method='commuting') == 0
    assert H.get_measurement(qv,diagonalisation_method='commuting_qw') == 0
    
    
    
    