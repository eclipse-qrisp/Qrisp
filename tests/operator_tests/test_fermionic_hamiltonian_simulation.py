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

import pytest
import time

from qrisp import *
from qrisp.operators.fermionic import a, c

def test_fermionic_hamiltonian_simulation():
    
    ##############
    H = a(0)*c(1)*c(2)*a(3)
    
    qv = QuantumVariable(4)
    temp = qv.duplicate()
    
    h(qv)
    cx(qv, temp)
    H.trotterization(qv)
    
    meas_res = multi_measurement([qv, temp])
    
    # This particular Hamiltonian acts as H|x> = 0
    # for all bitstrings x that are not "1001" or "0110".
    
    # Therefore Hamiltonian simulation on these bitstring should
    # be the identity
    
    for bs_0, bs_1 in meas_res.keys():
        if bs_0 not in ["1001", "0110"]:
            assert bs_0 == bs_1
            
    ################
    
    H = a(0)*c(1)*c(2)
    qv = QuantumVariable(3)
    
    temp = qv.duplicate()
    
    h(qv)
    cx(qv, temp)
    
    H.trotterization(qv)
    
    meas_res = multi_measurement([qv, temp])
    
    # Similar consideration as above
    for bs_0, bs_1 in meas_res.keys():
        if bs_0 not in ["100", "011"]:
            assert bs_0 == bs_1

    ############
    
    H = c(0)*a(0)*c(1)*c(2)*a(3)
    qv = QuantumVariable(4)
    
    temp = qv.duplicate()
    
    h(qv)
    cx(qv, temp)
    
    H.trotterization(qv)
    
    # For terms of the form c(0)*a(0) the hamiltonian
    # is essentially the controlled version on the 0-th qubit 
    # because c(0)*a(0) = |1><1|
    meas_res = multi_measurement([qv, temp])
    for bs_0, bs_1 in meas_res.keys():
        if bs_0[0] == "1":
            assert bs_0 == bs_1
    
    ###############
    # Test double indices            
    H = a(0)*c(0)*c(1)*c(2)*a(3)
    qv = QuantumVariable(4)
    
    temp = qv.duplicate()
    
    h(qv)
    cx(qv, temp)
    
    H.trotterization(qv)
    
    # For terms of the form a(0)*c(0) the hamiltonian
    # is essentially the controlled version on the 0-th qubit with control state 0
    # because a(0)*c(0) = |0><0|
    meas_res = multi_measurement([qv, temp])
    for bs_0, bs_1 in meas_res.keys():
        if bs_0[0] == "0":
            assert bs_0 == bs_1
            
    operators = [a, c, lambda x : 1]
    O = 1

    for op1 in operators:
        for op2 in operators:
            for op3 in operators:
                
                O += op1(0)*op2(1)*op3(2)
                
                if O is 1:
                    continue
                
                from numpy.linalg import norm
                qv = QuantumVariable(4)
                U = O.trotterization()
                U(qv, t = 0.2)
                qc = qv.qs.copy()            
                
                qc = qv.qs.copy()
                # Move the qubits to the top
                for i in range(qc.num_qubits() - len(qv)):
                    qc.qubits.insert(0, qc.qubits.pop(-1))

                # Compute the unitary
                unitary = qc.get_unitary()
                # Retrive the top left block matrix
                reduced_unitary_0 = unitary[:2**qv.size, :2**qv.size]

                
                qv = QuantumVariable(4)
                U = O.to_qubit_operator().trotterization()
                U(qv, t = 0.2)
                
                qc = qv.qs.copy()
                # Move the qubits to the top
                for i in range(qc.num_qubits() - len(qv)):
                    qc.qubits.insert(0, qc.qubits.pop(-1))

                # Compute the unitary
                unitary = qc.get_unitary()
                # Retrive the top left block matrix
                reduced_unitary_1 = unitary[:2**qv.size, :2**qv.size]
                
                assert norm(reduced_unitary_1 - reduced_unitary_0) < 1E-1
            
