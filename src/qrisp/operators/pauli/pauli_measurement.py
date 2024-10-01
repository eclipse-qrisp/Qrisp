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

from qrisp import QuantumVariable, QuantumArray, QuantumCircuit
import numpy as np

class PauliMeasurement:

    def __init__(self, bases, operators_ind, operators_int, coefficients):
        self.bases = bases
        self.operators_ind = operators_ind
        self.operators_int = operators_int
        self.coefficients = coefficients

    # Measurement settings for 'QWC' method
    def get_measurement_circuits(self):

        measurement_circuits = []
        measurement_qubits = []

        # construct change of basis circuits
        for basis in self.bases:

            basis_ = sorted(basis.pauli_dict.items())
            qubits_ = sorted(basis.pauli_dict.keys())

            n = len(basis_)
            qc = QuantumCircuit(n)
            for i in range(n):
                if basis_[i][1]=="X":
                    qc.ry(-np.pi/2,i)
                if basis_[i][1]=="Y":
                    qc.rx(np.pi/2,i)  

            measurement_circuits.append(qc)    
            measurement_qubits.append(qubits_)    

        return measurement_circuits, measurement_qubits
    
    """
    def get_measurement_circuits_old(self, qarg):

        if isinstance(qarg, QuantumArray):
            num_qubits = sum(qv.size for qv in list(qarg.flatten()))
        else:
            num_qubits = qarg.size
        
        #pauli_dicts, measurement_ops, index_ops, measurement_coeffs, constant_term = self.qubit_wise_commutativity()
        measurement_circuits = []

        # construct change of basis circuits
        for basis in self.bases:
            qc = QuantumCircuit(num_qubits)
            for item in basis.pauli_dict.items():
                if item[0] >= num_qubits:
                    raise Exception("Insufficient number of qubits")
                if item[1]=="X":
                    qc.ry(-np.pi/2,item[0])
                if item[1]=="Y":
                    qc.rx(np.pi/2,item[0])  
            measurement_circuits.append(qc)    

        return measurement_circuits
    """