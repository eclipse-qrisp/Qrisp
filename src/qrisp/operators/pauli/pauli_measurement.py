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
        """
        Parameters
        ----------
            bases : list[PauliTerm]
                The basis of each group as PauliTerm.
            operators_ind : list[list[list[int]]]
                The PauliTerms in each group as list of integers. 
                The integers correspond to the positions of "Z" in the PauliTerm (after change of basis).
            operators_int : list[list[int]]
                The PauliTerms in each group as integers. 
                A "1" at position j in binary representation corresponds to a "Z" at position j in the PauliTerm (after change of basis).
            coefficients : list[list[float]]
                The coeffcients of the PauliTerms in each group.

        """
        self.bases = bases
        self.operators_ind = operators_ind
        self.operators_int = operators_int
        self.coefficients = coefficients

        self.variances, self.shots = self.measurement_shots()
        self.circuits, self.qubits = self.measurement_circuits()


    def measurement_shots(self):
        """
        Calculates the optimal distribution and number of shots following https://quantum-journal.org/papers/q-2021-01-20-385/pdf/.
        
        """
        variances = []
        for coeffs in self.coefficients:
            var = sum(x**2 for x in coeffs)
            variances.append(var)
        N = sum(np.sqrt(x) for x in variances)
        shots = [np.sqrt(x)*N for x in variances]
        return variances, shots

    def measurement_circuits(self):
        """
        Constructs the change of basis circuits for the QWC method.
        
        """

        circuits = []
        qubits = []

        # Construct change of basis circuits
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

            circuits.append(qc)    
            qubits.append(qubits_)    

        return circuits, qubits
    
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