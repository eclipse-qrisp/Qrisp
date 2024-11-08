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


def get_integer_from_indices(indices,positions=None):
    if positions is not None:
        return sum(1 << positions[i] for i in indices)
    else:
        return sum(1 << i for i in indices)


class PauliMeasurement:
    """
    
    
    """

    def __init__(self, operator):
        """

        Parameters
        ----------
        operator: QubitHamiltonian or BoundQubitHamiltonian 
        
        Attributes
        ----------
        bases : list[QubitTerm]
            The basis of each group as QubitTerm.
        operators_ind : list[list[list[int]]]
            The QubitTerms in each group as list of integers. 
            The integers correspond to the positions of "Z" in the QubitTerm (after change of basis).
        operators_int : list[list[int]]
            The QubitTerms in each group as integers. 
            A "1" at position j in binary representation corresponds to a "Z" at position j in the QubitTerm (after change of basis).
        coefficients : list[list[float]]
            The coeffcients of the QubitTerms in each group.
        circuits : list[QuantumCircuit]
            The change of basis circuits for each group.
        qubits : list[list[int or Qubit]]
            The qubits to be measured.
        variances : list[float]
            The variances for the groups.
        shots : list[float]
            The optimal distribution of shots among the groups.

        """
        self.bases, self.operators_ind, self.operators_int, self.coefficients = self.commuting_qw_measurement(operator)
        self.variances, self.shots = self.measurement_shots()
        self.circuits = self.measurement_circuits()

    def commuting_qw_measurement(self, operator):
        """
        
        """

        groups, bases = operator.commuting_qw_groups(show_bases=True)
        self.groups = groups
        operators_ind = []
        operators_int = []
        coefficients = []

        # List of dictionaries with qubits in basis as keys and their position in an ordered list as values
        positions = []
        for basis in bases:
            ordered_keys = sorted(basis.factor_dict.keys())
            position_dict = {key: index for index, key in enumerate(ordered_keys)}
            positions.append(position_dict)

        n = len(groups)
        for i in range(n):
            curr_ind = []
            curr_int = []
            curr_coeff = []

            for term,coeff in groups[i].terms_dict.items():
                ind = list(term.factor_dict.keys())

                curr_ind.append(ind)
                curr_int.append(term.serialize())
                curr_coeff.append(float(coeff.real))

            operators_ind.append(curr_ind)
            operators_int.append(curr_int)
            coefficients.append(curr_coeff)
        return bases, operators_ind, operators_int, coefficients
        

    def measurement_shots(self):
        """
        Calculates the optimal distribution and number of shots following https://quantum-journal.org/papers/q-2021-01-20-385/pdf/.
        
        """
        n = len(self.coefficients)
        variances = []
        for i in range(n):
            m = len(self.coefficients[i])
            
            var = sum(self.coefficients[i][j]**2 for j in range(m) if (self.operators_int[i][j][0]+self.operators_int[i][j][1])>0) # Exclude constant term
            variances.append(var)
        N = sum(np.sqrt(x) for x in variances)
        shots = [np.sqrt(x)*N for x in variances]
        return variances, shots

    def measurement_circuits(self):
        """
        Constructs the change of basis circuits for the QWC method.
        
        """

        circuits = []

        # Construct change of basis circuits
        for i in range(len(self.bases)):
            
            group = self.groups[i]
            n = group.find_minimal_qubit_amount()
            qc = QuantumCircuit(n)
            
            factor_dict = self.bases[i].factor_dict
            for j in range(n):
                if factor_dict.get(j, "I")=="X":
                    qc.h(j)
                if factor_dict.get(j, "I")=="Y":
                    qc.sx(j)
                    
            for term in self.groups[i].terms_dict.keys():
                
                ladder_operators = [base for base in term.factor_dict.items() if base[1] in ["A", "C"]]
                
                if len(ladder_operators):
                    anchor_factor = ladder_operators[-1]
                
                    if anchor_factor[1] == "A":
                        qc.x(anchor_factor[0])
                        
                for j in range(len(ladder_operators)-1):
                    qc.cx(anchor_factor[0], ladder_operators[j][0])

                if len(ladder_operators):
                    if anchor_factor[1] == "A":
                        qc.x(anchor_factor[0])
                
                    qc.h(anchor_factor[0])

            circuits.append(qc)    

        return circuits
    
