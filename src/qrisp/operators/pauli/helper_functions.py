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

def get_integer_from_indices(indices,positions=None):
    if positions is not None:
        return sum(1 << positions[i] for i in indices)
    else:
        return sum(1 << i for i in indices)

#
# Trotterization
#

from qrisp import conjugate, rx, ry, rz, cx, h, sx, IterationEnvironment, gphase
import numpy as np

def change_of_basis(qarg, pauli_dict):
    for index, axis in pauli_dict.items():
        if axis=="X":
            h(qarg[index])
        if axis=="Y":
            sx(qarg[index])

def parity(qarg, indices):
    n = len(indices)
    for i in range(n-1):
        cx(qarg[indices[i]],qarg[indices[i+1]])

def change_of_basis_bound(pauli_dict):

    for qubit, axis in pauli_dict.items():
        if axis=="X":
            h(qubit)
        if axis=="Y":
            sx(qubit)

def parity_bound(qubits):
    n = len(qubits)
    for i in range(n-1):
        cx(qubits[i],qubits[i+1])