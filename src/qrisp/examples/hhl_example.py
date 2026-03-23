"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

import numpy as np
from numpy.linalg import solve
from scipy.linalg import expm
from qrisp import QuantumVariable, HHL, control, multi_measurement, ry, x

# , QuantumCircuit
# from qiskit.extensions import UnitaryGate


# Example based on https://arxiv.org/pdf/2108.09004.pdf
def hamiltonian_ev(qv):
    matrix = [[1, -1 / 3], [-1 / 3, 1]]

    t = 3 / 4 * np.pi

    matrix = expm(1j * t * (np.array(matrix)))

    # gate = UnitaryGate(matrix)
    # qc = gate.definition
    # qrisp_qc = QuantumCircuit.from_qiskit(qc)
    # qv.qs.append(qrisp_qc.to_gate("hamiltonian_ev"), qv)

    qv.qs.unitary(matrix, qv)


def ev_inversion(ev, ancilla):
    # Rotate the ancilla qubit, performed through controlled rotation on the
    # ancilla qubit
    ry_values = [np.pi, np.pi / 3]

    for i in range(ev.size):
        with control(ev[i]):
            ry(ry_values[i], ancilla)


qv = QuantumVariable(1)
x(qv)

ancilla = HHL(qv, hamiltonian_ev, ev_inversion, precision=2)

mes_res = multi_measurement([ancilla, qv])

print("Measurement result: ", mes_res)

print("Intensity ratio: ", mes_res[(True, "1")] / mes_res[(True, "0")])


matrix = [[1, -1 / 3], [-1 / 3, 1]]

matrix = np.array(matrix)

b = [0, 1]

b = np.array(b)

solution = solve(matrix, b)
print("Classical solution intensity ratio: ", solution[1] ** 2 / solution[0] ** 2)
