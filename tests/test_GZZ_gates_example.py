"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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

# Created by ann81984 at 05.05.2022
import pytest
import numpy as np

from qrisp import QuantumSession, QuantumVariable, QuantumCircuit
from qrisp.misc.GMS_tools import (
    gms_multi_cx_fan_out,
    gms_multi_cx_fan_in,
    gms_multi_cp_gate,
    gms_multi_cp_gate_mono_phase,
    GXX_wrapper,
    GZZ_converter,
)
from qrisp import x


def test_GZZ_gates_example():
    n = 5
    # M = np.array([[0,0,1,1,1,1,0,1,0], [0,1,0,0,1,0,1,1,1],[0,1,0,1,1,1,0,0,1],[1,0,0,0,1,1,0,0,0],[0,0,1,0,1,1,0,0,1],[1,1,1,0,1,1,1,1,1],[1,1,0,0,0,1,1,0,0],[0,1,1,0,1,1,0,0,0],[0,1,0,0,0,1,1,0,1]])

    def generate_random_inv_matrix(n, bit):
        from qrisp.misc import is_inv
        import random

        found = False

        while found == False:
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    matrix[i, j] = random.randint(0, 2**bit - 1)

            det = np.round(np.linalg.det(matrix) % 2**bit)

            found = is_inv(det, bit)

        return matrix

    M = generate_random_inv_matrix(n, 1)
    n = M.shape[0]
    row_swaps = []

    def swap_rows(M, i, j):
        M[[i, j]] = M[[j, i]]
        row_swaps.append((i, j))

    qc = QuantumCircuit(n)
    qc_gms = QuantumCircuit(n)
    K = M.copy()

    for i in range(n):
        for j in range(i, n):
            if K[j, i]:
                if i != j:
                    swap_rows(K, j, i)
                break
        else:
            raise

        fan_ins = [i]
        for j in range(n):
            if i != j and K[i, j]:
                qc.cx(j, i)
                fan_ins.append(j)
        if len(fan_ins) > 1:
            fan_ins.reverse()
            qc_gms.append(
                gms_multi_cx_fan_in(
                    len(fan_ins) - 1,
                    use_uniform=False,
                    phase_tolerant=False,
                    basis="GZZ",
                ),
                fan_ins,
            )

        temp = K[i, :].copy()
        temp[i] = 0
        for j in range(i, n):
            if K[j, i]:
                K[j, :] = (K[j, :] + temp) % 2

    assert qc.compare_unitary(qc_gms, precision=4) == True


def test_GZZ_converter():
    qc = QuantumCircuit(7)
    qc.cz(qc.qubits[0], qc.qubits[1])
    qc.cz(qc.qubits[0], qc.qubits[3])
    qc.cz(qc.qubits[5], qc.qubits[4])
    qc.p(0.5, qc.qubits[2])
    qc.cp(0.25, qc.qubits[1], qc.qubits[6])
    qc.cz(qc.qubits[3], qc.qubits[1])
    qc.cz(qc.qubits[2], qc.qubits[6])
    qc.rz(0.5, qc.qubits[4])
    qc.cz(qc.qubits[3], qc.qubits[4])

    qc_zz = GZZ_converter(qc)

    assert qc.compare_unitary(qc_zz, 4) == True
