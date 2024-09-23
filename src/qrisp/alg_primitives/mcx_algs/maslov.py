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

from qrisp.circuit import QuantumCircuit
from qrisp.alg_primitives.mcx_algs.circuit_library import maslov_qc, toffoli_qc, margolus_qc
from qrisp.misc.utility import bin_rep

# Ancilla supported multi controlled X gates from https://arxiv.org/pdf/1508.03273.pdf
def maslov_mcx(n, ctrl_state=-1):
    if not isinstance(ctrl_state, str):
        if ctrl_state == -1:
            ctrl_state += 2 ** n
        ctrl_state = bin_rep(ctrl_state, n)

    res = QuantumCircuit(n + 1)
    for i in range(len(ctrl_state)):
        if ctrl_state[i] == "0":
            res.x(i)

    if n == 1:
        res.cx(0, 1)
    elif n == 2:
        res.append(toffoli_qc.to_gate("toffoli"), res.qubits)
    elif n == 3:
        res.add_qubit()
        res.append(margolus_qc.to_gate(), [0, 1, 3])
        res.append(toffoli_qc.to_gate(), [2, 3, 4])
        res.append(margolus_qc.inverse().to_gate(), [0, 1, 3])
    elif n == 4:
        res.add_qubit()
        res.append(maslov_qc.to_gate(), [0, 1, 2, 4])
        res.append(toffoli_qc.to_gate(), [3, 4, 5])
        res.append(maslov_qc.inverse().to_gate(), [0, 1, 2, 4])
    else:
        raise Exception('Multi CX for method "Maslov" only defined for n <= 4')

    for i in range(len(ctrl_state)):
        if ctrl_state[i] == "0":
            res.x(i)

    return res.to_gate(f"maslov {n}cx")