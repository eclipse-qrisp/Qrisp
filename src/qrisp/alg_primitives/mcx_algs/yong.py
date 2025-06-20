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

from qrisp.core import mcx, h, s, s_dg, x
from qrisp.qtypes import QuantumBool
from qrisp.environments import invert


# Algorithm based on https://link.springer.com/article/10.1007/s10773-017-3389-4
def yong_mcx(input_qubits, target, ancilla=None, ctrl_state=None):
    if isinstance(target, list):
        if len(target) != 1:
            raise Exception("Tried to execute yong algorithm with multiple targets")
        target = target[0]

    if ctrl_state is None:
        ctrl_state = len(input_qubits) * "1"

    from qrisp.alg_primitives.mcx_algs import gray_pt_mcx

    if len(input_qubits) == 2:
        if ancilla is None:
            mcx(input_qubits, target, method="gray", ctrl_state=ctrl_state)
        else:
            target.qs().append(
                gray_pt_mcx(2, ctrl_state=ctrl_state), input_qubits + [target]
            )
            # mcx(input_qubits, target, method = "gray_pt", ctrl_state = ctrl_state)
        return

    for i in range(len(input_qubits)):
        if ctrl_state[i] == "0":
            x(input_qubits[i])

    ancilla_allocated = False

    if ancilla is None:
        ancilla_allocated = True
        ancilla_bl = QuantumBool(name="yong_anc*")
        ancilla = ancilla_bl[0]

    n = len(input_qubits)

    partition_k_1 = input_qubits[n // 2 :]
    partition_k_2 = input_qubits[: n // 2]

    h(target)

    yong_mcx(input_qubits=partition_k_1, target=ancilla, ancilla=partition_k_2[-1])

    s(ancilla)

    yong_mcx(
        input_qubits=partition_k_2 + [target], target=ancilla, ancilla=partition_k_1[-1]
    )

    s_dg(ancilla)

    with invert():
        s(ancilla)

        yong_mcx(
            input_qubits=partition_k_2 + [target],
            target=ancilla,
            ancilla=partition_k_1[0],
        )

        s_dg(ancilla)

        yong_mcx(input_qubits=partition_k_1, target=ancilla, ancilla=partition_k_2[0])

    h(target)

    if ancilla_allocated:
        ancilla_bl.delete()

    for i in range(len(input_qubits)):
        if ctrl_state[i] == "0":
            x(input_qubits[i])
