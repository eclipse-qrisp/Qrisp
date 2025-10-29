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

import numpy as np

from qrisp import QuantumCircuit, QuantumSession, QuantumVariable
from qrisp.alg_primitives.mcx_algs.circuit_library import (
    margolus_qc,
    maslov_qc,
    reduced_margolus_qc,
    reduced_maslov_qc,
)


# Function to synthesize a phase tolerant multi controlled X gate
def pt_multi_cx(n, reduced=False):
    res = QuantumCircuit(n + 1)

    if n == 1:
        res.cx(0, 1)

    # The special cases n = 2 and n = 3 are handled as described in
    # https://arxiv.org/pdf/1508.03273.pdf
    elif n == 2:
        if reduced:
            res.append(reduced_margolus_qc.to_gate("reduced margolus"), res.qubits)
        else:
            res.append(margolus_qc.to_gate("margolus"), res.qubits)

    elif n == 3:
        if reduced:
            res.append(reduced_maslov_qc.to_gate("reduced maslov"), res.qubits)
        else:
            res.append(maslov_qc.to_gate("maslov"), res.qubits)
    else:
        input_qv = QuantumVariable(n)
        output_qv = QuantumVariable(1, qs=input_qv.qs)

        from qrisp import TruthTable

        tt = TruthTable([(2 ** (n) - 1) * "0" + "1"])

        tt.q_synth(input_qv, output_qv, method="gray_pt")

        res = input_qv.qs.copy()

    return res.to_gate(f"pt{n}cx")


def gray_pt_mcx(n, ctrl_state):
    input_qv = QuantumVariable(n)
    output_qv = QuantumVariable(1, qs=input_qv.qs)

    tt_str = 2**n * ["0"]
    tt_str[int(ctrl_state[::-1], 2)] = "1"
    from qrisp import TruthTable

    tt = TruthTable([tt_str])

    tt.q_synth(input_qv, output_qv, method="gray_pt")

    res = input_qv.qs.copy()
    return res.to_gate(f"pt{n}cx")
