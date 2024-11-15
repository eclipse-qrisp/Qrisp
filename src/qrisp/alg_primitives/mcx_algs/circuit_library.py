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

import numpy as np

from qrisp import QuantumCircuit, CZGate, XGate, Instruction

def ctrl_state_wrap(qc, ctrl_state):
    res = qc.copy()
    for i in range(len(ctrl_state)):
        if ctrl_state[i] == "0":
            res.data.insert(0, Instruction(XGate(), [res.qubits[i]]))
            res.x(res.qubits[i])
    return res


# Toffoli Implementation according to https://arxiv.org/pdf/1206.0758.pdf
toffoli_qc = QuantumCircuit(3)
toffoli_qc.h(2)
toffoli_qc.p(-np.pi / 4, [0, 1])
toffoli_qc.cx(2, 0)
toffoli_qc.cx(1, 2)
toffoli_qc.p(np.pi / 4, 0)
toffoli_qc.cx(1, 0)
toffoli_qc.p(np.pi / 4, 2)
toffoli_qc.cx(1, 2)
toffoli_qc.p(-np.pi / 4, 0)
toffoli_qc.cx(2, 0)
toffoli_qc.p(np.pi / 4, 0)
toffoli_qc.p(-np.pi / 4, 2)
toffoli_qc.cx(1, 0)
toffoli_qc.h(2)
toffoli_qc.qubits[1], toffoli_qc.qubits[0] = toffoli_qc.qubits[0], toffoli_qc.qubits[1]


# Margolus gate as described here https://arxiv.org/abs/quant-ph/0312225 and 
# here https://www.iccs-meeting.org/archive/iccs2022/papers/133530169.pdf
margolus_qc = QuantumCircuit(3)

margolus_qc.sx(2)
margolus_qc.t(2)
margolus_qc.cx(1, 2)
margolus_qc.t(2)
margolus_qc.cx(0, 2)

reduced_margolus_qc = margolus_qc.copy()
reduced_margolus_qc.sx_dg(2)

margolus_qc.t_dg(2)
margolus_qc.cx(1, 2)
margolus_qc.t_dg(2)
margolus_qc.sx_dg(2)

# Ancilla supported multi controlled X gates from https://arxiv.org/pdf/1508.03273.pdf
# and https://www.iccs-meeting.org/archive/iccs2022/papers/133530169.pdf
maslov_qc = QuantumCircuit(4)

maslov_qc.h(3)
maslov_qc.t(3)
maslov_qc.cx(2, 3)
maslov_qc.t_dg(3)
maslov_qc.h(3)
maslov_qc.cx(0, 3)
maslov_qc.t(3)
maslov_qc.cx(1, 3)
maslov_qc.t_dg(3)
maslov_qc.cx(0, 3)

reduced_maslov_qc = maslov_qc.copy()

maslov_qc.t(3)
maslov_qc.cx(1, 3)
maslov_qc.t_dg(3)
maslov_qc.h(3)
maslov_qc.t(3)
maslov_qc.cx(2, 3)
maslov_qc.t_dg(3)
maslov_qc.h(3)


# Quasi-Toffoli implementation from https://arxiv.org/abs/1709.06648
gidney_qc = QuantumCircuit(3)

gidney_qc.h(2)
gidney_qc.t(2)
gidney_qc.cx([0,1], 2)
gidney_qc.cx(2, [0,1])
gidney_qc.t_dg([0,1])
gidney_qc.t(2)
gidney_qc.cx(2, [0,1])
gidney_qc.h([2])
gidney_qc.s([2])

gidney_qc_inv = QuantumCircuit(3, 1)

gidney_qc_inv.h(2)
gidney_qc_inv.measure(2, 0)
gidney_qc_inv.append(CZGate().c_if(1), [0,1], 0)
gidney_qc_inv.reset(2)

#Implementation of the T-depth 2 Toffoli found in https://arxiv.org/pdf/1206.0758.pdf
amy_toffoli_qc = QuantumCircuit(4)
amy_toffoli_qc.h(2)
amy_toffoli_qc.t([0,1,2])
amy_toffoli_qc.cx(1,0)
amy_toffoli_qc.cx(2,1)
amy_toffoli_qc.cx(0,2)
amy_toffoli_qc.cx(0,3)
amy_toffoli_qc.cx(1,3)
amy_toffoli_qc.t_dg([0,1,3])
amy_toffoli_qc.t([2])
amy_toffoli_qc.cx(1,3)
amy_toffoli_qc.cx(0,3)
amy_toffoli_qc.cx(0,2)
amy_toffoli_qc.cx(2,1)
amy_toffoli_qc.cx(1,0)
amy_toffoli_qc.h(2)

# Implementation of the T-Depth 1 quasi-Toffoli found in https://arxiv.org/pdf/1210.0974.pdf 
selinger_toffoli_qc = QuantumCircuit(4)
selinger_toffoli_qc.h(2)
selinger_toffoli_qc.cx(0,3)
selinger_toffoli_qc.cx(2,[0,1])
selinger_toffoli_qc.cx(1,3)
selinger_toffoli_qc.t([0,1])
selinger_toffoli_qc.t_dg([2,3])
selinger_toffoli_qc.cx(1,3)
selinger_toffoli_qc.cx(2,[0,1])
selinger_toffoli_qc.cx(0,3)
selinger_toffoli_qc.h(2)

# Implementation of the T-Depth 1 Toffoli found in https://arxiv.org/pdf/1212.5069.pdf
jones_toffoli_qc = QuantumCircuit(5,1)
jones_toffoli_qc.append(selinger_toffoli_qc.to_gate("selinger_quasi_toffoli"), [0,1,3,4])
jones_toffoli_qc.s(3)
jones_toffoli_qc.cx(3,2)
jones_toffoli_qc.h(3)
jones_toffoli_qc.measure(3,0)
jones_toffoli_qc.reset(3)
jones_toffoli_qc.append(CZGate().c_if(1), [0,1], 0)



