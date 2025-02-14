"""
\********************************************************************************
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
********************************************************************************/
"""


import numpy as np
import random
import time
from sympy import Symbol, Matrix, simplify

from qrisp.circuit import (
    PGate,
    QuantumCircuit,
)

n = 3

qc = QuantumCircuit(n + 1)

phi = Symbol("phi")
multi_cp_gate_0 = PGate(phi).control(n)


xi = Symbol("xi")
multi_cp_gate_1 = PGate(-xi).control(n)


qc.append(multi_cp_gate_0, qc.qubits)
qc.append(multi_cp_gate_1, qc.qubits)


subs_dic = {xi: 2, phi: 3}

unitary = simplify(qc.get_unitary())

unitary = Matrix(unitary)

# %%

use_qiskit = True

if use_qiskit:
    from qiskit.circuit import Parameter, QuantumCircuit
else:
    from sympy import Symbol as Parameter
    from qrisp.circuit import QuantumCircuit


n = 1000
param_name_list = ["p_" + str(i) for i in range(n)]
parameter_list = [Parameter(x) for x in param_name_list]


qc = QuantumCircuit(n + 1)

for i in range(n):
    qc.cx(i, i + 1)
    qc.p(parameter_list[i], i)


m = 100

start_time = time.time()
for i in range(int(m)):
    param_values = [
        random.randint(0, 100) / 100 * 2 * np.pi for j in range(len(parameter_list))
    ]
    qc.bind_parameters(
        {parameter_list[j]: param_values[j] for j in range(len(parameter_list))}
    )

duration = time.time() - start_time

print("Took " + str(duration / m) + " per binding iteration")
