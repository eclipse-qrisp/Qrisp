"""
/*********************************************************************
* Copyright (c) 2023 the Qrisp Authors
*
* This program and the accompanying materials are made
* available under the terms of the Eclipse Public License 2.0
* which is available at https://www.eclipse.org/legal/epl-2.0/
*
* SPDX-License-Identifier: EPL-2.0
**********************************************************************/
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
