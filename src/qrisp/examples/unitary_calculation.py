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

from numpy.linalg import norm
import time
from qiskit import Aer, execute
from qrisp import QuantumFloat, transpile
from qrisp.interface import convert_circuit

n = 2

x = QuantumFloat(n, signed=True)
y = QuantumFloat(n, signed=True)

s = x * y

qc = transpile(x.qs.compile())

print(len(qc.qubits))
start = time.time()
test_unitary_1 = qc.get_unitary()
end = time.time()
print("Qrisp calculation time = %s" % (end - start))

# %%
qc.qubits = qc.qubits[::-1]

qiskit_qc = convert_circuit(qc, target_api="qiskit")


backend = Aer.get_backend("unitary_simulator")

start = time.time()
job = execute(qiskit_qc, backend)
result = job.result()
test_unitary_2 = result.get_unitary(qiskit_qc).data
end = time.time()


print("Qiskit calculation time = %s" % (end - start))

# get the unitary matrix from the result object
print("Unitaries matching:", bool(norm(test_unitary_1 - test_unitary_2) < 1e-3))
