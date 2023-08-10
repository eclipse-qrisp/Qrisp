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

from qrisp import QuantumArray, QuantumFloat, QuantumBool, multi_measurement, \
    tensordot, dot, h

# Quantum-Quantum matrix multiplication
qf = QuantumFloat(3, 0, signed=False)

test_qm_0 = QuantumArray(qf)
test_qm_1 = QuantumArray(qf)


test_qm_0[:] = [2, 3]
test_qm_1[:] = 2 * np.eye(2)
print(test_qm_0)
res = dot(test_qm_0, test_qm_1)
print(res)
print(np.dot(test_qm_0.most_likely(), test_qm_1.most_likely()))


# %%

# Quantum-Quantum tensordot

qf = QuantumFloat(3, 0, signed=False)

test_qm_0 = QuantumArray(qf, shape=(4, 4))
test_qm_0[:] = np.eye(4)

test_qm_0 = test_qm_0.reshape(2, 2, 2, 2)

test_qm_1 = QuantumArray(qf, shape=(2, 2))
test_qm_1[:] = np.eye(2)

res = tensordot(test_qm_0, test_qm_1, (-1, 0))

print(res)
print(np.tensordot(test_qm_0.most_likely(), test_qm_1.most_likely(), axes=(-1, 0)))


# %%

# Semi classical matrix multiplication

qtype = QuantumFloat(3, signed=False)

cl_array = np.round(np.random.rand(3, 4) * 5)
q_array = QuantumArray(qtype=qtype)

q_array[:] = np.round(np.random.rand(4, 2) * 5)

print(cl_array @ q_array)
print((cl_array @ q_array.most_likely()) % 2**qtype.size)


# %%
# Quantum Indexing

q_array = QuantumArray(QuantumBool(), (4, 4))
index_0 = QuantumFloat(2, signed=False)
index_1 = QuantumFloat(2, signed=False)


index_0[:] = 2
index_1[:] = 1

h(index_0[0])

with q_array[index_0, index_1] as entry:
    entry.flip()

print(multi_measurement([index_0, index_1, q_array]))
