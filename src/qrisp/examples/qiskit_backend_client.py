"""
/********************************************************************************
* Copyright (c) 2023 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2 
* or later with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0
********************************************************************************/
"""


import numpy as np
from qrisp import QuantumCircuit, QuantumVariable
from qiskit import Aer
from qrisp.interface import VirtualQiskitBackend
from qrisp.interface.backends import VirtualBackend

# Create some QuantumCricuit
qc = QuantumCircuit(3, 1)

qc.h(0)

qc.rz(np.pi / 2, 0)

qc.x(0)
qc.cx(0, 1)
qc.measure(1, 0)

qc.append(qc.to_op("composed_op"), qc.qubits, qc.clbits)

qc.append(qc.to_op("multi_composed_op"), qc.qubits, qc.clbits)

print(qc)


# %%
# Create VirtualBackend

def sample_run_func(qc, shots, token):
    print("Executing Circuit")
    return {"0": shots}


test_virtual_backend = VirtualBackend(sample_run_func)
print(test_virtual_backend.run(qc, 100, ""))


# %%
# Create Virtual Qiskit Backend

qv = QuantumVariable(3)
qv.qs.add_clbit()

qv.qs.append(qc.to_op(), qv.reg, qv.qs.clbits)

qiskit_backend = Aer.get_backend("qasm_simulator")
vrtl_qasm_sim = VirtualQiskitBackend(qiskit_backend)

results = qv.get_measurement(backend=vrtl_qasm_sim)

print(vrtl_qasm_sim.run(qc, 2000))
