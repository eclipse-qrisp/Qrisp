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
import time
from qrisp import QuantumFloat, auto_uncompute
from qrisp.grover import grovers_alg, tag_state
from qrisp.misc import multi_measurement
from qrisp.interface import VirtualQiskitBackend


# Create two quantum variables

qf1 = QuantumFloat(2, signed=True)
qf2 = QuantumFloat(2, signed=True)

# Put into list
qf_list = [qf1, qf2]


# Create test oracle (simply tags the states where qf1 = -3 and qf2 = 2)
def test_oracle(qf_list):
    tag_dic = {qf_list[0]: -3, qf_list[1]: 2}

    tag_state(tag_dic)


# Apply grovers algorithm
grovers_alg(qf_list, test_oracle)


# Perform measurement

print(multi_measurement(qf_list))

# %%

# Create quantum floats
qf1 = QuantumFloat(3, -1, signed=True)
qf2 = QuantumFloat(3, -1, signed=True)

# Put into list
qf_list = [qf1, qf2]


# Create test oracle (tags the states where the multiplication results in -0.25)
@auto_uncompute
def equation_oracle(qf_list, phase=np.pi):
    # Set aliases
    factor_0 = qf_list[0]
    factor_1 = qf_list[1]

    # Calculate ancilla value
    ancilla = factor_0 * factor_1

    # Tag state
    tag_state({ancilla: -0.25}, phase=phase)


# Execute grovers algorithm
grovers_alg(qf_list, equation_oracle, winner_state_amount=2, exact=True)


qc = qf1.qs.compile()
print("Compilation done")

print(qc.depth())
print(qc.cnot_count())
print(qc.num_qubits())

# %%
# Perform measurement

qasm_simulator = VirtualQiskitBackend()

start_time = time.time()
print(multi_measurement(qf_list))
# print(multi_measurement(qf_list, backend = qasm_simulator))
print(start_time - time.time())
