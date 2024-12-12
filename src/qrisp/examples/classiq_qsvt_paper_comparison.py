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


# This file implements the circuits described in https://arxiv.org/abs/2412.07372
# and computes some performance metrics
from qrisp import *
coin_0 = QuantumBool()
qv = QuantumFloat(5)

def walk_operator(data, coin, inpl_adder = fourier_adder):

    inpl_adder(-1, data)
    
    with control(coin):
        inpl_adder(2, data)

walk_operator(qv, coin_0)
# This is the circuit described on page 6    
qc = qv.qs.compile()
print("Walk Operator: ")
print("CNOT count: ", qc.cnot_count())
print("Qubits: ", qc.num_qubits())
# Walk Operator: 
#CNOT count:  30
# Qubits:  6

# This is the circuit described on page 8
def controlled_reflection(data, ctrl_qb):
    
    h(ctrl_qb)
    h(data[0])
    mcx([ctrl_qb] + data[:-1], data[-1], method = "balauca", ctrl_state = 0)
    h(data[0])
    h(ctrl_qb)

qv = QuantumFloat(20)
coin_0 = QuantumBool()
coin_1 = QuantumBool()
walk_operator(qv, coin_0)
controlled_reflection(qv, coin_1)

qc = qv.qs.compile()
print("QSVT circuit:")
print("CNOT count: ", qc.cnot_count())
print("Qubits: ", qc.num_qubits())
# QSVT circuit:
# CNOT count:  536
# Qubits:  39