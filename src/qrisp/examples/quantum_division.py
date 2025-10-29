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

import time

from qrisp.alg_primitives.arithmetic import (
    QuantumFloat,
    q_div,
    q_divmod,
    qf_inversion,
)
from qrisp.misc import multi_measurement

numerator = QuantumFloat(4, -2, signed=True)
divisor = QuantumFloat(4, -2, signed=True)

N = 4 / 8
D = -5 / 4
numerator.encode(N)
divisor.encode(D)


prec = 4
quotient, remainder = q_divmod(numerator, divisor, prec=prec, adder="thapliyal")

# Q, R = list(multi_measurement([quotient, remainder], backend = qasm_simulator))[0]
Q, R = list(multi_measurement([quotient, remainder]))[0]

print("Q: ", Q)
print("delta_Q: ", abs(Q - N / D))
print("R: ", R)

# %%%

qf_0 = QuantumFloat(5, signed=False)
qf_1 = QuantumFloat(5, signed=False)

qf_0[:] = 8
qf_1[:] = 5

qf_2 = q_div(qf_0, qf_1, prec=0)
print(qf_2)

# %%
qf = QuantumFloat(5, -2, signed=False)
qf.encode(0.75)


inverted_float = qf_inversion(qf, prec=6)

start_time = time.time()
print(inverted_float)
print(0.75**-1)
print(time.time() - start_time)
