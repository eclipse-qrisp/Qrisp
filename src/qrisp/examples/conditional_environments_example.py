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

import time
from qrisp import (
    QuantumArray,
    QuantumBool,
    QuantumChar,
    QuantumFloat,
    QuantumVariable,
    adaptive_condition,
    cx,
    h,
    mcx,
    multi_measurement
)


def quantum_eq(input_0, input_1, test_kwargs="test"):
    return quantum_eq_inner(input_0, input_1, test_kwargs=test_kwargs)


@adaptive_condition
def quantum_eq_inner(input_0, input_1, test_kwargs="test"):
    res = QuantumBool(name="cond. bool*")

    if isinstance(input_1, QuantumVariable):
        if input_0.size != input_1.size:
            raise Exception(
                "Tried to evaluate equality condition for "
                "QuantumVariables of differing size"
            )

        cx(input_0, input_1)
        mcx(input_1, res, ctrl_state=0)
        cx(input_0, input_1)

        return res

    else:
        label_int = input_0.encoder(input_1)

        mcx(input_0, res, ctrl_state=label_int, method="gray_pt")

        return res


cond_1_qv = QuantumVariable(2)
cond_2_qv = QuantumVariable(2)
cond_3_qv = QuantumVariable(2)

cond_1_qv[:] = "11"
cond_2_qv[:] = "11"
cond_3_qv[:] = "11"

a = QuantumBool()
b = QuantumBool()
c = QuantumBool()

h(a)
cx(a, b)
h(cond_1_qv[0])
with quantum_eq(cond_1_qv, "11") as cond_qb_1:
    h(b)
    with quantum_eq(cond_2_qv, "11", test_kwargs="test_0"):
        cx(a, b)
        h(a)


print(a)
print(b)
print(c)

# %%

qc = QuantumChar()

qc[:] = "f"

qf = QuantumFloat(3, -1, signed=True)
q_array = QuantumArray(qf, 4)

h(qc[0])


q_array[0] += -0.5


with "f" == qc:
    h(q_array[0][0])

    with q_array[0] == -0.5:
        q_array[2] = -0.5
        q_array[1] -= -1

        with q_array[1] == 1:
            q_array[3].init_from(q_array[2])

start_time = time.time()
print(multi_measurement([qc, q_array]))
print(start_time - time.time())
print(qc.qs.compile().depth())
print(qc.qs.compile().cnot_count())
print(qc.qs.compile().num_qubits())
