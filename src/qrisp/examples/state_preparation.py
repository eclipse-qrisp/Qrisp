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


from qrisp import QuantumFloat

qf = QuantumFloat(4, -2, signed=True)


state_dic = {2.75: 1 / 4**0.5, -1.5: -1 / 4**0.5, 2: 1 / 4**0.5, 3: 1j / 4**0.5}


qf[:] = state_dic

debugger = qf.qs.statevector("function")

print("Amplitude of state 2.75: ", debugger({qf: 2.75}))
print("Amplitude of state -1.5: ", debugger({qf: -1.5}))
print("Amplitude of state 3: ", debugger({qf: 3}))

print(qf.qs.statevector())
