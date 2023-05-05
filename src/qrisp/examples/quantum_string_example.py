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


from qrisp import QuantumChar, QuantumString

q_str = QuantumString()
q_str_2 = QuantumString()
q_ch = QuantumChar()

q_str[:] = "hello"
q_ch[:] = " "
q_str_2[:] = "world"
q_str += q_ch
q_str += q_str_2
q_str += "! "
q_str_3 = q_str.duplicate(init=True)

print(q_str + q_str_3)
