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


def array_entry_manipulation_test():
    from qrisp import QuantumBool, QuantumArray, QuantumFloat, h, multi_measurement

    q_array = QuantumArray(QuantumBool(), (4, 4))
    index_0 = QuantumFloat(2, signed=False)
    index_1 = QuantumFloat(2, signed=False)

    index_0[:] = 2
    index_1[:] = 1

    with q_array[(index_0, index_1)] as entry:
        entry.flip()

    res_array = q_array.most_likely()

    assert res_array[index_0.most_likely(), index_1.most_likely()]
