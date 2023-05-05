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


from qrisp import QuantumFloat, h


def test_measurement_reduction():
    qf = QuantumFloat(4)
    h(qf)
    qbls = []
    for i in range(100):
        qbls.append(qf == 2)

    assert qbls[50].get_measurement() == {False: 0.9375, True: 0.0625}
