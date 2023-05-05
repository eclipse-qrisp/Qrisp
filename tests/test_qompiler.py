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

# Created by ann81984 at 06.05.2022
# import pytest


def test_qompiler():
    from qrisp import QuantumSession, QuantumVariable, x, cx

    qs = QuantumSession()

    qv_list = [QuantumVariable(1, qs=qs) for i in range(10)]

    for i in range(9):
        x(qv_list[i])
        x(qv_list[i])
        qv_list[i].delete()

    x(qv_list[-1])
    qc = qs.compile()
    assert len(qc.qubits) == 1
