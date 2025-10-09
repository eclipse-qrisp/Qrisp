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
