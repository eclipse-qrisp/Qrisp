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

# Created by ann81984 at 04.05.2022
import time
import random
import numpy as np

from qrisp.core import QuantumSession, QuantumVariable
from qrisp.circuit import RYGate, RXGate, transpile, HGate, ZGate, PGate, XGate, MCXGate
from sympy import Symbol, simplify


def test_abstract_parameters():
    n = 3
    ctrl_qv = QuantumVariable(n)
    target_qv = QuantumVariable(1)

    phi = Symbol("phi")
    multi_cp_gate_0 = PGate(phi).control(n)

    xi = Symbol("xi")
    multi_cp_gate_1 = PGate(-xi).control(n)

    # x(ctrl_qv)
    ctrl_qv.qs.append(multi_cp_gate_0, ctrl_qv.reg + target_qv.reg)
    ctrl_qv.qs.append(multi_cp_gate_1, ctrl_qv.reg + target_qv.reg)

    subs_dic = {xi: 2, phi: 3}
    print(target_qv.get_measurement(subs_dic=subs_dic))

    assert not str(target_qv.get_measurement(subs_dic=subs_dic)) == "{'1': 5.0}"
    assert str(target_qv.get_measurement(subs_dic=subs_dic)) == "{'0': 1.0}"

    ###################
    unitary = simplify(ctrl_qv.qs.get_unitary())

    for i in range(unitary.shape[0]):
        for j in range(unitary.shape[1]):
            if i != j:
                assert unitary[i, j] == 0

    #
    #
    # unitary = Matrix(unitary)
    # init_session()

    ###################
    use_qiskit = False

    if use_qiskit:
        from qiskit.circuit import Parameter, QuantumCircuit
    else:
        from qrisp.circuit import QuantumCircuit
        from sympy import Symbol as Parameter

    n = 1000
    param_name_list = ["p_" + str(i) for i in range(n)]
    parameter_list = [Parameter(x) for x in param_name_list]

    qc = QuantumCircuit(n + 1)

    for i in range(n):
        qc.cx(i, i + 1)
        qc.p(parameter_list[i], i)

    m = 100

    start_time = time.time()
    for i in range(int(m)):
        param_values = [
            random.randint(0, 100) / 100 * 2 * np.pi for j in range(len(parameter_list))
        ]
        qc.bind_parameters(
            {parameter_list[j]: param_values[j] for j in range(len(parameter_list))}
        )

    duration = time.time() - start_time
    # print("Took " + str(duration/m) + " per binding iteration")
