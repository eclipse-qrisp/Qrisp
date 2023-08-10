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

# Created by ann81984 at 04.05.2022
import pytest
import time

from qrisp.circuit import (
    multi_controlled_circuit,
    RYGate,
    RZGate,
    RXGate,
    transpile,
    HGate,
    XGate,
    PGate,
)
from qrisp import QuantumSession, QuantumVariable, QuantumCircuit, transpile, x
from qiskit.circuit.library import HGate


# Check by printing circuit
def test_controlled_gates():
    qc = QuantumCircuit(3)

    simple_case = False

    if simple_case:
        qc.x(0)
    else:
        # qc.h(0)
        # qc.cx(0,1)
        qc.cp(2.4, 0, 1)

    # print(qs)

    controlled_qc = multi_controlled_circuit(qc, 1)
    print(controlled_qc)

    ###################
    # Check by measuring circuit
    qs_0 = QuantumSession()

    qv = QuantumVariable(2, qs_0)

    if simple_case:
        qs_0.x(0)
    else:
        qs_0.h(0)
        qs_0.cx(0, 1)
        qs_0.cp(2.4, 0, 1)

    operation = qs_0.to_op()

    controlled_operation = operation.control(1)

    control_qv = QuantumVariable(1)
    qv = QuantumVariable(operation.num_qubits)

    control_on = True
    if control_on:
        x(control_qv[0])

    control_qv.qs.append(controlled_operation, control_qv.reg + qv.reg)

    tmp = qv.get_measurement()
    print(tmp)

    assert 0.4 < tmp["11"] < 0.6 and 0.4 < tmp["00"] < 0.6
    assert not (0.8 < tmp["11"] < 1 and 0.7 < tmp["00"] < 0.2)
    assert isinstance(tmp, dict)
    with pytest.raises(KeyError) as excinfo:
        bool(tmp["23"] < 0.6)
    assert "23" in str(excinfo.value)

    ###################
    start_time = time.time()

    n = 5
    test_gate = XGate().control(n)
    # test_gate = XGate().control(n)

    qs = QuantumSession()

    qv_ctrl = QuantumVariable(n, qs)
    qv_target = QuantumVariable(1, qs)

    x(qv_ctrl)

    qs.append(test_gate, qs.qubits)

    # Can be sped up to super-qiskit performance by enabling use_gray code
    # in logic_synthesis/gray_synthesis.py (this costs cnot count however)
    print("Took " + str(time.time() - start_time) + " s for gate synthesis")

    print(qv_target.get_measurement())
    tmp2 = qv_target.get_measurement()
    assert isinstance(tmp2, dict)
    assert tmp2["1"] == 1.0

    qv_0 = QuantumVariable(2)
    qv_1 = QuantumVariable(1)

    qv_0[:] = "10"
    qv_0.qs.append(XGate().control(2, ctrl_state="10"), qv_0.reg + qv_1.reg)

    assert qv_1.get_measurement() == {"1": 1.0}

    qv_0 = QuantumVariable(4)
    qv_1 = QuantumVariable(1)

    qv_0[:] = "0110"
    qv_0.qs.append(
        XGate().control(2, ctrl_state="10").control(2, ctrl_state="01"),
        qv_0.reg + qv_1.reg,
    )

    assert qv_1.get_measurement() == {"1": 1.0}
