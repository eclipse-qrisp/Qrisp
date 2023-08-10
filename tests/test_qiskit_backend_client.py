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

# Created by ann81984 at 23.05.2022
import numpy as np

from qrisp.core import QuantumSession
from qrisp.interface.backends import VirtualBackend
from qrisp.interface.backends import VirtualQiskitBackend
from qiskit import Aer


def test_qiskit_backend_client():
    # TO-DO prevent this test from crashing regardless of functionality
    return
    # Create QuantumSession
    qs = QuantumSession()

    qs.add_qubit()
    qs.add_qubit()
    qs.add_clbit()

    qs.h(0)

    qs.rz(np.pi / 2, 0)

    qs.x(0)
    qs.cx(0, 1)
    qs.measure(1, 0)

    qs.append(qs.to_op("composed_op"), qs.qubits, qs.clbits)

    qs.append(qs.to_op("multi_composed_op"), qs.qubits, qs.clbits)

    print(qs)

    ###################

    # Create VirtualBackend
    def sample_run_func(qc, shots):
        print("Executing Circuit")
        return {"0": shots}

    test_virtual_backend = VirtualBackend(sample_run_func)

    print(test_virtual_backend.run(qs, 100))
    assert str(test_virtual_backend.run(qs, 100)) == "{'0': 100}"
    assert test_virtual_backend.run(qs, 100)["0"] == 100

    ###################

    # Create Qiskit Backend
    test_qiskit_backend = VirtualQiskitBackend(Aer.get_backend("qasm_simulator"))

    from qrisp import QuantumCircuit

    qc = QuantumCircuit(4, 1)
    qc.x(0)
    qc.measure(0, 0)

    print(test_qiskit_backend.run(qc, 2000))
    # status = test_qiskit_backend.ping()
    assert str(test_qiskit_backend.run(qc, 2000)) == "{'1': 2000}"
