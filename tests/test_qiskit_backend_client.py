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

# Created by ann81984 at 23.05.2022
"""
import numpy as np

from qrisp import QuantumCircuit
from qrisp.interface import VirtualBackend, QiskitBackend

def test_qiskit_backend_client():

    try:
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
    except ImportError:
        import qiskit_aer as Aer

        backend = Aer.AerSimulator()

    # Create QuantumCricuit
    qc = QuantumCircuit()

    qc.add_qubit()
    qc.add_qubit()
    

    qc.h(0)

    qc.rz(np.pi / 2, 0)

    qc.x(0)
    qc.cx(0, 1)
    

    qc.append(qc.to_op("composed_op"), qc.qubits, qc.clbits)

    qc.append(qc.to_op("multi_composed_op"), qc.qubits, qc.clbits)

    qc.add_clbit()
    qc.measure(1, 0)    
    # TO-DO prevent this test from crashing regardless of functionality
    # Create QuantumSession
    qc = QuantumCircuit()

    qc.add_qubit()
    qc.add_qubit()
    qc.add_clbit()

    qc.h(0)

    qc.rz(np.pi / 2, 0)

    qc.x(0)
    qc.cx(0, 1)
    qc.measure(1, 0)


    def sample_run_func(qc, shots = None, token = ""):
        if shots is None:
            shots = 10000
        return {"0": shots}

    test_virtual_backend = VirtualBackend(sample_run_func)

    assert str(test_virtual_backend.run(qc, 100)) == "{'0': 100}"
    assert test_virtual_backend.run(qc, 100)["0"] == 100

    ###################

    # Create Qiskit Backend
    test_qiskit_backend = QiskitBackend()

    qc = QuantumCircuit(4, 1)
    qc.x(0)
    qc.measure(0, 0)

    print(test_qiskit_backend.run(qc, 2000))
    # status = test_qiskit_backend.ping()
    assert str(test_qiskit_backend.run(qc, 2000)) == "{'1': 2000}"
"""

def test_qiskit_aer_backend():

    from qrisp import QuantumFloat
    from qrisp.interface import QiskitBackend
    from qiskit_aer import AerSimulator
    example_backend = QiskitBackend(backend = AerSimulator())
    qf = QuantumFloat(4)
    qf[:] = 3
    res = qf*qf
    meas_res = res.get_measurement(backend = example_backend)
    assert meas_res == {9: 1.0}

"""
def test_qiskit_fake_backend():

    from qrisp import QuantumFloat
    from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
    from qrisp.interface import QiskitBackend
    example_backend = QiskitBackend(FakeWashingtonV2())
    qf = QuantumFloat(2)
    qf[:] = 2
    res = qf*qf
    meas_res = res.get_measurement(backend = example_backend)
    assert meas_res[4] > 0.5
"""