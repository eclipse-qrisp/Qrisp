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

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2

from qrisp import QuantumCircuit, QuantumFloat
from qrisp.interface import QiskitBackend, VirtualBackend

aer_simulator_backend = AerSimulator()
fake_backend = FakeWashingtonV2()


class TestQiskitBackend:
    """Test QiskitBackend functionality."""

    def test_backend_metadata(self):
        """Test that QiskitBackend correctly retrieves metadata from the Qiskit backend."""

        backend = QiskitBackend(backend=aer_simulator_backend)

        assert backend.name == aer_simulator_backend.name

        for key in aer_simulator_backend.options:
            assert backend.options[key] == aer_simulator_backend.options[key]

    def test_update_options(self):
        """Test updating options on QiskitBackend."""

        backend = QiskitBackend(backend=aer_simulator_backend)
        opts = {"shots": 12345}
        backend.update_options(**opts)

        assert backend.options["shots"] == 12345

        for key in aer_simulator_backend.options:
            if key != "shots":
                assert backend.options[key] == aer_simulator_backend.options[key]

    def test_qiskit_aer_backend(self):
        """Test QiskitBackend with Qiskit Aer simulator."""

        backend = QiskitBackend(backend=aer_simulator_backend)
        qf = QuantumFloat(4)
        qf[:] = 3
        res = qf * qf
        meas_res = res.get_measurement(backend=backend)
        assert meas_res == {9: 1.0}

    def test_qiskit_fake_backend(self):
        """Test QiskitBackend with Qiskit FakeWashingtonV2 backend."""

        backend = QiskitBackend(backend=fake_backend)
        qf = QuantumFloat(2)
        qf[:] = 2
        res = qf * qf
        meas_res = res.get_measurement(backend=backend)
        assert meas_res[4] > 0.5


# We keep this test even though VirtualBackend is deprecated
def test_qiskit_virtual_backend():
    """Test QiskitBackend client functionality with VirtualBackend."""

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

    def sample_run_func(qc, shots=None, token=""):
        if shots is None:
            shots = 10000
        return {"0": shots}

    test_virtual_backend = VirtualBackend(sample_run_func)

    assert str(test_virtual_backend.run(qc, 100)) == "{'0': 100}"
    assert test_virtual_backend.run(qc, 100)["0"] == 100
