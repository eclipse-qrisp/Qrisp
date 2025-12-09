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

import pytest

from qrisp import QuantumFloat
from qrisp.interface import BatchedBackend, IQMBackend


class FakeCircuitCompilationOptions:
    """A fake CircuitCompilationOptions to mock IQM behavior."""


class FakeJob:
    """A fake IQM Job to mock job behavior."""

    def __init__(self):

        # We simulate 2 qubits measured 10 times with alternating results
        self.fake_result = [
            {
                0: [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]],
                1: [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]],
            }
        ]

    def wait_for_completion(self):
        """Mock waiting for job completion."""
        return

    def result(self):
        """Mock retrieval of job result."""
        return self.fake_result


class FakeIQMProviderBackend:
    """A fake IQM Backend to mock IQMProviderBackend behavior."""

    def __init__(self, client):
        self.client = client

    def serialize_circuit(self, _):
        """Mock serialization of a circuit."""
        return {"serialized": True}


class FakeIQMClient:
    """A fake IQM Client to mock IQMClient behavior."""

    def __init__(self, iqm_server_url, token, quantum_computer):
        self.url = iqm_server_url
        self.token = token
        self.device = quantum_computer

    def submit_circuits(self, *_, **__):
        """Mock submission of circuits."""
        return FakeJob()


def fake_transpile_to_IQM(qc, _):
    """A fake transpiler to mock transpilation to IQM hardware."""
    return qc


class TestIQMBackendInitialization:
    """Test suite for IQMBackend initialization and parameter validation."""

    def test_api_token_must_be_string(self):
        """Test that the API token must be a string."""
        with pytest.raises(TypeError):
            IQMBackend(123, "garnet")

    def test_device_instance_must_be_string(self):
        """Test that the device instance must be a string."""
        with pytest.raises(TypeError):
            IQMBackend("mock_api_token", 123)

    def test_server_url_and_device_instance_conflict(self):
        """Test that error is raised when both server URL and device instance are provided."""
        with pytest.raises(ValueError):
            IQMBackend("mock_api_token", "garnet", server_url="www.qrisp.eu")

    def test_server_url_or_device_instance_required(self):
        """Test that error is raised when neither server URL nor device instance is provided."""
        with pytest.raises(ValueError):
            IQMBackend("mock_api_token")


def test_iqm_backend_measurement(monkeypatch):
    """Test IQMBackend measurement functionality with mocked IQM client and backend."""

    monkeypatch.setattr("iqm.iqm_client.iqm_client.IQMClient", FakeIQMClient)
    monkeypatch.setattr(
        "iqm.qiskit_iqm.iqm_provider.IQMBackend", FakeIQMProviderBackend
    )
    monkeypatch.setattr("iqm.qiskit_iqm.transpile_to_IQM", fake_transpile_to_IQM)
    monkeypatch.setattr(
        "iqm.iqm_client.CircuitCompilationOptions", FakeCircuitCompilationOptions
    )

    backend = IQMBackend(api_token="FAKE", device_instance="garnet")
    assert isinstance(backend, BatchedBackend)

    a = QuantumFloat(2)
    a[:] = 2
    b = a + a

    result = b.get_measurement(backend=backend, shots=10)
    assert result == {0: 0.5, 3: 0.5}
