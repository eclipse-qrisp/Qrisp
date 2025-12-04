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
    """A fake CircuitCompilationOptions to mock IQM compilation options."""

    pass


class FakeBatch:
    """A fake batch object to mock IQM batch results."""

    def __init__(self):
        """Initialize the fake batch with counts."""
        self.counts_batch = [type("FakeCountsObj", (), {"counts": {"00": 5, "11": 5}})]


class FakeIQMClient:
    """A fake IQMClient to mock IQM API calls."""

    def __init__(self, url, token):
        """Initialize the fake client with URL and token."""
        self.url = url
        self.token = token

    def submit_circuits(self, _, **__):
        """Simulate submitting circuits to IQM backend."""
        pass

    def wait_for_results(self, _):
        """Simulate waiting for results."""
        pass

    def get_run_counts(self, _):
        """Simulate getting run counts."""
        # returns format: counts_batch[i].counts
        return FakeBatch()


class FakeIQMBackend:
    """A fake IQMBackend to mock backend behavior."""

    def __init__(self, client):
        """Initialize the fake provider backend with a client."""
        self.client = client

    def serialize_circuit(self, _):
        """Simulate circuit serialization."""
        pass


def _fake_transpile_to_iqm(qc, _):
    """A fake transpile_to_IQM function to mock transpilation behavior."""
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
    """
    Test Qrisp→IQMBackend integration without calling the real IQM API.
    """

    monkeypatch.setattr("iqm.iqm_client.iqm_client.IQMClient", FakeIQMClient)
    monkeypatch.setattr("iqm.qiskit_iqm.iqm_provider.IQMBackend", FakeIQMBackend)
    monkeypatch.setattr("iqm.qiskit_iqm.transpile_to_IQM", _fake_transpile_to_iqm)
    monkeypatch.setattr(
        "iqm.iqm_client.CircuitCompilationOptions", FakeCircuitCompilationOptions
    )

    backend = IQMBackend(
        api_token="FAKE_TOKEN",
        device_instance="garnet",
    )

    assert isinstance(backend, BatchedBackend)

    a = QuantumFloat(2)
    a[:] = 2
    b = a + a

    result = b.get_measurement(backend=backend, shots=10)
    assert result == {0: 0.5, 3: 0.5}
