"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

import sys
from unittest.mock import MagicMock
import pytest

from qrisp import QuantumCircuit
from qrisp.interface.job import JobCancelledError, JobFailureError, JobStatus
from qrisp.interface import KipuBackend, KipuJob


@pytest.fixture()
def kipu_mocks(monkeypatch):
    """Inject mock qhub dependencies into sys.modules."""
    mock_client_cls = MagicMock(name="HubQuantumClient")
    mock_input_cls = MagicMock(name="CreateJobRequestInput_AzureIonqSimulator")
    mock_item_cls = MagicMock(name="AzureIonqJobInputCircuitItem")

    mock_qhub_api = MagicMock()
    mock_qhub_api.HubQuantumClient = mock_client_cls
    
    mock_jobs_mod = MagicMock()
    mock_jobs_mod.CreateJobRequestInput_AzureIonqSimulator = mock_input_cls
    
    mock_types_mod = MagicMock()
    mock_types_mod.AzureIonqJobInputCircuitItem = mock_item_cls

    monkeypatch.setitem(sys.modules, "qhub.api.quantum", mock_qhub_api)
    monkeypatch.setitem(sys.modules, "qhub.api.quantum.jobs", mock_jobs_mod)
    monkeypatch.setitem(sys.modules, "qhub.api.quantum.types", mock_types_mod)

    return mock_client_cls, mock_input_cls, mock_item_cls


@pytest.fixture()
def kipu_backend(kipu_mocks):
    mock_client_cls, _, _ = kipu_mocks
    mock_client = mock_client_cls.return_value
    
    backend = KipuBackend("fake_api_key", "azure.ionq.simulator")
    return backend, mock_client


def _simple_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


class TestKipuBackendConstruction:
    def test_invalid_types_raise(self):
        with pytest.raises(TypeError, match="api_key"):
            KipuBackend(12345)
        with pytest.raises(TypeError, match="backend_id"):
            KipuBackend("valid_key", backend_id=999)

    def test_missing_dependency_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "qhub.api.quantum", None)
        with pytest.raises(ImportError, match="required qhub API client"):
            KipuBackend("token")


class TestKipuBackendOptions:
    def test_default_shots(self, kipu_backend):
        backend, _ = kipu_backend
        assert backend.options["shots"] == 100

    def test_update_options(self, kipu_backend):
        backend, _ = kipu_backend
        backend.update_options(shots=500)
        assert backend.options["shots"] == 500


class TestKipuRunAsync:
    def test_returns_kipu_job(self, kipu_backend):
        backend, mock_client = kipu_backend
        mock_job = MagicMock()
        mock_job.id = "job-123"
        mock_client.jobs.create_job.return_value = mock_job

        job = backend.run_async(_simple_circuit(), shots=250)
        assert isinstance(job, KipuJob)
        assert job.last_known_status == JobStatus.QUEUED

    def test_shots_list_warns_and_takes_max(self, kipu_backend):
        backend, mock_client = kipu_backend
        mock_job = MagicMock()
        mock_job.id = "job-abc"
        mock_client.jobs.create_job.return_value = mock_job

        with pytest.warns(UserWarning, match="does not support per-circuit shot counts"):
            backend.run_async([_simple_circuit(), _simple_circuit()], shots=[50, 300])

        _, kwargs = mock_client.jobs.create_job.call_args
        assert kwargs["shots"] == 300


class TestKipuJobLifecycle:
    def test_status_mapping(self, kipu_backend):
        backend, mock_client = kipu_backend
        
        mock_status = MagicMock()
        mock_status.status = "RUNNING"
        mock_client.jobs.get_job_status.return_value = mock_status
        
        job = KipuJob(backend, mock_client, ["job-1"], shots=100)
        assert job.status() == JobStatus.RUNNING

    def test_result_success_parsing(self, kipu_backend):
        backend, mock_client = kipu_backend
        
        mock_status = MagicMock()
        mock_status.status = "COMPLETED"
        mock_client.jobs.get_job_status.return_value = mock_status
        mock_client.jobs.get_job_result.return_value = {"histogram": {"00": 0.7, "11": 0.3}}

        job = KipuJob(backend, mock_client, ["job-1"], shots=100)
        res = job.result()
        assert res.get_counts() == {"00": 0.7, "11": 0.3}

    def test_result_raises_failure(self, kipu_backend):
        backend, mock_client = kipu_backend
        mock_status = MagicMock()
        mock_status.status = "FAILED"
        mock_client.jobs.get_job_status.return_value = mock_status

        job = KipuJob(backend, mock_client, ["job-1"], shots=100)
        with pytest.raises(JobFailureError):
            job.result()

    def test_result_raises_cancelled(self, kipu_backend):
        backend, mock_client = kipu_backend
        mock_status = MagicMock()
        mock_status.status = "CANCELLED"
        mock_client.jobs.get_job_status.return_value = mock_status

        job = KipuJob(backend, mock_client, ["job-1"], shots=100)
        with pytest.raises(JobCancelledError):
            job.result()