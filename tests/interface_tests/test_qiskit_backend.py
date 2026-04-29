# ********************************************************************************
# * Copyright (c) 2026 the Qrisp Authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Tests for QiskitBackend and QiskitJob.

All tests are fully mocked — no real IBM Quantum credentials or hardware
are required.  The qiskit-aer package is used as the real backend device
(so transpile works correctly), while SamplerV2 is replaced by a mock so
no actual job submission takes place.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
from qiskit_aer import AerSimulator

from qrisp.circuit.quantum_circuit import QuantumCircuit as QrispQuantumCircuit
from qrisp.interface import QiskitBackend
from qrisp.interface.job import (
    JobCancelledError,
    JobFailureError,
    JobResult,
    JobStatus,
)
from qrisp.interface.provider_backends.qiskit_backend import QiskitJob


class _MockCircuitData:
    """Minimal stand-in for a qiskit PrimitiveResult circuit-data object.

    The result-parsing loop in QiskitJob.result() does:

        for reg_name in qiskit_result[i].data:
            reg_data = getattr(qiskit_result[i].data, reg_name)
            for key, val in reg_data.get_counts().items(): ...

    This class satisfies that protocol with a single register named ``"c"``.
    """

    def __init__(self, counts: dict):
        self._reg = MagicMock()
        self._reg.get_counts.return_value = counts

    def __iter__(self):
        return iter(["c"])

    @property
    def c(self):
        return self._reg


def _make_primitive_result(counts_per_circuit: list[dict]) -> MagicMock:
    """Return a mock primitive-job result carrying the given per-circuit counts."""
    circuit_results = []
    for counts in counts_per_circuit:
        mock_cr = MagicMock()
        mock_cr.data = _MockCircuitData(counts)
        circuit_results.append(mock_cr)

    mock_result = MagicMock()
    mock_result.__getitem__ = MagicMock(side_effect=lambda i: circuit_results[i])
    return mock_result


def _make_qiskit_job(
    counts_per_circuit: list[dict] | None = None,
    fail: Exception | None = None,
) -> MagicMock:
    """Return a mock Qiskit primitive job.

    Parameters
    ----------
    counts_per_circuit
        If provided, ``result()`` returns a mock carrying these distributions.
    fail
        If provided, ``result()`` raises this exception instead.
    """
    mock_job = MagicMock()
    if fail is not None:
        mock_job.result.side_effect = fail
    else:
        mock_job.result.return_value = _make_primitive_result(
            counts_per_circuit
            if counts_per_circuit is not None
            else [{"0": 512, "1": 512}]
        )
    return mock_job


def _simple_circuit() -> QrispQuantumCircuit:
    """Return a minimal 1-qubit circuit (H then measure)."""
    qc = QrispQuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    return qc


@pytest.fixture()
def sampler_mock(monkeypatch):
    """Inject a mock SamplerV2 into qiskit_ibm_runtime for the duration of a test."""
    mock_sampler_cls = MagicMock(name="SamplerV2")
    mock_module = MagicMock()
    mock_module.SamplerV2 = mock_sampler_cls
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", mock_module)
    return mock_sampler_cls


@pytest.fixture()
def qiskit_backend(sampler_mock):
    """A QiskitBackend backed by a real AerSimulator device and a mock SamplerV2.

    Using a real AerSimulator ensures that transpile() inside run_async()
    works correctly without hitting real hardware.
    """
    backend = QiskitBackend(backend=AerSimulator(), options={"shots": 100})
    return backend, sampler_mock


class TestQiskitJob:
    """Direct tests for QiskitJob lifecycle and result parsing."""

    def _make_job(
        self,
        counts_per_circuit: list[dict] | None = None,
        fail: Exception | None = None,
        num_circuits: int = 1,
    ) -> QiskitJob:
        qiskit_job = _make_qiskit_job(counts_per_circuit=counts_per_circuit, fail=fail)
        return QiskitJob(
            backend=MagicMock(),
            qiskit_job=qiskit_job,
            num_circuits=num_circuits,
        )

    # --- submit ---

    def test_submit_does_not_raise(self):
        """submit() completes without raising."""
        job = self._make_job()
        job.submit()

    def test_submit_sets_last_known_status_to_queued(self):
        """submit() transitions last_known_status from INITIALIZING to QUEUED."""
        job = self._make_job()
        assert job.last_known_status == JobStatus.INITIALIZING
        job.submit()
        assert job.last_known_status == JobStatus.QUEUED

    # --- result (success path) ---

    def test_result_returns_job_result_instance(self):
        """result() returns a JobResult instance on success."""
        job = self._make_job()
        assert isinstance(job.result(), JobResult)

    def test_result_parses_counts_correctly(self):
        """result() maps bitstring keys to their counts."""
        job = self._make_job(counts_per_circuit=[{"0": 600, "1": 400}])
        assert job.result().get_counts() == {"0": 600, "1": 400}

    def test_result_strips_whitespace_from_keys(self):
        """result() removes non-word characters (spaces) from bitstring keys."""
        job = self._make_job(counts_per_circuit=[{"0 0": 512, "1 1": 512}])
        counts = job.result().get_counts()
        assert "00" in counts
        assert "11" in counts
        assert "0 0" not in counts

    def test_result_sets_last_known_status_to_done_on_success(self):
        """result() updates last_known_status to DONE after a successful execution."""
        job = self._make_job()
        job.result()
        assert job.last_known_status == JobStatus.DONE

    def test_result_batch_splits_per_circuit(self):
        """result() returns one counts dict per submitted circuit."""
        job = self._make_job(
            counts_per_circuit=[{"0": 700, "1": 300}, {"0": 400, "1": 600}],
            num_circuits=2,
        )
        jr = job.result()
        assert jr.num_circuits == 2
        assert jr.get_counts(0) == {"0": 700, "1": 300}
        assert jr.get_counts(1) == {"0": 400, "1": 600}

    # --- result (failure path) ---

    def test_result_raises_job_failure_error_on_exception(self):
        """result() wraps an underlying exception as JobFailureError."""
        job = self._make_job(fail=RuntimeError("hardware fault"))
        with pytest.raises(JobFailureError):
            job.result()

    def test_result_preserves_original_exception_as_cause(self):
        """The original exception is chained so the root cause is visible."""
        job = self._make_job(fail=RuntimeError("network timeout"))
        with pytest.raises(JobFailureError) as exc_info:
            job.result()
        assert "network timeout" in str(exc_info.value.__cause__)

    def test_result_sets_last_known_status_on_failure(self):
        """result() updates last_known_status when the job fails."""
        qiskit_job = _make_qiskit_job(fail=RuntimeError("error"))
        qiskit_job.status.return_value = MagicMock(name="ERROR")
        qiskit_job.status.return_value.name = "ERROR"
        job = QiskitJob(backend=MagicMock(), qiskit_job=qiskit_job, num_circuits=1)
        with pytest.raises(JobFailureError):
            job.result()
        assert job.last_known_status == JobStatus.ERROR

    # --- result (cancellation path) ---

    def test_result_raises_job_cancelled_error_when_cancelled(self):
        """A cancelled Qiskit job raises JobCancelledError, not JobFailureError."""
        qiskit_job = _make_qiskit_job(fail=RuntimeError("job was cancelled"))
        qiskit_job.status.return_value = MagicMock()
        qiskit_job.status.return_value.name = "CANCELLED"
        job = QiskitJob(backend=MagicMock(), qiskit_job=qiskit_job, num_circuits=1)
        with pytest.raises(JobCancelledError):
            job.result()

    def test_result_sets_last_known_status_to_cancelled_on_cancellation(self):
        """result() updates last_known_status to CANCELLED when the job was cancelled."""
        qiskit_job = _make_qiskit_job(fail=RuntimeError("cancelled"))
        qiskit_job.status.return_value = MagicMock()
        qiskit_job.status.return_value.name = "CANCELLED"
        job = QiskitJob(backend=MagicMock(), qiskit_job=qiskit_job, num_circuits=1)
        with pytest.raises(JobCancelledError):
            job.result()
        assert job.last_known_status == JobStatus.CANCELLED

    # --- cancel ---

    def test_cancel_returns_false_when_in_final_state(self):
        """cancel() returns False once the job has reached a terminal state."""
        qiskit_job = _make_qiskit_job()
        qiskit_job.status.return_value = MagicMock(name="DONE")
        qiskit_job.status.return_value.name = "DONE"
        job = QiskitJob(backend=MagicMock(), qiskit_job=qiskit_job, num_circuits=1)
        assert job.cancel() is False

    def test_cancel_returns_true_when_running(self):
        """cancel() returns True and delegates to the underlying Qiskit job."""
        qiskit_job = _make_qiskit_job()
        qiskit_job.status.return_value = MagicMock(name="RUNNING")
        qiskit_job.status.return_value.name = "RUNNING"
        job = QiskitJob(backend=MagicMock(), qiskit_job=qiskit_job, num_circuits=1)
        assert job.cancel() is True
        qiskit_job.cancel.assert_called_once()

    def test_cancel_returns_false_when_cancel_raises(self):
        """cancel() returns False if the underlying Qiskit job raises on cancel()."""
        qiskit_job = _make_qiskit_job()
        qiskit_job.status.return_value = MagicMock(name="RUNNING")
        qiskit_job.status.return_value.name = "RUNNING"
        qiskit_job.cancel.side_effect = RuntimeError("cannot cancel")
        job = QiskitJob(backend=MagicMock(), qiskit_job=qiskit_job, num_circuits=1)
        assert job.cancel() is False


class TestMapQiskitStatus:
    """Tests for the _map_qiskit_status helper."""

    from qrisp.interface.provider_backends.qiskit_backend import _map_qiskit_status

    def _job_with_status(self, name: str) -> MagicMock:
        mock_job = MagicMock()
        mock_job.status.return_value = MagicMock(name=name)
        mock_job.status.return_value.name = name
        return mock_job

    def test_done_maps_to_done(self):
        from qrisp.interface.provider_backends.qiskit_backend import _map_qiskit_status

        assert _map_qiskit_status(self._job_with_status("DONE")) == JobStatus.DONE

    def test_running_maps_to_running(self):
        from qrisp.interface.provider_backends.qiskit_backend import _map_qiskit_status

        assert _map_qiskit_status(self._job_with_status("RUNNING")) == JobStatus.RUNNING

    def test_queued_maps_to_queued(self):
        from qrisp.interface.provider_backends.qiskit_backend import _map_qiskit_status

        assert _map_qiskit_status(self._job_with_status("QUEUED")) == JobStatus.QUEUED

    def test_validating_maps_to_queued(self):
        from qrisp.interface.provider_backends.qiskit_backend import _map_qiskit_status

        assert (
            _map_qiskit_status(self._job_with_status("VALIDATING")) == JobStatus.QUEUED
        )

    def test_cancelled_maps_to_cancelled(self):
        from qrisp.interface.provider_backends.qiskit_backend import _map_qiskit_status

        assert (
            _map_qiskit_status(self._job_with_status("CANCELLED"))
            == JobStatus.CANCELLED
        )

    def test_error_maps_to_error(self):
        from qrisp.interface.provider_backends.qiskit_backend import _map_qiskit_status

        assert _map_qiskit_status(self._job_with_status("ERROR")) == JobStatus.ERROR

    def test_unknown_status_falls_back_to_running(self):
        from qrisp.interface.provider_backends.qiskit_backend import _map_qiskit_status

        assert (
            _map_qiskit_status(self._job_with_status("SOME_VENDOR_STATE"))
            == JobStatus.RUNNING
        )

    def test_status_raises_falls_back_to_running(self):
        """If querying job status raises, _map_qiskit_status returns RUNNING."""
        from qrisp.interface.provider_backends.qiskit_backend import _map_qiskit_status

        mock_job = MagicMock()
        mock_job.status.side_effect = RuntimeError("connection error")
        assert _map_qiskit_status(mock_job) == JobStatus.RUNNING


class TestQiskitBackendRunAsync:
    """Tests for QiskitBackend.run_async() using a mocked SamplerV2."""

    def test_run_async_returns_qiskit_job(self, qiskit_backend):
        """run_async() returns a QiskitJob instance."""
        backend, sampler_mock = qiskit_backend
        sampler_mock.return_value.run.return_value = _make_qiskit_job()

        job = backend.run_async(_simple_circuit(), shots=100)
        assert isinstance(job, QiskitJob)

    def test_run_async_job_starts_queued(self, qiskit_backend):
        """The returned job's last_known_status is QUEUED after run_async()."""
        backend, sampler_mock = qiskit_backend
        sampler_mock.return_value.run.return_value = _make_qiskit_job()

        job = backend.run_async(_simple_circuit(), shots=100)
        assert job.last_known_status == JobStatus.QUEUED

    def test_shots_forwarded_to_sampler(self, qiskit_backend):
        """The shot count passed to run_async() is forwarded to sampler.run()."""
        backend, sampler_mock = qiskit_backend
        sampler_mock.return_value.run.return_value = _make_qiskit_job()

        backend.run_async(_simple_circuit(), shots=42)

        _, run_kwargs = sampler_mock.return_value.run.call_args
        assert run_kwargs["shots"] == 42

    def test_default_shots_used_when_none(self, qiskit_backend):
        """When shots=None, the backend's default shot count is used."""
        backend, sampler_mock = qiskit_backend
        sampler_mock.return_value.run.return_value = _make_qiskit_job()

        backend.run_async(_simple_circuit())

        _, run_kwargs = sampler_mock.return_value.run.call_args
        assert run_kwargs["shots"] == 100  # from options={"shots": 100} in fixture

    def test_invalid_shots_type_raises_type_error(self, qiskit_backend):
        """run_async() rejects non-integer shots with TypeError."""
        backend, _ = qiskit_backend
        with pytest.raises(TypeError):
            backend.run_async(_simple_circuit(), shots="ten")

    def test_zero_shots_raises_value_error(self, qiskit_backend):
        """run_async() rejects shots=0 with ValueError."""
        backend, _ = qiskit_backend
        with pytest.raises(ValueError):
            backend.run_async(_simple_circuit(), shots=0)

    def test_negative_shots_raises_value_error(self, qiskit_backend):
        """run_async() rejects negative shot counts with ValueError."""
        backend, _ = qiskit_backend
        with pytest.raises(ValueError):
            backend.run_async(_simple_circuit(), shots=-1)

    def test_batch_of_circuits_returns_single_job(self, qiskit_backend):
        """Multiple circuits are batched into a single QiskitJob."""
        backend, sampler_mock = qiskit_backend
        sampler_mock.return_value.run.return_value = _make_qiskit_job(
            counts_per_circuit=[{"0": 60, "1": 40}, {"0": 50, "1": 50}]
        )

        job = backend.run_async([_simple_circuit(), _simple_circuit()], shots=100)
        assert isinstance(job, QiskitJob)
        sampler_mock.return_value.run.assert_called_once()


class TestQiskitBackendConstruction:
    """Tests for QiskitBackend constructor branches not covered by the run_async fixture."""

    def test_default_backend_is_aer_simulator(self, sampler_mock):
        """When no backend is provided, AerSimulator is used as the default."""
        backend = QiskitBackend(options={"shots": 100})
        assert "aer" in backend.name.lower()

    def test_options_read_from_backend_when_not_provided(self, sampler_mock):
        """When options is omitted and backend exposes a dict, those options are used."""
        mock_device = MagicMock()
        mock_device.name = "mock_device"
        mock_device.options = {"shots": 200}
        backend = QiskitBackend(backend=mock_device)
        assert backend.options["shots"] == 200

    def test_default_options_used_when_backend_has_no_options(self, sampler_mock):
        """When options is omitted and backend.options is None, _default_options() is used."""
        mock_device = MagicMock()
        mock_device.name = "mock_device"
        mock_device.options = None
        backend = QiskitBackend(backend=mock_device)
        assert backend.options["shots"] == 1024
