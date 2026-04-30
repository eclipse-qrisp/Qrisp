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

"""Tests for QiskitBackend and QiskitJob."""

# All tests are fully mocked, so no real IBM Quantum credentials or hardware
# are required. The qiskit-aer package is used as the real backend device
# (so transpile works correctly), while SamplerV2 is replaced by a mock so
# no actual job submission takes place.

import sys
from unittest.mock import MagicMock

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
from qrisp.interface.provider_backends.qiskit_backend import (
    QiskitJob,
    _map_qiskit_status,
)


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
    job_id: str = "mock-job-id",
) -> MagicMock:
    """Return a mock Qiskit primitive job.

    Parameters
    ----------
    counts_per_circuit
        If provided, ``result()`` returns a mock carrying these distributions.
    fail
        If provided, ``result()`` raises this exception instead.
    job_id
        The value returned by ``mock_job.job_id()``.
    """
    mock_job = MagicMock()
    mock_job.job_id.return_value = job_id
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

    def test_job_id_falls_back_to_none_when_extraction_fails(self):
        """QiskitJob.job_id is None when job_id() raises."""
        qiskit_job = _make_qiskit_job()
        qiskit_job.job_id.side_effect = RuntimeError("id unavailable")
        job = QiskitJob(backend=MagicMock(), qiskit_job=qiskit_job, num_circuits=1)
        assert job.job_id is None

    def test_result_forwards_timeout_to_qiskit(self):
        """result(timeout=5) passes timeout=5 to the underlying Qiskit job."""
        job = self._make_job()
        job.result(timeout=5)
        job._qiskit_job.result.assert_called_once_with(timeout=5)

    def test_result_does_not_pass_timeout_when_none(self):
        """result() with no timeout calls the Qiskit job's result() without any kwargs.

        Local backends such as AerSimulator's PrimitiveJob reject an explicit
        timeout=None argument, so it must only be forwarded when not None.
        """
        job = self._make_job()
        job.result()
        job._qiskit_job.result.assert_called_once_with()

    def test_result_propagates_timeout_error(self):
        """TimeoutError from the Qiskit job is re-raised, not wrapped as JobFailureError."""
        job = self._make_job(fail=TimeoutError("deadline exceeded"))
        with pytest.raises(TimeoutError):
            job.result()

    def test_result_raises_immediately_when_already_cancelled(self):
        """result() raises JobCancelledError without calling Qiskit when already CANCELLED."""
        qiskit_job = _make_qiskit_job(fail=RuntimeError("cancelled"))
        qiskit_job.status.return_value = MagicMock()
        qiskit_job.status.return_value.name = "CANCELLED"
        job = QiskitJob(backend=MagicMock(), qiskit_job=qiskit_job, num_circuits=1)
        with pytest.raises(JobCancelledError):
            job.result()
        qiskit_job.result.reset_mock()
        with pytest.raises(JobCancelledError):
            job.result()
        qiskit_job.result.assert_not_called()

    def test_result_raises_immediately_when_already_errored(self):
        """result() raises JobFailureError without calling Qiskit when already ERROR."""
        qiskit_job = _make_qiskit_job(fail=RuntimeError("fault"))
        qiskit_job.status.return_value = MagicMock()
        qiskit_job.status.return_value.name = "ERROR"
        job = QiskitJob(backend=MagicMock(), qiskit_job=qiskit_job, num_circuits=1)
        with pytest.raises(JobFailureError):
            job.result()
        qiskit_job.result.reset_mock()
        with pytest.raises(JobFailureError):
            job.result()
        qiskit_job.result.assert_not_called()


class TestMapQiskitStatus:
    """Tests for the _map_qiskit_status helper."""

    def _job_with_status(self, name: str) -> MagicMock:
        mock_job = MagicMock()
        mock_job.status.return_value = MagicMock(name=name)
        mock_job.status.return_value.name = name
        return mock_job

    def test_done_maps_to_done(self):
        """A Qiskit job with status DONE maps to JobStatus.DONE."""
        assert _map_qiskit_status(self._job_with_status("DONE")) == JobStatus.DONE

    def test_running_maps_to_running(self):
        """A Qiskit job with status RUNNING maps to JobStatus.RUNNING."""
        assert _map_qiskit_status(self._job_with_status("RUNNING")) == JobStatus.RUNNING

    def test_queued_maps_to_queued(self):
        """A Qiskit job with status QUEUED maps to JobStatus.QUEUED."""
        assert _map_qiskit_status(self._job_with_status("QUEUED")) == JobStatus.QUEUED

    def test_validating_maps_to_queued(self):
        """A Qiskit job with status VALIDATING maps to JobStatus.QUEUED."""
        assert (
            _map_qiskit_status(self._job_with_status("VALIDATING")) == JobStatus.QUEUED
        )

    def test_cancelled_maps_to_cancelled(self):
        """A Qiskit job with status CANCELLED maps to JobStatus.CANCELLED."""
        assert (
            _map_qiskit_status(self._job_with_status("CANCELLED"))
            == JobStatus.CANCELLED
        )

    def test_error_maps_to_error(self):
        """A Qiskit job with status ERROR maps to JobStatus.ERROR."""
        assert _map_qiskit_status(self._job_with_status("ERROR")) == JobStatus.ERROR

    def test_unknown_status_falls_back_to_running(self):
        """A Qiskit job with an unrecognized status maps to JobStatus.RUNNING."""
        assert (
            _map_qiskit_status(self._job_with_status("SOME_VENDOR_STATE"))
            == JobStatus.RUNNING
        )

    def test_status_raises_falls_back_to_running(self):
        """If querying job status raises, _map_qiskit_status returns RUNNING."""

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

    def test_run_async_default_shots_fallback_is_1024(self, sampler_mock):
        """When 'shots' is absent from _options, run_async() falls back to 1024 not 1000."""
        # Use a real AerSimulator device so transpile() works, but pass options
        # without a 'shots' key to exercise the fallback path in run_async().
        backend = QiskitBackend(backend=AerSimulator(), options={"custom_key": "val"})
        sampler_mock.return_value.run.return_value = _make_qiskit_job()

        backend.run_async(_simple_circuit())

        _, run_kwargs = sampler_mock.return_value.run.call_args
        assert run_kwargs["shots"] == 1024


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


class TestQiskitBackendIntegration:
    """End-to-end tests that use a real AerSimulator and real SamplerV2."""

    # These tests exercise the full stack — Qrisp circuit → QASM → Qiskit
    # QuantumCircuit → transpile → SamplerV2 submission → DataBin parsing —
    # without any substitution of Qiskit internals.

    # They are deliberately narrow: each test targets the specific behaviour
    # introduced or fixed in the current implementation and would not be
    # caught by the mocked unit tests above.

    @pytest.fixture()
    def real_backend(self):
        """QiskitBackend backed by a real AerSimulator and real SamplerV2."""
        return QiskitBackend(backend=AerSimulator(), options={"shots": 100})

    def test_run_returns_counts_that_sum_to_shots(self, real_backend):
        """run() produces a counts dict whose values sum to the requested shot count."""
        counts = real_backend.run(_simple_circuit(), shots=80)
        assert sum(counts.values()) == 80

    @pytest.mark.parametrize(
        "circuits",
        [
            [_simple_circuit()],
            [_simple_circuit(), _simple_circuit()],
            (_simple_circuit(), _simple_circuit(), _simple_circuit()),
        ],
    )
    def test_run_multiple_circuits_returns_list_of_counts_dicts(
        self, circuits, real_backend
    ):
        """run() with a sequence of circuits returns a list of counts dicts."""
        results = real_backend.run(circuits, shots=20)
        assert isinstance(results, list)
        assert len(results) == len(circuits)
        assert all(isinstance(r, dict) for r in results)

    def test_run_async_result_returns_job_result_instance(self, real_backend):
        """run_async().result() returns a JobResult on a real Aer execution."""
        job = real_backend.run_async(_simple_circuit(), shots=10)
        assert isinstance(job.result(), JobResult)

    def test_job_id_is_non_empty_string(self, real_backend):
        """job_id is a non-empty string extracted from the real Aer job, not None."""
        job = real_backend.run_async(_simple_circuit(), shots=10)
        assert isinstance(job.job_id, str)
        assert len(job.job_id) > 0

    # Forwarding a non-None timeout to IBM Runtime's RuntimeJobV2.result() is
    # verified by the unit test test_result_forwards_timeout_to_qiskit.
    # Here we verify that the default (timeout=None) path works correctly on a
    # real Aer job — local PrimitiveJob.result() does not accept a timeout kwarg
    # at all, so it must not be passed when None.

    def test_result_with_default_timeout_completes_successfully(self, real_backend):
        """result(timeout=None) completes without raising on a real Aer job."""
        job = real_backend.run_async(_simple_circuit(), shots=10)
        result = job.result(timeout=None)
        assert isinstance(result, JobResult)

    def test_result_called_twice_returns_same_cached_object(self, real_backend):
        """Calling result() twice returns the identical cached JobResult instance."""
        job = real_backend.run_async(_simple_circuit(), shots=10)
        r1 = job.result()
        r2 = job.result()
        assert r1 is r2

    def test_last_known_status_is_done_after_result(self, real_backend):
        """last_known_status is DONE after result() completes."""
        job = real_backend.run_async(_simple_circuit(), shots=10)
        job.result()
        assert job.last_known_status == JobStatus.DONE

    # The mocked tests set mock_job.status.return_value.name = "DONE", which
    # trivially satisfies _map_qiskit_status. The real Aer/IBM Runtime job
    # returns a different object (JobStatus enum or string) whose .name value
    # may not match if _map_qiskit_status is wrong.

    def test_status_returns_done_after_completed_job(self, real_backend):
        """status() returns DONE (not RUNNING or INITIALIZING) after Aer execution."""
        job = real_backend.run_async(_simple_circuit(), shots=10)
        job.result()
        assert job.status() == JobStatus.DONE

    # The mocked batch test only validates the indexing loop against
    # _MockCircuitData. The real DataBin.__iter__ and getattr path
    # never runs in the mocked suite.

    def test_batch_of_two_circuits_produces_correct_per_circuit_counts(
        self, real_backend
    ):
        """Batch submission of 2 circuits returns individually correct count dicts."""
        job = real_backend.run_async([_simple_circuit(), _simple_circuit()], shots=50)
        jr = job.result()
        assert jr.num_circuits == 2
        assert sum(jr.get_counts(0).values()) == 50
        assert sum(jr.get_counts(1).values()) == 50

    # The mocked cancel() test controls in_final_state() by faking the Qiskit
    # status. On a real job, cancel() calls status() → _map_qiskit_status on
    # a real object. If the mapping is wrong, in_final_state() returns False
    # and the code tries to cancel an already-done job, which may raise.

    def test_cancel_on_completed_job_returns_false(self, real_backend):
        """cancel() returns False for a real completed Aer job without raising."""
        job = real_backend.run_async(_simple_circuit(), shots=10)
        job.result()
        assert job.cancel() is False
