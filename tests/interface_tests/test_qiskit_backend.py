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

"""Tests for QiskitBackend, QiskitJob, and QiskitRuntimeBackend."""

# All tests are fully mocked, so no real IBM Quantum credentials or hardware
# are required. The qiskit-aer package is used as the real backend device
# (so transpile works correctly), while SamplerV2 is replaced by a mock so
# no actual job submission takes place.

import sys
from unittest.mock import MagicMock
from collections.abc import Mapping

import pytest
from qiskit_aer import AerSimulator

try:
    import qiskit_ibm_runtime

    _qiskit_ibm_runtime_available = True
except ImportError:
    _qiskit_ibm_runtime_available = False

_skip_if_no_runtime = pytest.mark.skipif(
    not _qiskit_ibm_runtime_available,
    reason="qiskit-ibm-runtime is not installed",
)

from qrisp import QuantumFloat
from qrisp.circuit.quantum_circuit import QuantumCircuit as QrispQuantumCircuit
from qrisp.interface import BatchedBackend, QiskitBackend
from qrisp.interface.backend import Backend
from qrisp.interface.job import (
    JobCancelledError,
    JobFailureError,
    JobResult,
    JobStatus,
)
from qrisp.interface.measurement_result import LazyDict
from qrisp.interface.provider_backends.qiskit_backend import (
    QiskitJob,
    QiskitRuntimeBackend,
    _map_qiskit_status,
)
from conftest import CountingWrapper


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
    """Replace BackendSamplerV2 in the qiskit_backend module with a mock for the duration of a test."""
    mock_sampler_cls = MagicMock(name="BackendSamplerV2")
    monkeypatch.setattr(
        "qrisp.interface.provider_backends.qiskit_backend.BackendSamplerV2",
        mock_sampler_cls,
    )
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

    def test_result_timeout_updates_last_known_status(self):
        """last_known_status reflects the Qiskit job's actual state after a TimeoutError."""
        qiskit_job = _make_qiskit_job(fail=TimeoutError("deadline exceeded"))
        qiskit_job.status.return_value = MagicMock()
        qiskit_job.status.return_value.name = "RUNNING"
        job = QiskitJob(backend=MagicMock(), qiskit_job=qiskit_job, num_circuits=1)
        job.submit()
        with pytest.raises(TimeoutError):
            job.result()
        assert job.last_known_status == JobStatus.RUNNING

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

    def test_run_async_shots_list_warns_and_uses_max(self, qiskit_backend):
        """run_async with a list of shots must warn and run all circuits at max(shots)."""
        backend, sampler_mock = qiskit_backend
        sampler_mock.return_value.run.return_value = _make_qiskit_job(
            [{"0": 300}, {"0": 300}]
        )

        with pytest.warns(UserWarning, match="per-circuit shot counts"):
            backend.run_async([_simple_circuit(), _simple_circuit()], shots=[100, 300])

        _, run_kwargs = sampler_mock.return_value.run.call_args
        assert run_kwargs["shots"] == 300  # max([100, 300])

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
        """run() with a sequence of circuits returns a list of MeasurementResult objects."""

        results = real_backend.run(circuits, shots=20)
        assert isinstance(results, list)
        assert len(results) == len(circuits)
        assert all(isinstance(r, Mapping) for r in results)

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

    def test_batched_workflow_populates_lazy_result(self, real_backend):
        """batched() wraps QiskitBackend; dispatch() populates the lazy result via the real Aer stack."""
        from qrisp import QuantumFloat
        from qrisp.interface.measurement_result import LazyDict

        a = QuantumFloat(2)
        a[:] = 3  # deterministic: always 3
        bb = real_backend.batched()
        res = a.get_measurement(backend=bb)
        assert isinstance(res, LazyDict)
        assert not res._populated
        bb.dispatch()
        assert res == {3: 1.0}


# ---------------------------------------------------------------------------
# QiskitRuntimeBackend tests
# ---------------------------------------------------------------------------
# All tests are fully mocked: no real IBM Quantum credentials or network calls
# are made. QiskitRuntimeService, SamplerV2, and Session are replaced by
# MagicMock objects. Where transpile() is exercised (run_async tests), the
# mocked IBM backend is swapped for a real AerSimulator after construction so
# that transpile works correctly without hitting real hardware.


@pytest.fixture()
def runtime_mocks(monkeypatch):
    """Patch qiskit_ibm_runtime with mocks for the duration of a test.

    The IBM backend returned by the mock service has ``options = None`` so that
    ``Backend.__init__`` falls through to ``_default_options()`` instead of
    trying to use a non-Mapping Qiskit Options object.
    """
    mock_ibm_backend = MagicMock()
    mock_ibm_backend.name = "ibm_brisbane"
    mock_ibm_backend.options = None

    mock_service = MagicMock()
    mock_service.least_busy.return_value = mock_ibm_backend
    mock_service.backend.return_value = mock_ibm_backend

    mock_sampler_cls = MagicMock(name="SamplerV2")
    mock_session_cls = MagicMock(name="Session")
    mock_service_cls = MagicMock(return_value=mock_service)

    mock_module = MagicMock()
    mock_module.QiskitRuntimeService = mock_service_cls
    mock_module.SamplerV2 = mock_sampler_cls
    mock_module.Session = mock_session_cls
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", mock_module)

    return {
        "ibm_backend": mock_ibm_backend,
        "service": mock_service,
        "service_cls": mock_service_cls,
        "sampler_cls": mock_sampler_cls,
        "session_cls": mock_session_cls,
    }


@_skip_if_no_runtime
class TestQiskitRuntimeBackendConstruction:
    """Tests for QiskitRuntimeBackend.__init__ with a fully-mocked IBM Runtime."""

    def test_service_constructed_with_token_and_channel(self, runtime_mocks):
        """QiskitRuntimeService is called with the supplied api_token and channel."""
        QiskitRuntimeBackend(api_token="my-token", channel="ibm_quantum_platform")
        runtime_mocks["service_cls"].assert_called_once_with(
            channel="ibm_quantum_platform", token="my-token", instance=None
        )

    def test_least_busy_selected_when_backend_is_none(self, runtime_mocks):
        """service.least_busy() is called when no backend name is supplied."""
        QiskitRuntimeBackend(api_token="token")
        runtime_mocks["service"].least_busy.assert_called_once()
        runtime_mocks["service"].backend.assert_not_called()

    def test_named_backend_resolved_via_service(self, runtime_mocks):
        """service.backend(name) is called when a backend name string is supplied."""
        QiskitRuntimeBackend(api_token="token", backend="ibm_brisbane")
        runtime_mocks["service"].backend.assert_called_once_with("ibm_brisbane")
        runtime_mocks["service"].least_busy.assert_not_called()

    def test_is_instance_of_qiskit_backend(self, runtime_mocks):
        """QiskitRuntimeBackend is a subclass of QiskitBackend."""
        assert isinstance(QiskitRuntimeBackend(api_token="token"), QiskitBackend)

    def test_is_instance_of_backend(self, runtime_mocks):
        """QiskitRuntimeBackend satisfies the Backend abstract interface."""
        assert isinstance(QiskitRuntimeBackend(api_token="token"), Backend)

    def test_job_mode_does_not_create_session(self, runtime_mocks):
        """In mode='job', no Session is created and self.session is not set."""
        backend = QiskitRuntimeBackend(api_token="token", mode="job")
        assert not hasattr(backend, "session")
        runtime_mocks["session_cls"].assert_not_called()

    def test_session_mode_creates_session_bound_to_ibm_backend(self, runtime_mocks):
        """In mode='session', Session is instantiated with the resolved IBM backend."""
        ibm_backend = runtime_mocks["ibm_backend"]
        backend = QiskitRuntimeBackend(api_token="token", mode="session")
        runtime_mocks["session_cls"].assert_called_once_with(ibm_backend)
        assert backend.session is runtime_mocks["session_cls"].return_value

    def test_session_mode_sampler_bound_to_session(self, runtime_mocks):
        """In mode='session', the final SamplerV2 call receives the Session instance."""
        backend = QiskitRuntimeBackend(api_token="token", mode="session")  # noqa: F841
        session_instance = runtime_mocks["session_cls"].return_value
        # The last SamplerV2(...) call must have received the session, not the backend.
        last_sampler_arg = runtime_mocks["sampler_cls"].call_args[0][0]
        assert last_sampler_arg is session_instance

    def test_invalid_mode_raises_value_error(self, runtime_mocks):
        """An unrecognised mode string raises ValueError with a descriptive message."""
        with pytest.raises(ValueError, match="not available"):
            QiskitRuntimeBackend(api_token="token", mode="batch")

    def test_missing_dependency_raises_import_error(self, monkeypatch):
        """Missing qiskit-ibm-runtime raises ImportError with a pip install hint."""
        monkeypatch.delitem(sys.modules, "qiskit_ibm_runtime", raising=False)
        monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", None)
        with pytest.raises(ImportError, match="qiskit-ibm-runtime"):
            QiskitRuntimeBackend(api_token="token")


@_skip_if_no_runtime
class TestQiskitRuntimeBackendOptions:
    """Tests for QiskitRuntimeBackend default options."""

    def test_default_shots_is_1000(self, runtime_mocks):
        """Default shots is 1000, distinct from QiskitBackend's 1024."""
        backend = QiskitRuntimeBackend(api_token="token")
        assert backend.options["shots"] == 1000

    def test_default_options_differ_from_parent(self, runtime_mocks):
        """QiskitRuntimeBackend._default_options() overrides the parent's 1024 default."""
        assert (
            QiskitRuntimeBackend._default_options() != QiskitBackend._default_options()
        )


@_skip_if_no_runtime
class TestQiskitRuntimeBackendSession:
    """Tests for session lifecycle management."""

    def test_close_session_delegates_to_session_close(self, runtime_mocks):
        """close_session() calls close() on the underlying Session object."""
        backend = QiskitRuntimeBackend(api_token="token", mode="session")
        backend.close_session()
        runtime_mocks["session_cls"].return_value.close.assert_called_once()

    def test_close_session_not_available_in_job_mode(self, runtime_mocks):
        """close_session() raises AttributeError in mode='job' (no session is created)."""
        backend = QiskitRuntimeBackend(api_token="token", mode="job")
        with pytest.raises(AttributeError):
            backend.close_session()

    def test_run_async_returns_qiskit_job_in_session_mode(self, runtime_mocks):
        """run_async() returns a QiskitJob when mode='session'.

        Session mode cannot be exercised with a real IBM backend in CI, so the
        IBM authentication layer and Session class are mocked here. The sampler
        is also mocked because a real Session requires live IBM credentials.
        """
        backend = QiskitRuntimeBackend(api_token="token", mode="session")
        backend.backend = AerSimulator()
        runtime_mocks["sampler_cls"].return_value.run.return_value = _make_qiskit_job()
        job = backend.run_async(_simple_circuit(), shots=50)
        assert isinstance(job, QiskitJob)
        runtime_mocks["sampler_cls"].return_value.run.assert_called_once()


@_skip_if_no_runtime
class TestQiskitRuntimeBackendIntegration:
    """End-to-end tests using a real AerSimulator and real SamplerV2."""

    # Only QiskitRuntimeService (IBM authentication) is mocked. The actual
    # circuit-execution pipeline (Qrisp → QASM → Qiskit → transpile → SamplerV2
    # → DataBin parsing) is exercised without any substitution.

    @pytest.fixture()
    def backend(self, monkeypatch):
        """QiskitRuntimeBackend backed by a real AerSimulator and real SamplerV2."""
        from qiskit_ibm_runtime import SamplerV2

        aer = AerSimulator()
        mock_service = MagicMock()
        mock_service.least_busy.return_value = aer

        mock_module = MagicMock()
        mock_module.QiskitRuntimeService = MagicMock(return_value=mock_service)
        mock_module.SamplerV2 = SamplerV2
        mock_module.Session = MagicMock()
        monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", mock_module)

        return QiskitRuntimeBackend(api_token="fake-token")

    def test_run_returns_counts_that_sum_to_shots(self, backend):
        """run() produces a counts dict whose values sum to the requested shot count."""
        counts = backend.run(_simple_circuit(), shots=80)
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
        self, circuits, backend
    ):
        """run() with a sequence of circuits returns a list of MeasurementResult objects."""

        results = backend.run(circuits, shots=20)
        assert isinstance(results, list)
        assert len(results) == len(circuits)
        assert all(isinstance(r, Mapping) for r in results)

    def test_run_async_result_returns_job_result_instance(self, backend):
        """run_async().result() returns a JobResult on a real Aer execution."""
        job = backend.run_async(_simple_circuit(), shots=10)
        assert isinstance(job.result(), JobResult)

    def test_default_shots_of_1000_applied_in_real_execution(self, backend):
        """When no shots are given, the default of 1000 is used and the count sums match."""
        counts = backend.run(_simple_circuit())
        assert sum(counts.values()) == 1000

    def test_last_known_status_is_queued_immediately_after_run_async(self, backend):
        """last_known_status is QUEUED right after run_async(), before result() is called."""
        job = backend.run_async(_simple_circuit(), shots=10)
        assert job.last_known_status == JobStatus.QUEUED

    def test_result_called_twice_returns_same_cached_object(self, backend):
        """Calling result() twice returns the identical cached JobResult instance."""
        job = backend.run_async(_simple_circuit(), shots=10)
        assert job.result() is job.result()

    def test_last_known_status_is_done_after_result(self, backend):
        """last_known_status is DONE after result() completes."""
        job = backend.run_async(_simple_circuit(), shots=10)
        job.result()
        assert job.last_known_status == JobStatus.DONE

    def test_cancel_on_completed_job_returns_false(self, backend):
        """cancel() returns False for a real completed Aer job without raising."""
        job = backend.run_async(_simple_circuit(), shots=10)
        job.result()
        assert job.cancel() is False

    def test_batch_of_two_circuits_produces_correct_per_circuit_counts(self, backend):
        """Batch submission of 2 circuits returns individually correct count dicts."""
        job = backend.run_async([_simple_circuit(), _simple_circuit()], shots=50)
        jr = job.result()
        assert jr.num_circuits == 2
        assert sum(jr.get_counts(0).values()) == 50
        assert sum(jr.get_counts(1).values()) == 50


class TestQiskitBackendBatched:
    """Verify the four BatchedBackend properties for QiskitBackend (AerSimulator)."""

    def test_batched_returns_batched_backend(self):
        """QiskitBackend.batched() must return a BatchedBackend instance."""
        assert isinstance(
            QiskitBackend(backend=AerSimulator()).batched(), BatchedBackend
        )

    def test_result_is_lazy_before_dispatch(self):
        """get_measurement via batched QiskitBackend must raise before dispatch() is called."""
        bb = QiskitBackend(backend=AerSimulator()).batched()
        qf = QuantumFloat(3)
        qf[:] = 4
        res = qf.get_measurement(backend=bb)
        assert isinstance(res, LazyDict)
        with pytest.raises(RuntimeError, match="dispatch"):
            len(res)
        bb.dispatch()
        assert res == {4: 1.0}

    def test_dispatch_populates_two_results(self):
        """dispatch() must populate both results correctly for two circuits."""
        bb = QiskitBackend(backend=AerSimulator()).batched()

        qf1 = QuantumFloat(3)
        qf1[:] = 3
        qf2 = QuantumFloat(3)
        qf2[:] = 5

        res1 = qf1.get_measurement(backend=bb)
        res2 = qf2.get_measurement(backend=bb)
        bb.dispatch()

        assert res1 == {3: 1.0}
        assert res2 == {5: 1.0}

    def test_n_circuits_one_run_async_call(self):
        """Two circuits queued with QiskitBackend must produce exactly one run_async call."""
        counting = CountingWrapper(QiskitBackend(backend=AerSimulator()))
        bb = counting.batched()

        qf1 = QuantumFloat(3)
        qf1[:] = 1
        qf2 = QuantumFloat(3)
        qf2[:] = 2

        qf1.get_measurement(backend=bb)
        qf2.get_measurement(backend=bb)

        assert counting.run_async_call_count == 0
        bb.dispatch()
        assert counting.run_async_call_count == 1
