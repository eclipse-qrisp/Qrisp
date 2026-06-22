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

"""Tests for AQTBackend and AQTJob."""

# Real AQT credentials and hardware are not required.  The ``qiskit_aqt_provider``
# package is injected as a mock via ``sys.modules`` so the tests run in any CI
# environment without optional dependencies.

import sys
from unittest.mock import MagicMock

import pytest

from qrisp import QuantumCircuit
from qrisp.interface.backend import Backend
from qrisp.interface.job import JobCancelledError, JobFailureError, JobResult, JobStatus
from qrisp.interface.provider_backends.aqt_backend import AQTBackend, AQTJob


@pytest.fixture()
def aqt_mocks(monkeypatch):
    """
    Inject mock ``qiskit_aqt_provider`` modules into ``sys.modules``.

    This allows ``AQTBackend.__init__`` to execute its
    ``from qiskit_aqt_provider import ...`` imports without the real package
    being installed.

    Returns
    -------
    tuple[MagicMock, MagicMock]
        ``(mock_provider_cls, mock_sampler_cls)``, which are the mock class
        objects for ``AQTProvider`` and ``AQTSampler`` respectively.
    """
    mock_provider_cls = MagicMock(name="AQTProvider")
    mock_sampler_cls = MagicMock(name="AQTSampler")

    mock_aqt_module = MagicMock()
    mock_aqt_module.AQTProvider = mock_provider_cls

    mock_primitives_module = MagicMock()
    mock_primitives_module.AQTSampler = mock_sampler_cls

    monkeypatch.setitem(sys.modules, "qiskit_aqt_provider", mock_aqt_module)
    monkeypatch.setitem(sys.modules, "qiskit_aqt_provider.primitives", mock_primitives_module)

    return mock_provider_cls, mock_sampler_cls


@pytest.fixture()
def aqt_backend(aqt_mocks):
    """
    A ready-to-use ``AQTBackend`` wired to a mock AQT device named ``"test_device"``.

    The ``AQTSampler`` mock is returned alongside the backend so individual
    tests can configure its return values and inspect its call arguments.

    Returns
    -------
    tuple[AQTBackend, MagicMock]
        ``(backend, mock_sampler_cls)``
    """
    mock_provider_cls, mock_sampler_cls = aqt_mocks

    mock_device = MagicMock()
    mock_device.name = "test_device"
    mock_provider_cls.return_value.get_backend.return_value = mock_device

    backend = AQTBackend("fake_token", "test_device", "test_workspace")
    return backend, mock_sampler_cls


def _simple_circuit() -> QuantumCircuit:
    """Return a minimal 1-qubit, 1-clbit circuit (H then measure)."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def _two_bit_circuit() -> QuantumCircuit:
    """Return a 2-qubit, 2-clbit Bell-state circuit."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _make_quasi_result(quasi_dists: list[dict]) -> MagicMock:
    """Return a mock AQT primitive-job result carrying the given quasi_dists."""
    mock_result = MagicMock()
    mock_result.quasi_dists = quasi_dists
    return mock_result


def _make_aqt_job(quasi_dists: list[dict] | None = None, fail: Exception | None = None):
    """
    Return a mock AQT primitive job.

    Parameters
    ----------
    quasi_dists
        If provided, ``result()`` returns a mock carrying these distributions.
    fail
        If provided, ``result()`` raises this exception instead.
    """
    mock_job = MagicMock()
    if fail is not None:
        mock_job.result.side_effect = fail
    else:
        mock_job.result.return_value = _make_quasi_result(quasi_dists if quasi_dists is not None else [{0: 1.0}])
    return mock_job


class TestAQTBackendConstruction:
    """Constructor validation and name-extraction behaviour."""

    def test_wrong_api_token_type_raises(self):
        """API token must be a string, not an integer."""
        with pytest.raises(TypeError, match="api_token"):
            AQTBackend(123, "ibex", "workspace")

    def test_wrong_workspace_type_raises(self):
        """Workspace must be a string, not an integer."""
        with pytest.raises(TypeError, match="workspace"):
            AQTBackend("token", "ibex", 999)

    def test_wrong_device_instance_type_raises(self):
        """Device instance must be a string, not an integer."""
        with pytest.raises(TypeError, match="device_instance"):
            AQTBackend("token", 42, "workspace")

    def test_missing_dependency_raises_import_error(self, monkeypatch):
        """Setting the module to None in sys.modules simulates a missing install."""
        monkeypatch.setitem(sys.modules, "qiskit_aqt_provider", None)
        with pytest.raises(ImportError, match="pip install qiskit-aqt-provider"):
            AQTBackend("token", "ibex", "workspace")

    def test_name_taken_from_string_attribute(self, aqt_mocks):
        """When ``aqt_backend.name`` is a plain string it is used directly."""
        mock_provider_cls, _ = aqt_mocks
        mock_device = MagicMock()
        mock_device.name = "string_device"
        mock_provider_cls.return_value.get_backend.return_value = mock_device

        backend = AQTBackend("token", "string_device", "ws")
        assert backend.name == "string_device"

    def test_name_taken_from_callable_attribute(self, aqt_mocks):
        """When ``aqt_backend.name`` is callable it is invoked to get the string."""
        mock_provider_cls, _ = aqt_mocks
        mock_device = MagicMock()
        mock_device.name = lambda: "callable_device"
        mock_provider_cls.return_value.get_backend.return_value = mock_device

        backend = AQTBackend("token", "callable_device", "ws")
        assert backend.name == "callable_device"

    def test_provider_called_with_api_token(self, aqt_mocks):
        """``AQTProvider`` is constructed with the supplied API token."""
        mock_provider_cls, _ = aqt_mocks
        mock_device = MagicMock()
        mock_device.name = "dev"
        mock_provider_cls.return_value.get_backend.return_value = mock_device

        AQTBackend("my_secret_token", "dev", "ws")
        mock_provider_cls.assert_called_once_with("my_secret_token")

    def test_get_backend_called_with_device_and_workspace(self, aqt_mocks):
        """``provider.get_backend`` receives the correct device and workspace."""
        mock_provider_cls, _ = aqt_mocks
        mock_device = MagicMock()
        mock_device.name = "ibex"
        mock_provider_cls.return_value.get_backend.return_value = mock_device

        AQTBackend("token", "ibex", "my_workspace")
        mock_provider_cls.return_value.get_backend.assert_called_once_with(name="ibex", workspace="my_workspace")


class TestAQTBackendOptions:
    """Default options and shot-count propagation."""

    def test_default_options_shots_100(self):
        """The default shot count is 100 if not overridden by the user."""
        assert AQTBackend._default_options() == {"shots": 100}

    def test_instance_default_shots_100(self, aqt_backend):
        """AQTBackend instance starts with the default shot count of 100."""
        backend, _ = aqt_backend
        assert backend.options["shots"] == 100

    def test_shots_passed_to_sampler_run(self, aqt_backend):
        """The shot count supplied to ``run()`` is forwarded to ``AQTSampler.run``."""
        backend, mock_sampler_cls = aqt_backend
        mock_sampler = mock_sampler_cls.return_value
        mock_sampler.run.return_value = _make_aqt_job([{0: 1.0}])

        backend.run(_simple_circuit(), shots=42)

        _, run_kwargs = mock_sampler.run.call_args
        assert run_kwargs["shots"] == 42

    def test_update_options_shots(self, aqt_backend):
        """update_options() persists the new shot count for subsequent runs."""
        backend, mock_sampler_cls = aqt_backend
        mock_sampler = mock_sampler_cls.return_value
        mock_sampler.run.return_value = _make_aqt_job([{0: 1.0}])

        backend.update_options(shots=200)
        assert backend.options["shots"] == 200

        backend.run(_simple_circuit())  # no explicit shots → uses updated default

        _, run_kwargs = mock_sampler.run.call_args
        assert run_kwargs["shots"] == 200

    def test_update_options_unknown_key_raises(self, aqt_backend):
        """update_options() raises AttributeError for keys not in _default_options."""
        backend, _ = aqt_backend
        with pytest.raises(AttributeError):
            backend.update_options(nonexistent_option=42)


class TestAQTRunAsync:
    """Correctness of run_async: circuit conversion, result parsing, and
    sampler configuration."""

    def test_quasi_dists_converted_to_bitstrings(self, aqt_backend):
        """Integer keys from quasi_dists are converted to '0'/'1' bitstrings."""
        backend, mock_sampler_cls = aqt_backend
        mock_sampler_cls.return_value.run.return_value = _make_aqt_job([{0: 0.6, 1: 0.4}])

        result = backend.run(_simple_circuit(), shots=100)
        assert result == {"0": 0.6, "1": 0.4}

    def test_bitstrings_zero_padded_to_clbit_width(self, aqt_backend):
        """For a 2-clbit circuit, outcome 0 is rendered as '00', not '0'."""
        backend, mock_sampler_cls = aqt_backend
        # Outcomes: 0b00 → "00", 0b11 → "11"
        mock_sampler_cls.return_value.run.return_value = _make_aqt_job([{0: 0.5, 3: 0.5}])

        result = backend.run(_two_bit_circuit(), shots=100)
        assert "00" in result
        assert "11" in result
        assert "0" not in result
        assert "3" not in result

    def test_results_split_per_circuit(self, aqt_backend):
        """Each circuit in the batch gets its own result dict."""
        backend, mock_sampler_cls = aqt_backend
        mock_sampler_cls.return_value.run.return_value = _make_aqt_job([{0: 0.7, 1: 0.3}, {0: 0.4, 1: 0.6}])

        results = backend.run([_simple_circuit(), _simple_circuit()], shots=100)

        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0] == {"0": 0.7, "1": 0.3}
        assert results[1] == {"0": 0.4, "1": 0.6}

    def test_sampler_transpile_options_set(self, aqt_backend):
        """``set_transpile_options(optimization_level=3)`` is called before every run."""
        backend, mock_sampler_cls = aqt_backend
        mock_sampler_cls.return_value.run.return_value = _make_aqt_job([{0: 1.0}])

        backend.run(_simple_circuit(), shots=100)

        mock_sampler_cls.return_value.set_transpile_options.assert_called_once_with(optimization_level=3)

    def test_new_sampler_created_per_run_async(self, aqt_backend):
        """A fresh ``AQTSampler`` is constructed for each ``run_async`` call."""
        backend, mock_sampler_cls = aqt_backend
        mock_sampler_cls.return_value.run.return_value = _make_aqt_job([{0: 1.0}])

        backend.run(_simple_circuit(), shots=100)
        backend.run(_simple_circuit(), shots=100)

        assert mock_sampler_cls.call_count == 2

    def test_run_async_returns_aqt_job(self, aqt_backend):
        """run_async() returns an AQTJob instance."""
        backend, mock_sampler_cls = aqt_backend
        mock_sampler_cls.return_value.run.return_value = _make_aqt_job([{0: 1.0}])

        job = backend.run_async(_simple_circuit(), shots=100)
        assert isinstance(job, AQTJob)

    def test_run_async_shots_list_warns_and_uses_max(self, aqt_backend):
        """run_async with a list of shots must warn and run all circuits at max(shots)."""
        backend, mock_sampler_cls = aqt_backend
        mock_sampler_cls.return_value.run.return_value = _make_aqt_job([{0: 1.0}, {0: 1.0}])

        with pytest.warns(UserWarning, match="per-circuit shot counts"):
            backend.run_async([_simple_circuit(), _simple_circuit()], shots=[100, 300])

        _, run_kwargs = mock_sampler_cls.return_value.run.call_args
        assert run_kwargs["shots"] == 300  # max([100, 300])


class TestAQTJob:
    """Unit tests for AQTJob: lifecycle, result parsing, and cancellation."""

    def _make_job(
        self,
        quasi_dists: list[dict] | None = None,
        fail: Exception | None = None,
    ) -> AQTJob:
        """Return an AQTJob backed by a mock AQT primitive job."""
        aqt_job = _make_aqt_job(quasi_dists=quasi_dists, fail=fail)
        return AQTJob(
            backend=MagicMock(),
            aqt_job=aqt_job,
            cl_bits_per_circuit=[1],
        )

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

    def test_cancel_always_returns_false(self):
        """AQT does not expose a cancellation API."""
        job = self._make_job()
        assert job.cancel() is False

    def test_cancel_returns_false_when_done(self):
        """cancel() returns False even after result() has been collected."""
        job = self._make_job()
        job.result()
        assert job.cancel() is False

    def test_result_parses_quasi_dists(self):
        """result() converts integer quasi_dist keys to zero-padded bitstrings."""
        job = self._make_job(quasi_dists=[{0: 0.6, 1: 0.4}])
        job_result = job.result()
        assert job_result.get_counts() == {"0": 0.6, "1": 0.4}

    def test_result_returns_job_result_instance(self):
        """result() returns an instance of JobResult."""
        job = self._make_job()
        assert isinstance(job.result(), JobResult)

    def test_result_raises_job_failure_error_on_aqt_failure(self):
        """If the AQT job raises, result() wraps it as JobFailureError."""
        job = self._make_job(fail=RuntimeError("hardware fault"))
        with pytest.raises(JobFailureError):
            job.result()

    def test_result_propagates_timeout_error(self):
        """TimeoutError from the AQT job is re-raised, not wrapped as JobFailureError."""
        job = self._make_job(fail=TimeoutError("deadline exceeded"))
        with pytest.raises(TimeoutError):
            job.result()

    def test_result_timeout_updates_last_known_status(self):
        """last_known_status reflects the AQT job's actual state after a TimeoutError."""
        aqt_job = _make_aqt_job(fail=TimeoutError("deadline exceeded"))
        aqt_job.status.return_value = MagicMock()
        aqt_job.status.return_value.name = "RUNNING"
        job = AQTJob(backend=MagicMock(), aqt_job=aqt_job, cl_bits_per_circuit=[1])
        job.submit()
        with pytest.raises(TimeoutError):
            job.result()
        assert job.last_known_status == JobStatus.RUNNING

    def test_original_exception_preserved_as_cause(self):
        """The original AQT exception is chained so the root cause is visible."""
        job = self._make_job(fail=RuntimeError("network timeout"))
        with pytest.raises(JobFailureError) as exc_info:
            job.result()
        assert "network timeout" in str(exc_info.value.__cause__)

    def test_result_sets_last_known_status_to_done_on_success(self):
        """result() updates last_known_status to DONE after a successful execution."""
        job = self._make_job(quasi_dists=[{0: 1.0}])
        job.result()
        assert job.last_known_status == JobStatus.DONE

    def test_result_raises_job_cancelled_error_when_cancelled(self):
        """A cancelled AQT job raises JobCancelledError, not JobFailureError."""
        aqt_job = _make_aqt_job(fail=RuntimeError("job was cancelled by user"))
        aqt_job.status.return_value = MagicMock()
        aqt_job.status.return_value.name = "CANCELLED"
        job = AQTJob(backend=MagicMock(), aqt_job=aqt_job, cl_bits_per_circuit=[1])
        with pytest.raises(JobCancelledError):
            job.result()

    def test_result_sets_last_known_status_to_cancelled_on_cancellation(self):
        """result() updates last_known_status to CANCELLED when the job was cancelled."""
        aqt_job = _make_aqt_job(fail=RuntimeError("cancelled"))
        aqt_job.status.return_value = MagicMock()
        aqt_job.status.return_value.name = "CANCELLED"
        job = AQTJob(backend=MagicMock(), aqt_job=aqt_job, cl_bits_per_circuit=[1])
        with pytest.raises(JobCancelledError):
            job.result()
        assert job.last_known_status == JobStatus.CANCELLED

    def test_status_done_when_aqt_job_done(self):
        """status() returns DONE when the AQT job reports DONE."""
        aqt_job = _make_aqt_job()
        aqt_job.status.return_value = MagicMock(name="DONE")
        aqt_job.status.return_value.name = "DONE"
        job = AQTJob(backend=MagicMock(), aqt_job=aqt_job, cl_bits_per_circuit=[1])
        assert job.status() == JobStatus.DONE

    def test_status_running_when_aqt_job_running(self):
        """status() returns RUNNING when the AQT job reports RUNNING."""
        aqt_job = _make_aqt_job()
        aqt_job.status.return_value = MagicMock(name="RUNNING")
        aqt_job.status.return_value.name = "RUNNING"
        job = AQTJob(backend=MagicMock(), aqt_job=aqt_job, cl_bits_per_circuit=[1])
        assert job.status() == JobStatus.RUNNING

    def test_status_falls_back_to_running_on_unknown(self):
        """Any unrecognised AQT status string falls back to RUNNING."""
        aqt_job = _make_aqt_job()
        aqt_job.status.return_value = MagicMock()
        aqt_job.status.return_value.name = "SOME_VENDOR_SPECIFIC_STATE"
        job = AQTJob(backend=MagicMock(), aqt_job=aqt_job, cl_bits_per_circuit=[1])
        assert job.status() == JobStatus.RUNNING

    def test_status_falls_back_to_running_when_status_raises(self):
        """If querying AQT job status raises, status() falls back to RUNNING."""
        aqt_job = _make_aqt_job()
        aqt_job.status.side_effect = RuntimeError("connection error")
        job = AQTJob(backend=MagicMock(), aqt_job=aqt_job, cl_bits_per_circuit=[1])
        assert job.status() == JobStatus.RUNNING

    def test_done_returns_true_when_aqt_job_done(self):
        """done() returns True when the AQT job reports DONE."""
        aqt_job = _make_aqt_job()
        aqt_job.status.return_value = MagicMock(name="DONE")
        aqt_job.status.return_value.name = "DONE"
        job = AQTJob(backend=MagicMock(), aqt_job=aqt_job, cl_bits_per_circuit=[1])
        assert job.done() is True

    def test_in_final_state_true_when_done(self):
        """in_final_state() returns True when the AQT job reports DONE."""
        aqt_job = _make_aqt_job()
        aqt_job.status.return_value = MagicMock(name="DONE")
        aqt_job.status.return_value.name = "DONE"
        job = AQTJob(backend=MagicMock(), aqt_job=aqt_job, cl_bits_per_circuit=[1])
        assert job.in_final_state() is True


class TestAQTBackendEdgeCases:
    """Input validation and multi-circuit submission."""

    def test_run_with_invalid_shots_type_raises_type_error(self, aqt_backend):
        """run() rejects non-integer shots with TypeError."""
        backend, _ = aqt_backend
        with pytest.raises(TypeError):
            backend.run(_simple_circuit(), shots="ten")

    def test_run_with_zero_shots_raises_value_error(self, aqt_backend):
        """run() rejects shots=0 with ValueError."""
        backend, _ = aqt_backend
        with pytest.raises(ValueError):
            backend.run(_simple_circuit(), shots=0)

    def test_run_with_negative_shots_raises_value_error(self, aqt_backend):
        """run() rejects negative shot counts with ValueError."""
        backend, _ = aqt_backend
        with pytest.raises(ValueError):
            backend.run(_simple_circuit(), shots=-1)

    def test_run_async_with_invalid_shots_type_raises_type_error(self, aqt_backend):
        """run_async() also rejects non-integer shots, not just run()."""
        backend, _ = aqt_backend
        with pytest.raises(TypeError):
            backend.run_async(_simple_circuit(), shots="ten")

    def test_run_with_list_of_circuits_returns_list(self, aqt_backend):
        """run() returns a list of dicts when a sequence of circuits is submitted."""
        backend, mock_sampler_cls = aqt_backend
        mock_sampler_cls.return_value.run.return_value = _make_aqt_job([{0: 0.6, 1: 0.4}, {0: 0.3, 1: 0.7}])

        results = backend.run([_simple_circuit(), _simple_circuit()], shots=100)

        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0] == {"0": 0.6, "1": 0.4}
        assert results[1] == {"0": 0.3, "1": 0.7}


class TestAQTErrorPropagation:
    """Exceptions from the AQT sampler surface correctly to callers."""

    def test_sampler_run_error_propagates_from_run_async(self, aqt_backend):
        """If ``AQTSampler.run()`` raises at submission time, the error propagates
        directly from ``run_async`` (before a job handle is created)."""
        backend, mock_sampler_cls = aqt_backend
        mock_sampler_cls.return_value.run.side_effect = RuntimeError("submission failed")

        with pytest.raises(RuntimeError, match="submission failed"):
            backend.run_async(_simple_circuit(), shots=100)

    def test_aqt_job_failure_raises_job_failure_error(self, aqt_backend):
        """If the AQT job fails at result time, ``result()`` raises JobFailureError."""
        backend, mock_sampler_cls = aqt_backend
        mock_job = _make_aqt_job(fail=RuntimeError("hardware error"))
        mock_sampler_cls.return_value.run.return_value = mock_job

        with pytest.raises(JobFailureError):
            backend.run(_simple_circuit(), shots=100)

    def test_aqt_job_failure_message_is_preserved(self, aqt_backend):
        """The original AQT exception is chained as __cause__ so root cause is visible."""
        backend, mock_sampler_cls = aqt_backend
        mock_job = _make_aqt_job(fail=RuntimeError("network timeout"))
        mock_sampler_cls.return_value.run.return_value = mock_job

        with pytest.raises(JobFailureError) as exc_info:
            backend.run(_simple_circuit(), shots=100)

        assert exc_info.value.__cause__ is not None
        assert "network timeout" in str(exc_info.value.__cause__)


class TestAQTBackendBatched:
    """Contract-only tests for AQTBackend batching support.

    Integration tests are skipped unconditionally because they require AQT hardware
    credentials and the real ``qiskit-aqt-provider`` package.  These contract tests
    verify the class-level interface and constructor validation, which fire before any
    import of optional dependencies or network call.
    """

    def test_aqt_backend_is_backend_subclass(self):
        """AQTBackend must be a subclass of Backend."""
        assert issubclass(AQTBackend, Backend)

    def test_aqt_backend_has_batched_method(self):
        """AQTBackend must expose the batched() method inherited from Backend."""
        assert callable(getattr(AQTBackend, "batched", None))

    def test_rejects_non_string_api_token(self):
        """AQTBackend.__init__ must raise TypeError for a non-string api_token."""
        with pytest.raises(TypeError, match="api_token"):
            AQTBackend(api_token=42, device_instance="ibex")

    def test_rejects_non_string_device_instance(self):
        """AQTBackend.__init__ must raise TypeError for a non-string device_instance."""
        with pytest.raises(TypeError, match="device_instance"):
            AQTBackend(api_token="token", device_instance=99)

    def test_rejects_non_string_workspace(self):
        """AQTBackend.__init__ must raise TypeError for a non-string workspace."""
        with pytest.raises(TypeError, match="workspace"):
            AQTBackend(api_token="token", device_instance="ibex", workspace=42)

    @pytest.mark.skip(reason="requires AQT hardware credentials and qiskit-aqt-provider")
    def test_integration_batching_properties(self):
        """Placeholder: full batching integration test for AQTBackend.

        To run manually, replace the credentials and remove the skip mark::

            from conftest import CountingWrapper
            counting = CountingWrapper(
                AQTBackend(
                    api_token="YOUR_TOKEN",
                    device_instance="offline_simulator_no_noise",
                )
            )
            bb = counting.batched()
            qf1 = QuantumFloat(3); qf1[:] = 3
            qf2 = QuantumFloat(3); qf2[:] = 5
            res1 = qf1.get_measurement(backend=bb)
            res2 = qf2.get_measurement(backend=bb)
            bb.dispatch()
            assert res1 == {3: 1.0}
            assert res2 == {5: 1.0}
            assert counting.run_async_call_count == 1
        """
