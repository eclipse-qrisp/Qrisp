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

"""Tests for the Backend-related classes defined in qrisp.interface.backend"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from conftest import MinimalBackend
from qrisp.interface.backend import Backend
from qrisp.interface.job import Job, JobResult, JobStatus
from qrisp.interface.qunicorn.backend_client import BackendClient
from qrisp.interface.qunicorn.backend_server import BackendServer
from qrisp.interface.virtual_backend import VirtualBackend
from qrisp.misc.exceptions import QrispDeprecationWarning


# ===========================================================================
# Full-featured fake backend used by the integration tests
# ===========================================================================


class ExecutionMode:
    """Namespace for the four supported test-backend execution modes."""

    SYNC = "sync"
    ASYNC = "async"
    ERROR = "error"
    CANCEL = "cancel"


class TestJob(Job):
    """
    Concrete :class:`Job` produced by :class:`TestBackend`.

    Demonstrates how a backend author implements the Job contract using
    internal threading, without any threading in the abstract base class.

    Extra attributes (``circuits``, ``shots``) and internal helpers
    (``_set_result``, ``_set_error``, ``_set_cancelled``, ``add_callback``)
    are additions of this concrete class — they are not part of the
    base interface.
    """

    def __init__(self, backend, circuits, shots, job_id=None):
        """Initialise the TestJob with the backend, circuits, shot count, and optional job ID."""
        super().__init__(backend=backend, job_id=job_id)
        self.circuits = circuits
        self.shots = shots
        self._status = JobStatus.INITIALIZING
        self._result_data = None
        self._error = None
        self._done_event = threading.Event()
        self._cancel_requested = threading.Event()
        self._callbacks = []

    # ── Abstract interface ────────────────────────────────────────────

    def submit(self) -> None:
        """Submit the job. TestBackend triggers execution directly in run()."""

    def result(self, timeout=None):
        """Block until the job finishes and return the JobResult, or raise on failure."""
        if not self._done_event.wait(timeout=timeout):
            raise TimeoutError(
                f"Job '{self._job_id}' did not complete within {timeout}s."
            )
        if self._status == JobStatus.ERROR:
            raise RuntimeError(f"Job failed: {self._error}") from self._error
        if self._status == JobStatus.CANCELLED:
            raise RuntimeError("Job was cancelled.")
        return self._result_data

    def cancel(self) -> bool:
        """Request cancellation. Returns False if the job is already in a terminal state."""
        if self.in_final_state():
            return False
        self._cancel_requested.set()
        return True

    def status(self) -> JobStatus:
        """Return the current JobStatus of the job."""
        return self._status

    # ── Convenience helpers (not part of the abstract contract) ──────

    def add_callback(self, fn) -> None:
        """Register a callback invoked when the job reaches a terminal state.

        If the job is already done, the callback is fired immediately.
        """
        if self.in_final_state():
            fn(self)
        else:
            self._callbacks.append(fn)

    def _set_status(self, status: JobStatus) -> None:
        """Set the job status directly."""
        self._status = status

    def _set_result(self, result: JobResult) -> None:
        """Mark the job as DONE, store the result, and fire all registered callbacks."""
        self._result_data = result
        self._status = JobStatus.DONE
        self._done_event.set()
        for cb in self._callbacks:
            try:
                cb(self)
            except Exception:  # noqa: BLE001
                pass

    def _set_error(self, error: Exception) -> None:
        """Mark the job as ERROR, store the exception, and fire all registered callbacks."""
        self._error = error
        self._status = JobStatus.ERROR
        self._done_event.set()
        for cb in self._callbacks:
            try:
                cb(self)
            except Exception:  # noqa: BLE001
                pass

    def _set_cancelled(self) -> None:
        """Mark the job as CANCELLED and fire all registered callbacks."""
        self._status = JobStatus.CANCELLED
        self._done_event.set()
        for cb in self._callbacks:
            try:
                cb(self)
            except Exception:  # noqa: BLE001
                pass


class TestBackend(Backend):
    """
    Full-featured fake backend used by the integration tests.

    Supports four execution modes (sync, async, error, cancel) and generates
    fake measurement counts using a NumPy multinomial draw over all
    2^num_qubits bitstrings.
    """

    def __init__(
        self,
        mode: str = ExecutionMode.SYNC,
        num_qubits: int = 2,
        async_delay: float = 1.0,
        seed: int | None = None,
        name: str | None = None,
        options=None,
    ):
        """Initialise the TestBackend with an execution mode, qubit count, delay, and RNG seed."""
        super().__init__(name=name, options=options)
        valid_modes = {
            ExecutionMode.SYNC,
            ExecutionMode.ASYNC,
            ExecutionMode.ERROR,
            ExecutionMode.CANCEL,
        }
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode '{mode}'.")
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        self.mode = mode
        self._num_qubits = num_qubits
        self.async_delay = async_delay
        self._rng = np.random.default_rng(seed)

    @classmethod
    def _default_options(cls):
        """Return the default runtime options for the TestBackend."""
        return {"shots": 1024}

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits exposed by this fake backend."""
        return self._num_qubits

    @property
    def backend_info(self):
        """Return basic provider and version metadata for the test backend."""
        return {"provider": "Qrisp Test Suite", "version": "0.1.0"}

    @property
    def gate_set(self):
        """Return the set of gate names supported by this fake backend."""
        return ["H", "X", "Y", "Z", "CNOT", "CZ", "RZ", "RX", "RY", "T", "S"]

    def run_async(self, circuits, shots=None) -> TestJob:
        """Submit one or more circuits and return a TestJob according to the current mode."""
        if not isinstance(circuits, list):
            circuits = [circuits]
        n_shots = shots if shots is not None else self._options["shots"]
        job = TestJob(circuits=circuits, shots=n_shots, backend=self)

        if self.mode == ExecutionMode.SYNC:
            self._run_sync(job)
        elif self.mode == ExecutionMode.ASYNC:
            job._set_status(JobStatus.QUEUED)
            threading.Thread(target=self._run_async, args=(job,), daemon=True).start()
        elif self.mode == ExecutionMode.ERROR:
            if self.async_delay == 0:
                self._run_error_sync(job)
            else:
                job._set_status(JobStatus.QUEUED)
                threading.Thread(
                    target=self._run_error_async, args=(job,), daemon=True
                ).start()
        elif self.mode == ExecutionMode.CANCEL:
            job._set_status(JobStatus.QUEUED)
            threading.Thread(
                target=self._run_cancellable, args=(job,), daemon=True
            ).start()
        return job

    def _fake_counts(self, shots: int) -> dict[str, int]:
        """Generate a sparse dict of fake measurement counts using a multinomial draw."""
        n = 2**self._num_qubits
        raw = self._rng.multinomial(shots, np.ones(n) / n)
        return {
            format(i, f"0{self._num_qubits}b"): int(c)
            for i, c in enumerate(raw)
            if c > 0
        }

    def _run_sync(self, job: TestJob) -> None:
        """Execute all circuits synchronously and resolve the job before returning."""
        job._set_status(JobStatus.RUNNING)
        counts = [self._fake_counts(job.shots) for _ in job.circuits]
        job._set_result(JobResult(counts, mode="sync"))

    def _run_async(self, job: TestJob) -> None:
        """Sleep for async_delay seconds, then resolve the job with fake counts."""
        job._set_status(JobStatus.RUNNING)
        time.sleep(self.async_delay)
        counts = [self._fake_counts(job.shots) for _ in job.circuits]
        job._set_result(JobResult(counts, mode="async"))

    def _run_error_sync(self, job: TestJob) -> None:
        """Fail the job synchronously with a simulated hardware fault."""
        job._set_status(JobStatus.RUNNING)
        job._set_error(RuntimeError("Simulated hardware fault."))

    def _run_error_async(self, job: TestJob) -> None:
        """Sleep for async_delay seconds, then fail the job with a simulated hardware fault."""
        job._set_status(JobStatus.RUNNING)
        time.sleep(self.async_delay)
        job._set_error(RuntimeError("Simulated hardware fault."))

    def _run_cancellable(self, job: TestJob) -> None:
        """Execute in steps of 0.5 s, honouring cancellation requests between steps."""
        step = 0.5
        total_steps = max(1, int(self.async_delay / step))
        job._set_status(JobStatus.RUNNING)
        for _ in range(total_steps):
            if job._cancel_requested.is_set():
                job._set_cancelled()
                return
            time.sleep(step)
        counts = [self._fake_counts(job.shots) for _ in job.circuits]
        job._set_result(JobResult(counts, mode="cancel"))


# ===========================================================================
# Minimal dummy backends used only for option / name / abstract-interface tests
# ===========================================================================


class DummyBackend(Backend):
    """
    Minimal subclass of Backend for unit tests.

    The run() method returns a plain dict so tests can inspect the
    arguments that were passed to it without any Job machinery.
    """

    @classmethod
    def _default_options(cls):
        """Return default options with a custom shots value and a flag field."""
        return {"shots": 1000, "flag": False}

    def run_async(self, circuits, shots: int | None = None):
        """No-op run_async used only for instantiation and options tests."""


class BackendNoDefaultOptions(Backend):
    """Backend instantiated without options — must fall back to Backend._default_options()."""

    def run_async(self, circuits, shots: int | None = None):
        """No-op run method used only to satisfy the abstract interface."""


class BackendWithExplicitOptions(Backend):
    """Backend instantiated with explicit options — must override the base defaults entirely."""

    def __init__(self, options=None):
        """Initialise with the provided options mapping."""
        super().__init__(options=options)

    def run_async(self, circuits, shots: int | None = None):
        """No-op run method used only to satisfy the abstract interface."""


class BackendWithChildDefaultOptions(Backend):
    """Backend that defines its own _default_options() at the child class level."""

    @classmethod
    def _default_options(cls):
        """Return child-level defaults including a custom_default field."""
        return {"shots": 1024, "custom_default": 42}

    def run_async(self, circuits, shots: int | None = None):
        """No-op run method used only to satisfy the abstract interface."""


# ===========================================================================
# Test suite
# ===========================================================================


class TestBackendAbstractInterface:
    """Tests that Backend enforces its abstract contract correctly."""

    def test_backend_cannot_be_instantiated_directly(self):
        """Ensure that the abstract Backend class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Backend()

    def test_run_async_is_abstract(self):
        """Ensure that Backend.run_async is declared as an abstract method."""
        assert getattr(Backend.run_async, "__isabstractmethod__", False)

    def test_concrete_subclass_can_be_instantiated(self):
        """Ensure that a concrete subclass implementing run() can be instantiated."""
        assert MinimalBackend() is not None


class TestBackendName:
    """Tests for backend name handling at construction and after mutation."""

    def test_name_defaults_to_class_name(self):
        """Ensure the backend name defaults to the class name when not provided."""
        assert MinimalBackend().name == "MinimalBackend"

    def test_name_can_be_set_explicitly(self):
        """Ensure an explicitly provided name is stored correctly."""
        assert MinimalBackend(name="my_backend").name == "my_backend"

    def test_name_is_mutable(self):
        """Ensure the backend name can be updated after construction."""
        b = MinimalBackend(name="first")
        b.name = "second"
        assert b.name == "second"


class TestBackendOptions:
    """Tests for backend options: defaults, explicit overrides, copy semantics, and update guards."""

    def test_default_options_are_shots_1024(self):
        """Ensure that the base default options contain shots=1024."""
        assert MinimalBackend().options == {"shots": 1024}

    def test_child_default_options_override_base(self):
        """Ensure that a child class _default_options() replaces the base defaults."""
        b = BackendWithChildDefaultOptions()
        assert b.options["shots"] == 1024
        assert b.options["custom_default"] == 42

    def test_explicit_options_override_defaults_entirely(self):
        """Ensure that passing options to the constructor replaces the defaults entirely."""
        b = BackendWithExplicitOptions(options={"shots": 2048, "extra": "value"})
        assert b.options == {"shots": 2048, "extra": "value"}

    def test_no_options_uses_default_options(self):
        """Ensure that omitting options falls back to Backend._default_options()."""
        b = BackendNoDefaultOptions()
        assert b.options == {"shots": 1024}

    def test_options_must_be_mapping_string_rejected(self):
        """Ensure that passing a string as options raises a TypeError."""
        with pytest.raises(TypeError, match="Mapping"):
            MinimalBackend(options="not_a_mapping")  # type: ignore[arg-type]

    def test_options_must_be_mapping_list_rejected(self):
        """Ensure that passing a list as options raises a TypeError."""
        with pytest.raises(TypeError):
            MinimalBackend(options=[1, 2, 3])  # type: ignore[arg-type]

    def test_options_are_copied_from_constructor(self):
        """Ensure that mutating the original mapping after construction does not affect the backend."""
        original = {"shots": 1024, "flag": False}
        b = BackendWithExplicitOptions(options=original)
        original["flag"] = True
        original["shots"] = 9999
        assert b.options["flag"] is False
        assert b.options["shots"] == 1024

    def test_update_options_valid_key(self):
        """Ensure that a valid key can be updated via update_options()."""
        b = MinimalBackend()
        b.update_options(shots=4096)
        assert b.options["shots"] == 4096

    def test_update_options_multiple_valid_keys(self):
        """Ensure that multiple valid keys can be updated in a single call."""

        class MultiBackend(Backend):
            """Backend with multiple configurable options for multi-key update tests."""

            @classmethod
            def _default_options(cls):
                """Return options with shots, flag, and level fields."""
                return {"shots": 1024, "flag": False, "level": 1}

            def run_async(self, circuits, shots=None):
                """No-op run method."""

        b = MultiBackend()
        b.update_options(shots=2048, flag=True, level=3)
        assert b.options["shots"] == 2048
        assert b.options["flag"] is True
        assert b.options["level"] == 3

    def test_update_options_invalid_key_raises_attribute_error(self):
        """Ensure that updating an unknown option key raises an AttributeError."""
        b = MinimalBackend()
        with pytest.raises(AttributeError, match="not a valid backend option"):
            b.update_options(unknown_key=42)

    def test_update_options_invalid_key_does_not_add_new_key(self):
        """Ensure that a failed update_options() call does not add the rejected key."""
        b = MinimalBackend()
        try:
            b.update_options(new_key="value")
        except AttributeError:
            pass
        assert "new_key" not in b.options

    def test_options_are_independent_between_instances(self):
        """Ensure that updating one backend instance does not affect another."""
        b1 = MinimalBackend()
        b2 = MinimalBackend()
        b1.update_options(shots=2048)
        assert b2.options["shots"] == 1024

    def test_backend_with_explicit_options_update(self):
        """Ensure that an explicit-options backend allows updates to known keys and rejects unknown ones."""
        options = {"shots": 1024, "custom_option": "old_value"}
        b = BackendWithExplicitOptions(options=options)
        b.update_options(custom_option="new_value")
        assert b.options["custom_option"] == "new_value"
        with pytest.raises(AttributeError):
            b.update_options(unknown_field=123)

    def test_backend_with_child_defaults_update(self):
        """Ensure that a child-defaults backend allows updates to its custom keys and rejects unknown ones."""
        b = BackendWithChildDefaultOptions()
        b.update_options(shots=2048, custom_default="updated")
        assert b.options["shots"] == 2048
        assert b.options["custom_default"] == "updated"
        with pytest.raises(AttributeError):
            b.update_options(new_param=999)


class TestBackendHardwareMetadata:
    """Tests that all hardware metadata properties return None (or empty dict) by default."""

    @pytest.fixture
    def backend(self):
        """Return a fresh MinimalBackend instance for each test."""
        return MinimalBackend()

    def test_num_qubits_is_none(self, backend):
        """Ensure num_qubits returns None when not overridden."""
        assert backend.num_qubits is None

    def test_connectivity_is_none(self, backend):
        """Ensure connectivity returns None when not overridden."""
        assert backend.connectivity is None

    def test_gate_set_is_none(self, backend):
        """Ensure gate_set returns None when not overridden."""
        assert backend.gate_set is None

    def test_error_rates_is_none(self, backend):
        """Ensure error_rates returns None when not overridden."""
        assert backend.error_rates is None

    def test_backend_health_is_none(self, backend):
        """Ensure backend_health returns None when not overridden."""
        assert backend.backend_health is None

    def test_backend_info_is_none(self, backend):
        """Ensure backend_info returns None when not overridden."""
        assert backend.backend_info is None

    def test_backend_queue_is_none(self, backend):
        """Ensure backend_queue returns None when not overridden."""
        assert backend.backend_queue is None

    def test_capabilities_is_empty_dict(self, backend):
        """Ensure capabilities returns an empty dict when not overridden."""
        assert backend.capabilities == {}

    def test_num_qubits_can_be_overridden(self):
        """Ensure a concrete backend can expose a meaningful num_qubits value."""
        assert TestBackend(num_qubits=5).num_qubits == 5

    def test_backend_info_can_be_overridden(self):
        """Ensure a concrete backend can expose a non-None backend_info dict."""
        info = TestBackend().backend_info
        assert isinstance(info, dict)
        assert "provider" in info

    def test_gate_set_can_be_overridden(self):
        """Ensure a concrete backend can expose a non-empty gate_set list."""
        gs = TestBackend().gate_set
        assert isinstance(gs, list)
        assert len(gs) > 0


class TestDummyBackend:
    """Tests for the DummyBackend helper, which returns a plain dict from run()."""

    def test_backend_is_abstract(self):
        """Ensure the base Backend class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Backend()

    def test_dummy_backend_instantiation_with_name(self):
        """Ensure DummyBackend can be instantiated with an explicit name."""
        b = DummyBackend(name="test_backend")
        assert b.name == "test_backend"

    def test_dummy_backend_instantiation_without_name(self):
        """Ensure DummyBackend defaults to its class name when no name is provided."""
        b = DummyBackend()
        assert b.name == "DummyBackend"

    def test_dummy_backend_default_options_are_correct(self):
        """Ensure DummyBackend initialises with the expected default options."""
        b = DummyBackend()
        assert b.options == {"shots": 1000, "flag": False}


# ===========================================================================
# Integration tests: synchronous execution
# ===========================================================================


class TestSyncExecution:
    """Integration tests for synchronous (blocking) execution via TestBackend."""

    def test_single_circuit_job_is_done_before_run_returns(self):
        """Ensure a sync job reaches DONE status before run() returns to the caller."""
        job = TestBackend(mode=ExecutionMode.SYNC, seed=0).run_async("c")
        assert job.status() == JobStatus.DONE

    def test_single_circuit_result_immediately_available(self):
        """Ensure the result is accessible immediately after run() for a sync job."""
        backend = TestBackend(mode=ExecutionMode.SYNC, num_qubits=2, seed=0)
        result = backend.run_async("c", shots=256).result()
        assert result.num_circuits == 1
        assert sum(result.get_counts().values()) == 256

    def test_bitstring_width_matches_num_qubits(self):
        """Ensure generated bitstrings have exactly num_qubits characters."""
        for n in [1, 2, 3, 4]:
            backend = TestBackend(mode=ExecutionMode.SYNC, num_qubits=n, seed=0)
            result = backend.run_async("c", shots=128).result()
            for bitstring in result.get_counts():
                assert len(bitstring) == n

    def test_batch_of_five_is_done_immediately(self):
        """Ensure a batch of five circuits reaches DONE status before run() returns."""
        job = TestBackend(mode=ExecutionMode.SYNC, seed=1).run_async(
            ["c0", "c1", "c2", "c3", "c4"]
        )
        assert job.status() == JobStatus.DONE

    def test_batch_result_has_one_dict_per_circuit(self):
        """Ensure a batch result contains exactly one counts dict per submitted circuit."""
        result = (
            TestBackend(mode=ExecutionMode.SYNC, seed=1)
            .run_async(["c0", "c1", "c2", "c3", "c4"], shots=100)
            .result()
        )
        assert result.num_circuits == 5

    def test_batch_each_counts_dict_sums_to_shots(self):
        """Ensure each per-circuit counts dict sums to the requested shot count."""
        backend = TestBackend(mode=ExecutionMode.SYNC, num_qubits=2, seed=2)
        result = backend.run_async(["c0", "c1", "c2"], shots=200).result()
        for i in range(3):
            assert sum(result.get_counts(i).values()) == 200

    def test_batch_index_out_of_range_raises(self):
        """Ensure get_counts() raises IndexError for an out-of-range index."""
        result = (
            TestBackend(mode=ExecutionMode.SYNC, seed=0)
            .run_async(["c0", "c1"])
            .result()
        )
        with pytest.raises(IndexError):
            result.get_counts(2)

    def test_single_circuit_normalised_to_one_result(self):
        """Ensure a single circuit submitted as a non-list yields num_circuits == 1."""
        result = (
            TestBackend(mode=ExecutionMode.SYNC, seed=0)
            .run_async("single", shots=512)
            .result()
        )
        assert result.num_circuits == 1

    def test_shots_override_respected(self):
        """Ensure the shots argument to run() overrides the backend's default."""
        backend = TestBackend(mode=ExecutionMode.SYNC, num_qubits=2, seed=0)
        result = backend.run_async("c", shots=333).result()
        assert sum(result.get_counts().values()) == 333

    def test_default_shots_used_when_none(self):
        """Ensure the backend's shots option is used when no shots argument is passed."""
        backend = TestBackend(mode=ExecutionMode.SYNC, num_qubits=2, seed=0)
        backend.update_options(shots=128)
        result = backend.run_async("c").result()
        assert sum(result.get_counts().values()) == 128

    def test_done_and_in_final_state_true_after_sync(self):
        """Ensure done() and in_final_state() both return True after a successful sync job."""
        job = TestBackend(mode=ExecutionMode.SYNC, seed=0).run_async("c")
        assert job.done() is True
        assert job.in_final_state() is True

    def test_metadata_contains_sync_mode(self):
        """Ensure the result metadata carries the mode key set by the sync worker."""
        result = TestBackend(mode=ExecutionMode.SYNC, seed=0).run_async("c").result()
        assert result.metadata.get("mode") == "sync"

    def test_different_seeds_give_different_counts(self):
        """Ensure two backends with different seeds produce different count distributions."""
        counts_a = (
            TestBackend(mode=ExecutionMode.SYNC, num_qubits=3, seed=0)
            .run_async("c", shots=1024)
            .result()
            .get_counts()
        )
        counts_b = (
            TestBackend(mode=ExecutionMode.SYNC, num_qubits=3, seed=99)
            .run_async("c", shots=1024)
            .result()
            .get_counts()
        )
        assert counts_a != counts_b

    def test_same_seed_gives_same_counts(self):
        """Ensure two backends with the same seed produce identical count distributions."""
        counts_a = (
            TestBackend(mode=ExecutionMode.SYNC, num_qubits=2, seed=42)
            .run_async("c", shots=512)
            .result()
            .get_counts()
        )
        counts_b = (
            TestBackend(mode=ExecutionMode.SYNC, num_qubits=2, seed=42)
            .run_async("c", shots=512)
            .result()
            .get_counts()
        )
        assert counts_a == counts_b

    def test_accepts_arbitrary_circuit_types(self):
        """Ensure run() accepts any Python object as a circuit without raising."""
        backend = TestBackend(mode=ExecutionMode.SYNC, seed=0)
        for circuit in ["string", 42, None, object(), {"name": "qft"}, 3.14]:
            assert backend.run_async(circuit, shots=10).done()

    def test_heterogeneous_batch_accepted(self):
        """Ensure a batch of mixed Python objects is accepted and yields the correct num_circuits."""
        backend = TestBackend(mode=ExecutionMode.SYNC, seed=0)
        batch = ["bell", {"depth": 3}, None, 42, object()]
        assert backend.run_async(batch, shots=64).result().num_circuits == 5

    def test_cancel_on_finished_job_returns_false(self):
        """Ensure cancel() returns False when called on a job that is already DONE."""
        job = TestBackend(mode=ExecutionMode.SYNC, seed=0).run_async("c")
        assert job.cancel() is False

    def test_cancel_on_finished_job_does_not_change_status(self):
        """Ensure cancel() on a finished job leaves the status unchanged."""
        job = TestBackend(mode=ExecutionMode.SYNC, seed=0).run_async("c")
        job.cancel()
        assert job.status() == JobStatus.DONE

    def test_result_still_accessible_after_late_cancel(self):
        """Ensure result() remains accessible even if cancel() is called after completion."""
        job = TestBackend(mode=ExecutionMode.SYNC, num_qubits=2, seed=0).run_async(
            "c", shots=256
        )
        job.cancel()
        assert job.result().num_circuits == 1

    def test_multiple_sequential_jobs_from_same_backend(self):
        """Ensure the same backend instance can run multiple sequential jobs correctly."""
        backend = TestBackend(mode=ExecutionMode.SYNC, num_qubits=2, seed=0)
        for _ in range(5):
            result = backend.run_async("c", shots=64).result()
            assert sum(result.get_counts().values()) == 64


# ===========================================================================
# Integration tests: asynchronous execution
# ===========================================================================


class TestAsyncExecution:
    """Integration tests for asynchronous execution via TestBackend."""

    def test_run_returns_before_job_is_done(self):
        """Ensure run() returns immediately before the async job has completed."""
        job = TestBackend(mode=ExecutionMode.ASYNC, async_delay=2.0, seed=0).run_async(
            "c"
        )
        assert not job.done()

    def test_status_is_queued_or_running_immediately_after_run(self):
        """Ensure the job status is QUEUED or RUNNING right after an async run() call."""
        job = TestBackend(mode=ExecutionMode.ASYNC, async_delay=2.0, seed=0).run_async(
            "c"
        )
        assert job.status() in (JobStatus.QUEUED, JobStatus.RUNNING)

    def test_result_blocks_and_returns(self):
        """Ensure result() blocks until the async job completes and returns valid counts."""
        backend = TestBackend(
            mode=ExecutionMode.ASYNC, num_qubits=2, async_delay=0.3, seed=0
        )
        result = backend.run_async("c", shots=256).result()
        assert result.num_circuits == 1
        assert sum(result.get_counts().values()) == 256

    def test_status_is_done_after_result(self):
        """Ensure the job status is DONE after result() has returned successfully."""
        backend = TestBackend(mode=ExecutionMode.ASYNC, async_delay=0.3, seed=0)
        job = backend.run_async("c")
        job.result()
        assert job.status() == JobStatus.DONE

    def test_async_batch_all_circuits_in_result(self):
        """Ensure an async batch result contains one counts dict per submitted circuit."""
        backend = TestBackend(
            mode=ExecutionMode.ASYNC, num_qubits=2, async_delay=0.3, seed=1
        )
        result = backend.run_async(["c0", "c1", "c2", "c3", "c4"], shots=100).result()
        assert result.num_circuits == 5
        for i in range(5):
            assert sum(result.get_counts(i).values()) == 100

    def test_metadata_contains_async_mode(self):
        """Ensure the result metadata carries the mode key set by the async worker."""
        result = (
            TestBackend(mode=ExecutionMode.ASYNC, async_delay=0.3, seed=0)
            .run_async("c")
            .result()
        )
        assert result.metadata.get("mode") == "async"

    def test_callback_fires_when_job_completes(self):
        """Ensure a registered callback is invoked with DONE status when the job completes."""
        backend = TestBackend(mode=ExecutionMode.ASYNC, async_delay=0.3, seed=0)
        job = backend.run_async("c")
        fired = []
        job.add_callback(lambda j: fired.append(j.status()))
        job.result()
        assert fired == [JobStatus.DONE]

    def test_callback_on_already_done_job_fires_immediately(self):
        """Ensure a callback registered on an already-done job fires synchronously."""
        job = TestBackend(mode=ExecutionMode.SYNC, seed=0).run_async("c")
        fired = []
        job.add_callback(lambda j: fired.append(True))
        assert len(fired) == 1

    def test_multiple_callbacks_all_fire(self):
        """Ensure all registered callbacks are invoked when the job reaches a terminal state."""
        backend = TestBackend(mode=ExecutionMode.ASYNC, async_delay=0.3, seed=0)
        job = backend.run_async("c")
        results = []
        job.add_callback(lambda j: results.append("cb1"))
        job.add_callback(lambda j: results.append("cb2"))
        job.result()
        assert "cb1" in results
        assert "cb2" in results

    def test_misbehaving_callback_does_not_crash_job(self):
        """Ensure an exception raised inside a callback does not prevent result() from returning."""
        backend = TestBackend(mode=ExecutionMode.ASYNC, async_delay=0.3, seed=0)
        job = backend.run_async("c")
        job.add_callback(lambda j: 1 / 0)
        result = job.result()
        assert result is not None

    def test_multiple_concurrent_jobs(self):
        """Ensure multiple async jobs submitted concurrently all complete successfully."""
        backend = TestBackend(
            mode=ExecutionMode.ASYNC, num_qubits=2, async_delay=0.3, seed=0
        )
        jobs = [backend.run_async(f"c{i}", shots=100) for i in range(4)]
        results = [job.result() for job in jobs]
        for result in results:
            assert result.num_circuits == 1


# ===========================================================================
# Integration tests: timeout behaviour
# ===========================================================================


class TestTimeoutBehaviour:
    """Tests for the timeout parameter of Job.result()."""

    def test_timeout_raises_timeout_error(self):
        """Ensure result(timeout=...) raises TimeoutError when the deadline expires."""
        job = TestBackend(mode=ExecutionMode.ASYNC, async_delay=5.0, seed=0).run_async(
            "c"
        )
        with pytest.raises(TimeoutError):
            job.result(timeout=0.1)

    def test_job_still_running_after_timeout(self):
        """Ensure the job continues executing after a TimeoutError from result()."""
        job = TestBackend(mode=ExecutionMode.ASYNC, async_delay=5.0, seed=0).run_async(
            "c"
        )
        try:
            job.result(timeout=0.1)
        except TimeoutError:
            pass
        assert not job.done()

    def test_result_accessible_after_timeout_once_complete(self):
        """Ensure result() can be called again successfully after a prior TimeoutError."""
        backend = TestBackend(
            mode=ExecutionMode.ASYNC, num_qubits=2, async_delay=0.4, seed=0
        )
        job = backend.run_async("c", shots=128)
        try:
            job.result(timeout=0.05)  # too short; will time out
        except TimeoutError:
            pass
        result = job.result()  # now wait properly
        assert result.num_circuits == 1


# ===========================================================================
# Integration tests: error behaviour
# ===========================================================================


class TestErrorBehaviour:
    """Integration tests for the error execution mode of TestBackend."""

    def test_sync_error_job_status_is_error(self):
        """Ensure a synchronous error immediately sets the job status to ERROR."""
        job = TestBackend(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run_async(
            "c"
        )
        assert job.status() == JobStatus.ERROR

    def test_sync_error_result_raises_runtime_error(self):
        """Ensure result() raises RuntimeError for a synchronously failed job."""
        job = TestBackend(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run_async(
            "c"
        )
        with pytest.raises(RuntimeError):
            job.result()

    def test_async_error_job_not_done_immediately(self):
        """Ensure an async error job is not yet in a terminal state right after run()."""
        job = TestBackend(mode=ExecutionMode.ERROR, async_delay=0.5, seed=0).run_async(
            "c"
        )
        assert not job.done()

    def test_async_error_result_raises_runtime_error(self):
        """Ensure result() raises RuntimeError after an async error has been set."""
        job = TestBackend(mode=ExecutionMode.ERROR, async_delay=0.3, seed=0).run_async(
            "c"
        )
        with pytest.raises(RuntimeError):
            job.result()

    def test_async_error_status_is_error_after_result(self):
        """Ensure the job status is ERROR after result() has raised on an async failure."""
        job = TestBackend(mode=ExecutionMode.ERROR, async_delay=0.3, seed=0).run_async(
            "c"
        )
        try:
            job.result()
        except RuntimeError:
            pass
        assert job.status() == JobStatus.ERROR

    def test_error_job_is_in_final_state(self):
        """Ensure a failed job is in a final state (in_final_state() is True).

        Note that done() returns False because it means "completed successfully",
        not "reached any terminal state".
        """
        job = TestBackend(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run_async(
            "c"
        )
        assert job.done() is False  # done() means success only
        assert job.in_final_state() is True  # in_final_state() covers ERROR too

    def test_error_job_cancel_returns_false(self):
        """Ensure cancel() returns False on a job that has already failed."""
        job = TestBackend(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run_async(
            "c"
        )
        assert job.cancel() is False

    def test_error_in_batch_affects_whole_job(self):
        """Ensure that a hardware fault in a batch job marks the entire job as ERROR."""
        job = TestBackend(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run_async(
            ["c0", "c1", "c2"]
        )
        assert job.status() == JobStatus.ERROR

    def test_callback_fires_on_error(self):
        """Ensure a registered callback is invoked with ERROR status when the job fails."""
        backend = TestBackend(mode=ExecutionMode.ERROR, async_delay=0.3, seed=0)
        job = backend.run_async("c")
        fired = []
        job.add_callback(lambda j: fired.append(j.status()))
        try:
            job.result()
        except RuntimeError:
            pass
        assert fired == [JobStatus.ERROR]


# ===========================================================================
# Integration tests: cancellation
# ===========================================================================


class TestCancellation:
    """Integration tests for job cancellation via TestBackend in CANCEL mode."""

    def test_cancel_returns_true_on_running_job(self):
        """Ensure cancel() returns True when called on a running job."""
        job = TestBackend(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run_async(
            "c"
        )
        time.sleep(0.6)
        assert job.cancel() is True

    def test_cancel_job_reaches_cancelled_status(self):
        """Ensure the job status reaches CANCELLED after a successful cancel() call."""
        job = TestBackend(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run_async(
            "c"
        )
        time.sleep(0.6)
        job.cancel()
        try:
            job.result()
        except RuntimeError:
            pass
        assert job.status() == JobStatus.CANCELLED

    def test_cancel_result_raises_runtime_error(self):
        """Ensure result() raises RuntimeError after the job has been cancelled."""
        job = TestBackend(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run_async(
            "c"
        )
        time.sleep(0.6)
        job.cancel()
        with pytest.raises(RuntimeError):
            job.result()

    def test_cancelled_job_is_in_final_state(self):
        """Ensure a cancelled job is in a final state (in_final_state() is True).

        Note that done() returns False because it means "completed successfully",
        not "reached any terminal state".
        """
        job = TestBackend(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run_async(
            "c"
        )
        time.sleep(0.6)
        job.cancel()
        try:
            job.result()
        except RuntimeError:
            pass
        assert job.done() is False  # done() means success only
        assert job.in_final_state() is True  # in_final_state() covers CANCELLED too

    def test_cancel_returns_false_on_finished_job(self):
        """Ensure cancel() returns False when the job has already completed successfully."""
        job = TestBackend(mode=ExecutionMode.CANCEL, async_delay=0.3, seed=0).run_async(
            "c"
        )
        job.result()
        assert job.cancel() is False

    def test_cancel_returns_false_on_already_cancelled_job(self):
        """Ensure cancel() returns False when called a second time on an already-cancelled job."""
        job = TestBackend(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run_async(
            "c"
        )
        time.sleep(0.6)
        job.cancel()
        try:
            job.result()
        except RuntimeError:
            pass
        assert job.cancel() is False

    def test_natural_completion_without_cancellation(self):
        """Ensure a CANCEL-mode job completes normally if cancel() is never called."""
        backend = TestBackend(
            mode=ExecutionMode.CANCEL, num_qubits=2, async_delay=0.5, seed=3
        )
        result = backend.run_async("c", shots=128).result()
        assert sum(result.get_counts().values()) == 128

    def test_cancel_batch_job(self):
        """Ensure cancel() returns True when called on a running batch job."""
        job = TestBackend(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run_async(
            ["c0", "c1", "c2", "c3", "c4"]
        )
        time.sleep(0.6)
        assert job.cancel() is True

    def test_callback_fires_on_cancellation(self):
        """Ensure a registered callback is invoked with CANCELLED status after cancellation."""
        job = TestBackend(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run_async(
            "c"
        )
        fired = []
        job.add_callback(lambda j: fired.append(j.status()))
        time.sleep(0.6)
        job.cancel()
        try:
            job.result()
        except RuntimeError:
            pass
        assert fired == [JobStatus.CANCELLED]


# ===========================================================================
# Deprecation tests for the old backend infrastructure
# ===========================================================================


def test_backend_client_deprecation_warning():
    """Ensure BackendClient raises a QrispDeprecationWarning upon instantiation."""
    with pytest.warns(QrispDeprecationWarning):
        _ = BackendClient(api_endpoint="not_used")


def test_backend_server_deprecation_warning():
    """Ensure BackendServer raises a QrispDeprecationWarning upon instantiation."""

    def dummy_run():
        """No-op run function passed to BackendServer to satisfy its constructor."""

    with pytest.warns(QrispDeprecationWarning):
        _ = BackendServer(run_func=dummy_run)


def test_virtual_backend_deprecation_warning():
    """Ensure VirtualBackend raises a QrispDeprecationWarning upon instantiation."""

    def dummy_run():
        """No-op run function passed to VirtualBackend to satisfy its constructor."""

    with pytest.warns(QrispDeprecationWarning):
        _ = VirtualBackend(run_func=dummy_run)
