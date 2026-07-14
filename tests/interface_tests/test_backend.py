# """
# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
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
# """

"""Tests for the Backend-related classes defined in qrisp.interface.backend"""

from __future__ import annotations

import threading
import time
import unittest.mock
from typing import cast

import numpy as np
import pytest
from conftest import MinimalBackend, MinimalJob

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.interface.backend import Backend, BackendLike
from qrisp.interface.job import (
    Job,
    JobCancelledError,
    JobFailureError,
    JobResult,
    JobStatus,
)
from qrisp.interface.virtual_backend import VirtualBackend
from qrisp.misc.exceptions import QrispDeprecationWarning


class ExecutionMode:
    """Namespace for the four supported test-backend execution modes."""

    SYNC = "sync"
    ASYNC = "async"
    ERROR = "error"
    CANCEL = "cancel"


class JobTest(Job):
    """Concrete :class:`Job` produced by :class:`BackendTest`.

    Demonstrates how a backend author implements the Job contract using
    internal threading, without any threading in the abstract base class.

    Extra attributes (``circuits``, ``shots``) and internal helpers
    (``_set_result``, ``_set_error``, ``_set_cancelled``, ``add_callback``)
    are additions of this concrete class. That is, they are not part of the
    base interface.
    """

    def __init__(self, backend, circuits, shots, job_id=None):
        """Initialise the JobTest with the backend, circuits, shot count, and optional job ID."""
        super().__init__(backend=backend, job_id=job_id)
        self.circuits = circuits
        self.shots = shots
        self._result_data = None
        self._error = None
        self._done_event = threading.Event()
        self._cancel_requested = threading.Event()
        self._callbacks = []

    # ── Abstract interface ────────────────────────────────────────────

    def submit(self) -> None:
        """Transition to QUEUED. BackendTest then drives further state changes."""
        self._last_known_status = JobStatus.QUEUED

    def result(self, timeout=None) -> JobResult:
        """Block until the job finishes and return the JobResult, or raise on failure."""
        if not self._done_event.wait(timeout=timeout):
            raise TimeoutError(f"Job '{self._job_id}' did not complete within {timeout}s.")
        # Pass the already-known terminal status so _raise_for_status
        # does not make a redundant status() call.
        self._raise_for_status(self._last_known_status)
        return cast(JobResult, self._result_data)

    def cancel(self) -> bool:
        """Request cancellation. Returns False if the job is already in a terminal state."""
        if self.in_final_state():
            return False
        self._cancel_requested.set()
        return True

    def status(self) -> JobStatus:
        """Return the current JobStatus of the job."""
        return self._last_known_status

    # ------------------------------------------------------------------
    # Convenience helpers (not part of the abstract contract)
    # ------------------------------------------------------------------

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
        self._last_known_status = status

    def _set_result(self, result: JobResult) -> None:
        """Mark the job as DONE, store the result, and fire all registered callbacks."""
        self._result_data = result
        self._last_known_status = JobStatus.DONE
        self._done_event.set()
        for cb in self._callbacks:
            try:
                cb(self)
            except Exception:
                pass

    def _set_error(self, error: Exception) -> None:
        """Mark the job as ERROR, store the exception, and fire all registered callbacks."""
        self._failure_cause = error  # base-class attribute, chained by _raise_for_status
        self._last_known_status = JobStatus.ERROR
        self._done_event.set()
        for cb in self._callbacks:
            try:
                cb(self)
            except Exception:
                pass

    def _set_cancelled(self) -> None:
        """Mark the job as CANCELLED and fire all registered callbacks."""
        self._last_known_status = JobStatus.CANCELLED
        self._done_event.set()
        for cb in self._callbacks:
            try:
                cb(self)
            except Exception:
                pass


class BackendTest(Backend):
    """Full-featured fake backend used by the integration tests.

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
        """Initialise the BackendTest with an execution mode, qubit count, delay, and RNG seed."""
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
        """Return the default runtime options for the BackendTest."""
        return {"shots": 1024}

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits exposed by this fake backend."""
        return self._num_qubits

    @property
    def info(self):
        """Return basic provider and version metadata for the test backend."""
        return {"provider": "Qrisp Test Suite", "version": "0.1.0"}

    @property
    def gate_set(self):
        """Return the set of gate names supported by this fake backend."""
        return ["H", "X", "Y", "Z", "CNOT", "CZ", "RZ", "RX", "RY", "T", "S"]

    def run_async(self, circuits, shots: int | list[int] | None = None) -> JobTest:
        """Submit one or more circuits and return a JobTest according to the current mode."""
        # We use a different check for list/tuple here to allow
        # other iterable types (e.g. generators) to be accepted as batches.
        if not isinstance(circuits, (list, tuple)):
            circuits = [circuits]
        else:
            circuits = list(circuits)

        n_shots = shots if shots is not None else self._options["shots"]
        job = JobTest(circuits=circuits, shots=n_shots, backend=self)

        job.submit()  # INITIALIZING → QUEUED

        if self.mode == ExecutionMode.SYNC:
            self._run_sync(job)  # QUEUED → RUNNING → DONE
        elif self.mode == ExecutionMode.ASYNC:
            # submit() already set QUEUED; just start the worker thread.
            threading.Thread(target=self._run_async, args=(job,), daemon=True).start()
        elif self.mode == ExecutionMode.ERROR:
            if self.async_delay == 0:
                self._run_error_sync(job)  # QUEUED → RUNNING → ERROR
            else:
                threading.Thread(target=self._run_error_async, args=(job,), daemon=True).start()
        elif self.mode == ExecutionMode.CANCEL:
            threading.Thread(target=self._run_cancellable, args=(job,), daemon=True).start()
        return job

    def _fake_counts(self, shots: int) -> dict[str, int]:
        """Generate a sparse dict of fake measurement counts using a multinomial draw."""
        n = 2**self._num_qubits
        raw = self._rng.multinomial(shots, np.ones(n) / n)
        return {format(i, f"0{self._num_qubits}b"): int(c) for i, c in enumerate(raw) if c > 0}

    def _run_sync(self, job: JobTest) -> None:
        """Execute all circuits synchronously and resolve the job before returning."""
        job._set_status(JobStatus.RUNNING)
        counts = [self._fake_counts(job.shots) for _ in job.circuits]
        job._set_result(JobResult(counts, mode="sync"))

    def _run_async(self, job: JobTest) -> None:
        """Sleep for async_delay seconds, then resolve the job with fake counts."""
        job._set_status(JobStatus.RUNNING)
        time.sleep(self.async_delay)
        counts = [self._fake_counts(job.shots) for _ in job.circuits]
        job._set_result(JobResult(counts, mode="async"))

    def _run_error_sync(self, job: JobTest) -> None:
        """Fail the job synchronously with a simulated hardware fault."""
        job._set_status(JobStatus.RUNNING)
        job._set_error(RuntimeError("Simulated hardware fault."))

    def _run_error_async(self, job: JobTest) -> None:
        """Sleep for async_delay seconds, then fail the job with a simulated hardware fault."""
        job._set_status(JobStatus.RUNNING)
        time.sleep(self.async_delay)
        job._set_error(RuntimeError("Simulated hardware fault."))

    def _run_cancellable(self, job: JobTest) -> None:
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


class DummyBackend(Backend):
    """Minimal subclass of Backend for unit tests.

    The run() method returns a plain dict so tests can inspect the
    arguments that were passed to it without any Job machinery.
    """

    @classmethod
    def _default_options(cls):
        """Return default options with a custom shots value and a flag field."""
        return {"shots": 1000, "flag": False}

    def run_async(self, circuits, shots: int | list[int] | None = None):
        """No-op run_async used only for instantiation and options tests."""


class BackendNoDefaultOptions(Backend):
    """Backend instantiated without options — must fall back to Backend._default_options()."""

    def run_async(self, circuits, shots: int | list[int] | None = None):
        """No-op run method used only to satisfy the abstract interface."""


class BackendWithExplicitOptions(Backend):
    """Backend instantiated with explicit options — must override the base defaults entirely."""

    def __init__(self, options=None):
        """Initialise with the provided options mapping."""
        super().__init__(options=options)

    def run_async(self, circuits, shots: int | list[int] | None = None):
        """No-op run method used only to satisfy the abstract interface."""


class BackendWithChildDefaultOptions(Backend):
    """Backend that defines its own _default_options() at the child class level."""

    @classmethod
    def _default_options(cls):
        """Return child-level defaults including a custom_default field."""
        return {"shots": 1024, "custom_default": 42}

    def run_async(self, circuits, shots: int | list[int] | None = None):
        """No-op run method used only to satisfy the abstract interface."""


class BackendTestAbstractInterface:
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


class BackendTestName:
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


class BackendTestOptions:
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
            MinimalBackend(options="not_a_mapping")

    def test_options_must_be_mapping_list_rejected(self):
        """Ensure that passing a list as options raises a TypeError."""
        with pytest.raises(TypeError):
            MinimalBackend(options=[1, 2, 3])

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

    def test_update_options_no_args_is_no_op(self):
        """Calling update_options() with no arguments must leave options unchanged and not raise."""
        b = MinimalBackend()
        original = dict(b.options)
        b.update_options()
        assert dict(b.options) == original


class BackendTestHardwareMetadata:
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

    def test_health_is_none(self, backend):
        """Ensure health returns None when not overridden."""
        assert backend.health is None

    def test_info_is_none(self, backend):
        """Ensure info returns None when not overridden."""
        assert backend.info is None

    def test_queue_is_none(self, backend):
        """Ensure queue returns None when not overridden."""
        assert backend.queue is None

    def test_num_qubits_can_be_overridden(self):
        """Ensure a concrete backend can expose a meaningful num_qubits value."""
        assert BackendTest(num_qubits=5).num_qubits == 5

    def test_info_can_be_overridden(self):
        """Ensure a concrete backend can expose a non-None info dict."""
        info = BackendTest().info
        assert isinstance(info, dict)
        assert "provider" in info

    def test_gate_set_can_be_overridden(self):
        """Ensure a concrete backend can expose a non-empty gate_set list."""
        gs = BackendTest().gate_set
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


class TestSyncExecution:
    """Integration tests for synchronous (blocking) execution via BackendTest."""

    def test_single_circuit_job_is_done_before_run_returns(self):
        """Ensure a sync job reaches DONE status before run() returns to the caller."""
        job = BackendTest(mode=ExecutionMode.SYNC, seed=0).run_async("c")
        assert job.status() == JobStatus.DONE

    def test_single_circuit_result_immediately_available(self):
        """Ensure the result is accessible immediately after run() for a sync job."""
        backend = BackendTest(mode=ExecutionMode.SYNC, num_qubits=2, seed=0)
        result = backend.run_async("c", shots=256).result()
        assert result.num_circuits == 1
        assert sum(result.get_counts().values()) == 256

    def test_bitstring_width_matches_num_qubits(self):
        """Ensure generated bitstrings have exactly num_qubits characters."""
        for n in [1, 2, 3, 4]:
            backend = BackendTest(mode=ExecutionMode.SYNC, num_qubits=n, seed=0)
            result = backend.run_async("c", shots=128).result()
            for bitstring in result.get_counts():
                assert len(bitstring) == n

    def test_batch_of_five_is_done_immediately(self):
        """Ensure a batch of five circuits reaches DONE status before run() returns."""
        job = BackendTest(mode=ExecutionMode.SYNC, seed=1).run_async(["c0", "c1", "c2", "c3", "c4"])
        assert job.status() == JobStatus.DONE

    def test_batch_result_has_one_dict_per_circuit(self):
        """Ensure a batch result contains exactly one counts dict per submitted circuit."""
        result = (
            BackendTest(mode=ExecutionMode.SYNC, seed=1).run_async(["c0", "c1", "c2", "c3", "c4"], shots=100).result()
        )
        assert result.num_circuits == 5

    def test_batch_each_counts_dict_sums_to_shots(self):
        """Ensure each per-circuit counts dict sums to the requested shot count."""
        backend = BackendTest(mode=ExecutionMode.SYNC, num_qubits=2, seed=2)
        result = backend.run_async(["c0", "c1", "c2"], shots=200).result()
        for i in range(3):
            assert sum(result.get_counts(i).values()) == 200

    def test_batch_index_out_of_range_raises(self):
        """Ensure get_counts() raises IndexError for an out-of-range index."""
        result = BackendTest(mode=ExecutionMode.SYNC, seed=0).run_async(["c0", "c1"]).result()
        with pytest.raises(IndexError):
            result.get_counts(2)

    def test_single_circuit_normalised_to_one_result(self):
        """Ensure a single circuit submitted as a non-list yields num_circuits == 1."""
        result = BackendTest(mode=ExecutionMode.SYNC, seed=0).run_async("single", shots=512).result()
        assert result.num_circuits == 1

    def test_shots_override_respected(self):
        """Ensure the shots argument to run() overrides the backend's default."""
        backend = BackendTest(mode=ExecutionMode.SYNC, num_qubits=2, seed=0)
        result = backend.run_async("c", shots=333).result()
        assert sum(result.get_counts().values()) == 333

    def test_default_shots_used_when_none(self):
        """Ensure the backend's shots option is used when no shots argument is passed."""
        backend = BackendTest(mode=ExecutionMode.SYNC, num_qubits=2, seed=0)
        backend.update_options(shots=128)
        result = backend.run_async("c").result()
        assert sum(result.get_counts().values()) == 128

    def test_done_and_in_final_state_true_after_sync(self):
        """Ensure done() and in_final_state() both return True after a successful sync job."""
        job = BackendTest(mode=ExecutionMode.SYNC, seed=0).run_async("c")
        assert job.done() is True
        assert job.in_final_state() is True

    def test_metadata_contains_sync_mode(self):
        """Ensure the result metadata carries the mode key set by the sync worker."""
        result = BackendTest(mode=ExecutionMode.SYNC, seed=0).run_async("c").result()
        assert result.metadata.get("mode") == "sync"

    def test_different_seeds_give_different_counts(self):
        """Ensure two backends with different seeds produce different count distributions."""
        counts_a = (
            BackendTest(mode=ExecutionMode.SYNC, num_qubits=3, seed=0).run_async("c", shots=1024).result().get_counts()
        )
        counts_b = (
            BackendTest(mode=ExecutionMode.SYNC, num_qubits=3, seed=99).run_async("c", shots=1024).result().get_counts()
        )
        assert counts_a != counts_b

    def test_same_seed_gives_same_counts(self):
        """Ensure two backends with the same seed produce identical count distributions."""
        counts_a = (
            BackendTest(mode=ExecutionMode.SYNC, num_qubits=2, seed=42).run_async("c", shots=512).result().get_counts()
        )
        counts_b = (
            BackendTest(mode=ExecutionMode.SYNC, num_qubits=2, seed=42).run_async("c", shots=512).result().get_counts()
        )
        assert counts_a == counts_b

    def test_accepts_arbitrary_circuit_types(self):
        """Ensure run() accepts any Python object as a circuit without raising."""
        backend = BackendTest(mode=ExecutionMode.SYNC, seed=0)
        for circuit in ["string", 42, None, object(), {"name": "qft"}, 3.14]:
            assert backend.run_async(circuit, shots=10).done()

    def test_heterogeneous_batch_accepted(self):
        """Ensure a batch of mixed Python objects is accepted and yields the correct num_circuits."""
        backend = BackendTest(mode=ExecutionMode.SYNC, seed=0)
        batch = ["bell", {"depth": 3}, None, 42, object()]
        assert backend.run_async(batch, shots=64).result().num_circuits == 5

    def test_cancel_on_finished_job_returns_false(self):
        """Ensure cancel() returns False when called on a job that is already DONE."""
        job = BackendTest(mode=ExecutionMode.SYNC, seed=0).run_async("c")
        assert job.cancel() is False

    def test_cancel_on_finished_job_does_not_change_status(self):
        """Ensure cancel() on a finished job leaves the status unchanged."""
        job = BackendTest(mode=ExecutionMode.SYNC, seed=0).run_async("c")
        job.cancel()
        assert job.status() == JobStatus.DONE

    def test_result_still_accessible_after_late_cancel(self):
        """Ensure result() remains accessible even if cancel() is called after completion."""
        job = BackendTest(mode=ExecutionMode.SYNC, num_qubits=2, seed=0).run_async("c", shots=256)
        job.cancel()
        assert job.result().num_circuits == 1

    def test_multiple_sequential_jobs_from_same_backend(self):
        """Ensure the same backend instance can run multiple sequential jobs correctly."""
        backend = BackendTest(mode=ExecutionMode.SYNC, num_qubits=2, seed=0)
        for _ in range(5):
            result = backend.run_async("c", shots=64).result()
            assert sum(result.get_counts().values()) == 64

    def test_run_async_accepts_tuple_of_circuits(self):
        """run_async() must treat a tuple of circuits as a batch, not as a single circuit."""
        backend = BackendTest(mode=ExecutionMode.SYNC, seed=0)
        circuits = ("a", "b", "c")
        result = backend.run_async(circuits, shots=32).result()
        assert result.num_circuits == 3

    def test_run_accepts_tuple_of_circuits(self):
        """run() must return a list of results when a tuple of circuits is submitted."""
        backend = BackendTest(mode=ExecutionMode.SYNC, seed=0)
        circuits = ("a", "b")
        result = backend.run(circuits, shots=32)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_run_async_tuple_and_list_produce_same_result(self):
        """run_async() must produce the same number of results for a tuple and an equivalent list."""
        backend = BackendTest(mode=ExecutionMode.SYNC, seed=42)
        circuits = ["x", "y", "z"]
        result_list = backend.run_async(circuits, shots=16).result()
        result_tuple = backend.run_async(tuple(circuits), shots=16).result()
        assert result_list.num_circuits == result_tuple.num_circuits

    def test_result_idempotent_on_done_job(self):
        """Calling result() twice on a completed job must return consistent counts without raising."""
        backend = MinimalBackend()
        job = backend.run_async("c", shots=128)
        first = job.result()
        second = job.result()
        assert second.all_counts == first.all_counts


class TestRunReturnType:
    """Tests that Backend.run() returns a dict for a single QuantumCircuit and a list for a batch.

    This is the primary backward-compatibility guarantee of the run() method.
    The distinction is driven by isinstance(circuits, QuantumCircuit) in the base class.
    """

    def test_run_single_quantum_circuit_returns_mapping(self):
        """run() must return a Mapping (not a list) when passed a single QuantumCircuit."""
        from collections.abc import Mapping

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure(0)
        result = MinimalBackend().run(qc, shots=128)
        assert isinstance(result, Mapping)

    def test_run_single_quantum_circuit_dict_has_string_keys(self):
        """The dict returned for a single QuantumCircuit must have string bitstring keys."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure(0)
        result = MinimalBackend().run(qc, shots=64)
        assert all(isinstance(k, str) for k in result)

    def test_run_list_of_quantum_circuits_returns_list(self):
        """run() must return a list when passed a list of QuantumCircuits."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure(0)
        result = MinimalBackend().run([qc, qc], shots=64)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_run_list_of_quantum_circuits_each_element_is_mapping(self):
        """Each element of the list returned by run([qc, qc]) must be a Mapping."""
        from collections.abc import Mapping

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure(0)
        result = MinimalBackend().run([qc, qc], shots=64)
        assert all(isinstance(r, Mapping) for r in result)

    def test_run_async_single_quantum_circuit_num_circuits_is_one(self):
        """run_async() must return a job with num_circuits==1 for a single QuantumCircuit."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure(0)
        job = MinimalBackend().run_async(qc, shots=64)
        assert job.result().num_circuits == 1


class TestAsyncExecution:
    """Integration tests for asynchronous execution via BackendTest."""

    def test_run_returns_before_job_is_done(self):
        """Ensure run() returns immediately before the async job has completed."""
        job = BackendTest(mode=ExecutionMode.ASYNC, async_delay=2.0, seed=0).run_async("c")
        assert not job.done()

    def test_status_is_queued_or_running_immediately_after_run(self):
        """Ensure the job status is QUEUED or RUNNING right after an async run() call."""
        job = BackendTest(mode=ExecutionMode.ASYNC, async_delay=2.0, seed=0).run_async("c")
        assert job.status() in (JobStatus.QUEUED, JobStatus.RUNNING)

    def test_result_blocks_and_returns(self):
        """Ensure result() blocks until the async job completes and returns valid counts."""
        backend = BackendTest(mode=ExecutionMode.ASYNC, num_qubits=2, async_delay=0.3, seed=0)
        result = backend.run_async("c", shots=256).result()
        assert result.num_circuits == 1
        assert sum(result.get_counts().values()) == 256

    def test_status_is_done_after_result(self):
        """Ensure the job status is DONE after result() has returned successfully."""
        backend = BackendTest(mode=ExecutionMode.ASYNC, async_delay=0.3, seed=0)
        job = backend.run_async("c")
        job.result()
        assert job.status() == JobStatus.DONE

    def test_async_batch_all_circuits_in_result(self):
        """Ensure an async batch result contains one counts dict per submitted circuit."""
        backend = BackendTest(mode=ExecutionMode.ASYNC, num_qubits=2, async_delay=0.3, seed=1)
        result = backend.run_async(["c0", "c1", "c2", "c3", "c4"], shots=100).result()
        assert result.num_circuits == 5
        for i in range(5):
            assert sum(result.get_counts(i).values()) == 100

    def test_metadata_contains_async_mode(self):
        """Ensure the result metadata carries the mode key set by the async worker."""
        result = BackendTest(mode=ExecutionMode.ASYNC, async_delay=0.3, seed=0).run_async("c").result()
        assert result.metadata.get("mode") == "async"

    def test_callback_fires_when_job_completes(self):
        """Ensure a registered callback is invoked with DONE status when the job completes."""
        backend = BackendTest(mode=ExecutionMode.ASYNC, async_delay=0.3, seed=0)
        job = backend.run_async("c")
        fired = []
        job.add_callback(lambda j: fired.append(j.status()))
        job.result()
        assert fired == [JobStatus.DONE]

    def test_callback_on_already_done_job_fires_immediately(self):
        """Ensure a callback registered on an already-done job fires synchronously."""
        job = BackendTest(mode=ExecutionMode.SYNC, seed=0).run_async("c")
        fired = []
        job.add_callback(lambda j: fired.append(True))
        assert len(fired) == 1

    def test_multiple_callbacks_all_fire(self):
        """Ensure all registered callbacks are invoked when the job reaches a terminal state."""
        backend = BackendTest(mode=ExecutionMode.ASYNC, async_delay=0.3, seed=0)
        job = backend.run_async("c")
        results = []
        job.add_callback(lambda j: results.append("cb1"))
        job.add_callback(lambda j: results.append("cb2"))
        job.result()
        assert "cb1" in results
        assert "cb2" in results

    def test_misbehaving_callback_does_not_crash_job(self):
        """Ensure an exception raised inside a callback does not prevent result() from returning."""
        backend = BackendTest(mode=ExecutionMode.ASYNC, async_delay=0.3, seed=0)
        job = backend.run_async("c")
        job.add_callback(lambda j: 1 / 0)
        result = job.result()
        assert result is not None

    def test_multiple_concurrent_jobs(self):
        """Ensure multiple async jobs submitted concurrently all complete successfully."""
        backend = BackendTest(mode=ExecutionMode.ASYNC, num_qubits=2, async_delay=0.3, seed=0)
        jobs = [backend.run_async(f"c{i}", shots=100) for i in range(4)]
        results = [job.result() for job in jobs]
        for result in results:
            assert result.num_circuits == 1

    def test_concurrent_result_calls_return_same_data(self):
        """Multiple threads calling result() on the same job simultaneously must all succeed.

        This guards against implementations that clear or overwrite the stored
        result on first access, which would cause a race condition.
        """
        backend = BackendTest(mode=ExecutionMode.ASYNC, num_qubits=2, async_delay=0.3, seed=7)
        job = backend.run_async("c", shots=100)
        results = []
        errors = []

        def fetch():
            try:
                results.append(job.result())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=fetch) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Unexpected errors: {errors}"
        assert len(results) == 5
        first = results[0].all_counts
        assert all(r.all_counts == first for r in results)


class TestTimeoutBehaviour:
    """Tests for the timeout parameter of Job.result()."""

    def test_timeout_raises_timeout_error(self):
        """Ensure result(timeout=...) raises TimeoutError when the deadline expires."""
        job = BackendTest(mode=ExecutionMode.ASYNC, async_delay=5.0, seed=0).run_async("c")
        with pytest.raises(TimeoutError):
            job.result(timeout=0.1)

    def test_job_still_running_after_timeout(self):
        """Ensure the job continues executing after a TimeoutError from result()."""
        job = BackendTest(mode=ExecutionMode.ASYNC, async_delay=5.0, seed=0).run_async("c")
        try:
            job.result(timeout=0.1)
        except TimeoutError:
            pass
        assert not job.done()

    def test_result_accessible_after_timeout_once_complete(self):
        """Ensure result() can be called again successfully after a prior TimeoutError."""
        backend = BackendTest(mode=ExecutionMode.ASYNC, num_qubits=2, async_delay=0.4, seed=0)
        job = backend.run_async("c", shots=128)
        try:
            job.result(timeout=0.05)  # too short, will time out
        except TimeoutError:
            pass
        result = job.result()  # now wait properly
        assert result.num_circuits == 1


class TestErrorBehaviour:
    """Integration tests for the error execution mode of BackendTest."""

    def test_sync_error_job_status_is_error(self):
        """Ensure a synchronous error immediately sets the job status to ERROR."""
        job = BackendTest(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run_async("c")
        assert job.status() == JobStatus.ERROR

    def test_sync_error_result_raises_runtime_error(self):
        """Ensure result() raises RuntimeError for a synchronously failed job."""
        job = BackendTest(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run_async("c")
        with pytest.raises(RuntimeError):
            job.result()

    def test_async_error_job_not_done_immediately(self):
        """Ensure an async error job is not yet in a terminal state right after run()."""
        job = BackendTest(mode=ExecutionMode.ERROR, async_delay=0.5, seed=0).run_async("c")
        assert not job.done()

    def test_async_error_result_raises_runtime_error(self):
        """Ensure result() raises RuntimeError after an async error has been set."""
        job = BackendTest(mode=ExecutionMode.ERROR, async_delay=0.3, seed=0).run_async("c")
        with pytest.raises(RuntimeError):
            job.result()

    def test_async_error_status_is_error_after_result(self):
        """Ensure the job status is ERROR after result() has raised on an async failure."""
        job = BackendTest(mode=ExecutionMode.ERROR, async_delay=0.3, seed=0).run_async("c")
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
        job = BackendTest(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run_async("c")
        assert job.done() is False  # done() means success only
        assert job.in_final_state() is True  # in_final_state() covers ERROR too

    def test_error_job_cancel_returns_false(self):
        """Ensure cancel() returns False on a job that has already failed."""
        job = BackendTest(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run_async("c")
        assert job.cancel() is False

    def test_error_in_batch_affects_whole_job(self):
        """Ensure that a hardware fault in a batch job marks the entire job as ERROR."""
        job = BackendTest(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run_async(["c0", "c1", "c2"])
        assert job.status() == JobStatus.ERROR

    def test_callback_fires_on_error(self):
        """Ensure a registered callback is invoked with ERROR status when the job fails."""
        backend = BackendTest(mode=ExecutionMode.ERROR, async_delay=0.3, seed=0)
        job = backend.run_async("c")
        fired = []
        job.add_callback(lambda j: fired.append(j.status()))
        try:
            job.result()
        except RuntimeError:
            pass
        assert fired == [JobStatus.ERROR]

    def test_sync_error_result_raises_job_failure_error(self):
        """result() must raise JobFailureError (not just RuntimeError) for a failed job.

        Vendor code and framework internals that want to distinguish a hardware
        fault from a cancellation or timeout can catch JobFailureError specifically.
        JobFailureError is a RuntimeError subclass, so existing ``except RuntimeError``
        blocks continue to work.
        """
        job = BackendTest(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run_async("c")
        with pytest.raises(JobFailureError):
            job.result()

    def test_backend_run_raises_job_failure_error(self):
        """Backend.run() (the blocking convenience wrapper) must also surface JobFailureError.

        This verifies the full call-stack: run() → run_async() → job.result(),
        so that callers using the high-level API can still distinguish error types.
        """
        backend = BackendTest(mode=ExecutionMode.ERROR, async_delay=0, seed=0)
        with pytest.raises(JobFailureError):
            backend.run("c")


class TestCancellation:
    """Integration tests for job cancellation via BackendTest in CANCEL mode."""

    def test_cancel_returns_true_on_running_job(self):
        """Ensure cancel() returns True when called on a running job."""
        job = BackendTest(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run_async("c")
        time.sleep(0.6)
        assert job.cancel() is True

    def test_cancel_job_reaches_cancelled_status(self):
        """Ensure the job status reaches CANCELLED after a successful cancel() call."""
        job = BackendTest(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run_async("c")
        time.sleep(0.6)
        job.cancel()
        try:
            job.result()
        except RuntimeError:
            pass
        assert job.status() == JobStatus.CANCELLED

    def test_cancel_result_raises_runtime_error(self):
        """Ensure result() raises RuntimeError after the job has been cancelled."""
        job = BackendTest(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run_async("c")
        time.sleep(0.6)
        job.cancel()
        with pytest.raises(RuntimeError):
            job.result()

    def test_cancelled_job_is_in_final_state(self):
        """Ensure a cancelled job is in a final state (in_final_state() is True).

        Note that done() returns False because it means "completed successfully",
        not "reached any terminal state".
        """
        job = BackendTest(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run_async("c")
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
        job = BackendTest(mode=ExecutionMode.CANCEL, async_delay=0.3, seed=0).run_async("c")
        job.result()
        assert job.cancel() is False

    def test_cancel_returns_false_on_already_cancelled_job(self):
        """Ensure cancel() returns False when called a second time on an already-cancelled job."""
        job = BackendTest(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run_async("c")
        time.sleep(0.6)
        job.cancel()
        try:
            job.result()
        except RuntimeError:
            pass
        assert job.cancel() is False

    def test_natural_completion_without_cancellation(self):
        """Ensure a CANCEL-mode job completes normally if cancel() is never called."""
        backend = BackendTest(mode=ExecutionMode.CANCEL, num_qubits=2, async_delay=0.5, seed=3)
        result = backend.run_async("c", shots=128).result()
        assert sum(result.get_counts().values()) == 128

    def test_cancel_batch_job(self):
        """Ensure cancel() returns True when called on a running batch job."""
        job = BackendTest(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run_async(["c0", "c1", "c2", "c3", "c4"])
        time.sleep(0.6)
        assert job.cancel() is True

    def test_callback_fires_on_cancellation(self):
        """Ensure a registered callback is invoked with CANCELLED status after cancellation."""
        job = BackendTest(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run_async("c")
        fired = []
        job.add_callback(lambda j: fired.append(j.status()))
        time.sleep(0.6)
        job.cancel()
        try:
            job.result()
        except RuntimeError:
            pass
        assert fired == [JobStatus.CANCELLED]

    def test_cancel_result_raises_job_cancelled_error(self):
        """result() must raise JobCancelledError (not just RuntimeError) for a cancelled job.

        This lets callers distinguish a user-initiated cancellation from a hardware
        fault. JobCancelledError is a RuntimeError subclass, so existing
        ``except RuntimeError`` blocks continue to work.
        """
        job = BackendTest(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run_async("c")
        time.sleep(0.6)
        job.cancel()
        with pytest.raises(JobCancelledError):
            job.result()


def test_virtual_backend_deprecation_warning():
    """Ensure VirtualBackend raises a QrispDeprecationWarning upon instantiation."""

    def dummy_run():
        """No-op run function passed to VirtualBackend to satisfy its constructor."""

    with pytest.warns(QrispDeprecationWarning):
        _ = VirtualBackend(run_func=dummy_run)


class JobTestLifecycle:
    """Tests that the job lifecycle contract is honoured.

    The only hard guarantee is that a job must exit INITIALIZING before
    run_async() returns. The exact state after that depends on the backend:
    asynchronous backends transition to QUEUED. Synchronous simulators may
    go directly to RUNNING or DONE.
    """

    def test_job_starts_in_initializing_state(self):
        """A newly constructed job must start in INITIALIZING state."""
        backend = MinimalBackend()
        job = MinimalJob(backend=backend)
        assert job.status() == JobStatus.INITIALIZING

    def test_last_known_status_starts_as_initializing(self):
        """last_known_status must mirror INITIALIZING right after construction."""
        backend = MinimalBackend()
        job = MinimalJob(backend=backend)
        assert job.last_known_status == JobStatus.INITIALIZING

    def test_submit_transitions_to_queued(self):
        """MinimalJob.submit() transitions to QUEUED (the async-backend pattern).

        Note: this tests a specific implementation. Synchronous simulators are
        allowed to transition to RUNNING or DONE instead.
        """
        backend = MinimalBackend()
        job = MinimalJob(backend=backend)
        job.submit()
        assert job.status() == JobStatus.QUEUED

    def test_run_async_does_not_leave_job_in_initializing(self):
        """After run_async() returns, the job must no longer be in INITIALIZING state."""
        backend = MinimalBackend()
        job = backend.run_async("c")
        assert job.status() != JobStatus.INITIALIZING

    def test_sync_job_reaches_done_after_run_async(self):
        """A synchronous backend must fully resolve the job before run_async() returns."""
        backend = MinimalBackend()
        job = backend.run_async("c")
        assert job.status() == JobStatus.DONE

    def test_full_lifecycle_order_via_set_status(self):
        """States must progress in the documented order (no skipping QUEUED)."""
        backend = MinimalBackend()
        job = MinimalJob(backend=backend)
        assert job.status() == JobStatus.INITIALIZING

        job.submit()
        assert job.status() == JobStatus.QUEUED

        job._set_status(JobStatus.RUNNING)
        assert job.status() == JobStatus.RUNNING

        job._resolve(JobResult([{"0": 1}]))
        assert job.status() == JobStatus.DONE


class TestRefreshAndLastKnownStatus:
    """Tests for refresh() and the cached last_known_status property."""

    def test_last_known_status_updated_by_status_call(self):
        """Calling status() must update last_known_status as a side effect."""
        backend = MinimalBackend()
        job = MinimalJob(backend=backend)
        job._set_status(JobStatus.RUNNING)
        returned = job.status()
        assert returned == JobStatus.RUNNING
        assert job.last_known_status == JobStatus.RUNNING

    def test_refresh_updates_last_known_status(self):
        """refresh() must update last_known_status and return the new value."""
        backend = MinimalBackend()
        job = backend.run_async("c")  # job is DONE after MinimalBackend.run_async
        returned = job.refresh()
        assert returned == JobStatus.DONE
        assert job.last_known_status == JobStatus.DONE

    def test_refresh_return_equals_last_known_status(self):
        """The value returned by refresh() must equal last_known_status afterwards."""
        backend = MinimalBackend()
        job = backend.run_async("c")
        returned = job.refresh()
        assert returned == job.last_known_status

    def test_last_known_status_updates_on_state_transitions(self):
        """last_known_status always reflects the most recent state change."""
        backend = MinimalBackend()
        job = MinimalJob(backend=backend)
        assert job.last_known_status == JobStatus.INITIALIZING
        job.submit()
        assert job.last_known_status == JobStatus.QUEUED
        job._set_status(JobStatus.RUNNING)
        assert job.last_known_status == JobStatus.RUNNING
        job._resolve(JobResult([{"0": 10}]))
        assert job.last_known_status == JobStatus.DONE


class JobTestRepr:
    """Tests that Job.__repr__ is side-effect free (no live status() call)."""

    def test_repr_does_not_call_status(self):
        """repr(job) must not invoke status() — it must read last_known_status instead."""
        backend = MinimalBackend()
        job = backend.run_async("c")
        with unittest.mock.patch.object(type(job), "status", wraps=job.status) as mock_status:
            _ = repr(job)
            mock_status.assert_not_called()

    def test_repr_reflects_current_last_known_status(self):
        """repr(job) always reflects the current last_known_status."""
        backend = MinimalBackend()
        job = MinimalJob(backend=backend)
        job.submit()
        job._set_status(JobStatus.RUNNING)
        job._resolve(JobResult([{"0": 1}]))
        # last_known_status is DONE after _resolve — repr shows it immediately
        assert "done" in repr(job).lower()

    def test_repr_after_refresh_shows_current_status(self):
        """After refresh(), repr(job) must show the updated cached status."""
        backend = MinimalBackend()
        job = backend.run_async("c")
        job.refresh()
        assert "done" in repr(job).lower()


class TestShotsValidation:
    """Tests that Backend.run() rejects invalid shot counts early."""

    @pytest.fixture
    def backend(self):
        """Return a fresh MinimalBackend for each test."""
        return MinimalBackend()

    def test_shots_zero_raises_value_error(self, backend):
        """shots=0 must raise ValueError."""
        with pytest.raises(ValueError, match="shots"):
            backend.run(object(), shots=0)

    def test_shots_negative_raises_value_error(self, backend):
        """shots=-1 must raise ValueError."""
        with pytest.raises(ValueError, match="shots"):
            backend.run(object(), shots=-1)

    def test_shots_float_raises_type_error(self, backend):
        """shots=10.5 (a float) must raise TypeError."""
        with pytest.raises(TypeError, match="shots"):
            backend.run(object(), shots=10.5)

    def test_shots_string_raises_type_error(self, backend):
        """shots='100' (a string) must raise TypeError."""
        with pytest.raises(TypeError, match="shots"):
            backend.run(object(), shots="100")

    def test_shots_bool_raises_type_error(self, backend):
        """shots=True (a bool) must raise TypeError even though bool is a subclass of int."""
        with pytest.raises(TypeError, match="shots"):
            backend.run(object(), shots=True)

    def test_shots_none_is_accepted(self, backend):
        """shots=None must not raise — it means 'use the backend default'."""
        # "c" is not a QuantumCircuit so run() treats it as a batch and returns a list;
        # what matters here is that _validate_shots(None) does not raise.
        result = backend.run("c", shots=None)
        assert result is not None

    def test_shots_positive_integer_is_accepted(self, backend):
        """A positive integer shot count must be accepted without error."""
        result = backend.run("c", shots=128)
        assert result is not None

    def test_validate_shots_static_method_zero(self):
        """Backend._validate_shots(0) must raise ValueError."""
        with pytest.raises(ValueError):
            Backend._validate_shots(0)

    def test_validate_shots_static_method_negative(self):
        """Backend._validate_shots(-5) must raise ValueError."""
        with pytest.raises(ValueError):
            Backend._validate_shots(-5)

    def test_validate_shots_static_method_none_accepted(self):
        """Backend._validate_shots(None) must not raise."""
        Backend._validate_shots(None)  # no exception

    def test_validate_shots_static_method_positive_accepted(self):
        """Backend._validate_shots(512) must not raise."""
        Backend._validate_shots(512)  # no exception


class BackendTestOptionsExtended:
    """Additional options tests: atomic update and read-only proxy."""

    def test_update_options_is_atomic_valid_key_not_changed_on_failure(self):
        """update_options must not apply any change if any key in the call is invalid."""

        class MultiBackend(Backend):
            @classmethod
            def _default_options(cls):
                return {"shots": 1024, "level": 1}

            def run_async(self, circuits, shots=None):
                pass

        b = MultiBackend()
        original_shots = b.options["shots"]
        with pytest.raises(AttributeError, match="not a valid backend option"):
            b.update_options(shots=9999, totally_unknown=42)
        # shots must be unchanged — the whole call must be rolled back
        assert b.options["shots"] == original_shots

    def test_options_property_is_read_only(self):
        """Direct mutation of backend.options must raise TypeError (MappingProxyType)."""
        b = MinimalBackend()
        with pytest.raises(TypeError):
            b.options["shots"] = 9999

    def test_options_property_does_not_allow_new_keys(self):
        """Injecting a new key via backend.options must raise TypeError."""
        b = MinimalBackend()
        with pytest.raises(TypeError):
            b.options["injected"] = "value"

    def test_options_read_access_still_works(self):
        """Read access to backend.options must still work after making it read-only."""
        b = MinimalBackend()
        assert b.options["shots"] == 1024

    def test_options_equality_with_plain_dict(self):
        """MappingProxyType must compare equal to an equivalent plain dict."""
        b = MinimalBackend()
        assert b.options == {"shots": 1024}


class _LimitedBackend(MinimalBackend):
    """MinimalBackend subclass that exposes a fixed max_circuits limit."""

    def __init__(self, limit: int):
        super().__init__()
        self._limit = limit

    @property
    def max_circuits(self) -> int:
        return self._limit


class TestMaxCircuits:
    """Tests for the max_circuits execution constraint on Backend."""

    def test_max_circuits_is_none_by_default(self):
        """The base Backend must return None for max_circuits (no limit)."""
        assert MinimalBackend().max_circuits is None

    def test_max_circuits_can_be_overridden(self):
        """A concrete backend must be able to expose a positive integer limit."""
        assert _LimitedBackend(5).max_circuits == 5

    def test_run_raises_when_batch_exceeds_limit(self):
        """run() must raise ValueError when more circuits than max_circuits are submitted."""
        b = _LimitedBackend(2)
        with pytest.raises(ValueError, match="circuit"):
            b.run(["a", "b", "c"])

    def test_run_accepts_batch_equal_to_limit(self):
        """run() must accept a batch whose size exactly equals max_circuits."""
        b = _LimitedBackend(3)
        result = b.run(["a", "b", "c"])
        assert len(result) == 3

    def test_run_accepts_batch_below_limit(self):
        """run() must accept a batch whose size is smaller than max_circuits."""
        b = _LimitedBackend(5)
        result = b.run(["a", "b"])
        assert len(result) == 2

    def test_single_circuit_never_rejected(self):
        """A list of exactly one circuit is always within any positive limit."""
        b = _LimitedBackend(1)
        result = b.run(["only_one"])
        assert len(result) == 1

    def test_error_message_contains_class_name(self):
        """The ValueError message must name the backend class to aid debugging."""
        b = _LimitedBackend(1)
        with pytest.raises(ValueError, match="_LimitedBackend"):
            b.run(["a", "b"])

    def test_error_message_mentions_limit(self):
        """The ValueError message must state the actual limit."""
        b = _LimitedBackend(2)
        with pytest.raises(ValueError, match="2"):
            b.run(["a", "b", "c"])

    def test_check_circuit_limit_helper_is_no_op_when_limit_is_none(self):
        """_check_circuit_limit must not raise when max_circuits is None."""
        b = MinimalBackend()
        b._check_circuit_limit(["a", "b", "c", "d", "e"])

    def test_check_circuit_limit_helper_raises_over_limit(self):
        """_check_circuit_limit must raise ValueError when the batch is too large."""
        b = _LimitedBackend(2)
        with pytest.raises(ValueError):
            b._check_circuit_limit(["a", "b", "c"])


class TestRetrieveJob:
    """Tests for Backend.retrieve_job()."""

    def test_retrieve_job_raises_not_implemented_by_default(self):
        """The base implementation must raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            MinimalBackend().retrieve_job("some-id")

    def test_retrieve_job_error_mentions_class_name(self):
        """The NotImplementedError message must name the backend class."""
        with pytest.raises(NotImplementedError, match="MinimalBackend"):
            MinimalBackend().retrieve_job("some-id")

    def test_retrieve_job_error_suggests_override(self):
        """The NotImplementedError message must hint that override is needed."""
        with pytest.raises(NotImplementedError, match="[Oo]verride"):
            MinimalBackend().retrieve_job("some-id")

    def test_retrieve_job_can_be_overridden(self):
        """A concrete backend must be able to override retrieve_job to return a job."""

        class RecoverableBackend(MinimalBackend):
            """Backend that can reconnect to jobs by ID."""

            def retrieve_job(self, job_id: str) -> MinimalJob:
                """Return a synthetic job stub with the requested ID."""
                job = MinimalJob(backend=self, job_id=job_id)
                return job

        b = RecoverableBackend()
        job = b.retrieve_job("job-42")
        assert job.job_id == "job-42"
        assert isinstance(job, MinimalJob)


class JobTestFailureCause:
    """Tests that JobFailureError preserves the original exception as __cause__."""

    def test_sync_failure_cause_is_chained(self):
        """The original hardware exception must be the __cause__ of JobFailureError."""
        job = BackendTest(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run_async("c")
        with pytest.raises(JobFailureError) as exc_info:
            job.result()
        assert exc_info.value.__cause__ is not None

    def test_sync_failure_cause_preserves_original_message(self):
        """The __cause__ must carry the original exception message."""
        job = BackendTest(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run_async("c")
        with pytest.raises(JobFailureError) as exc_info:
            job.result()
        assert "Simulated hardware fault" in str(exc_info.value.__cause__)

    def test_async_failure_cause_is_chained(self):
        """Cause chaining must work for async (threaded) failures too."""
        job = BackendTest(mode=ExecutionMode.ERROR, async_delay=0.3, seed=0).run_async("c")
        with pytest.raises(JobFailureError) as exc_info:
            job.result()
        assert exc_info.value.__cause__ is not None
        assert "Simulated hardware fault" in str(exc_info.value.__cause__)

    def test_failure_cause_is_none_when_not_set(self):
        """If _failure_cause is not set by the implementation, __cause__ is None."""
        backend = MinimalBackend()
        job = MinimalJob(backend=backend)
        job._set_status(JobStatus.ERROR)
        job._done_event.set()
        # _failure_cause was never set, so __cause__ should be None
        with pytest.raises(JobFailureError) as exc_info:
            job.result()
        assert exc_info.value.__cause__ is None

    def test_done_job_does_not_raise_on_raise_for_status(self):
        """_raise_for_status must be a no-op when the job completed successfully."""
        job = BackendTest(mode=ExecutionMode.SYNC, seed=0).run_async("c")
        # Should not raise
        job._raise_for_status(JobStatus.DONE)


class BackendTestLikeProtocol:
    """Tests for the BackendLike structural protocol.

    BackendLike is @runtime_checkable, so isinstance() checks must work without
    explicit inheritance. Both Backend subclasses and BatchedBackend must satisfy
    it (arbitrary objects must not).
    """

    def test_backend_subclass_satisfies_protocol(self):
        """Any concrete Backend subclass must satisfy BackendLike at runtime."""
        assert isinstance(MinimalBackend(), BackendLike)

    def test_batched_backend_satisfies_protocol(self):
        """BatchedBackend must satisfy BackendLike even though it does not inherit Backend."""
        batched = MinimalBackend().batched()
        assert isinstance(batched, BackendLike)

    def test_partial_implementation_missing_run_does_not_satisfy(self):
        """An object missing run() must not satisfy BackendLike."""

        class NoRun:
            name = "no-run"

            @property
            def options(self):
                return {}

            def update_options(self, **kwargs):
                pass

        assert not isinstance(NoRun(), BackendLike)

    def test_structural_match_satisfies_protocol(self):
        """A third-party class with all required members must satisfy BackendLike without inheritance."""

        class ThirdPartyBackend:
            @property
            def name(self):
                return "third-party"

            def run(self, circuits, shots=None):
                return {}

            @property
            def options(self):
                return {}

            def update_options(self, **kwargs):
                pass

        assert isinstance(ThirdPartyBackend(), BackendLike)

    def test_multiple_backend_instances_all_satisfy_protocol(self):
        """Every Backend subclass in the test suite must satisfy BackendLike."""
        for backend in [
            MinimalBackend(),
            DummyBackend(),
            BackendNoDefaultOptions(),
            BackendWithChildDefaultOptions(),
        ]:
            assert isinstance(backend, BackendLike), f"{type(backend).__name__} does not satisfy BackendLike"

    def test_batched_backend_is_not_backend_subclass(self):
        """BatchedBackend must NOT be a subclass of Backend — that is the LSP design decision."""
        batched = MinimalBackend().batched()
        assert not isinstance(batched, Backend)

    def test_batched_backend_update_options_delegates(self):
        """BatchedBackend.update_options must delegate to the wrapped backend."""
        backend = MinimalBackend()
        batched = backend.batched()
        batched.update_options(shots=512)
        assert backend.options["shots"] == 512
        assert batched.options["shots"] == 512


class BackendTestRepr:
    """Tests that Backend.__repr__ exposes useful information for debugging."""

    def test_repr_contains_class_name(self):
        """repr(backend) must include the concrete class name."""
        b = MinimalBackend()
        assert "MinimalBackend" in repr(b)

    def test_repr_contains_backend_name(self):
        """repr(backend) must include the backend's name attribute."""
        b = MinimalBackend(name="my_hw")
        assert "my_hw" in repr(b)

    def test_repr_contains_options_key(self):
        """repr(backend) must mention 'options' so users know options are configurable."""
        b = MinimalBackend()
        assert "options" in repr(b).lower()
