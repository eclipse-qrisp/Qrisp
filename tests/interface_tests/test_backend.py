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

"""Test suite for the Backend abstraction layer."""

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
    Concrete `Job` produced by `TestBackend`.

    Demonstrates how a backend author implements the Job contract using
    internal threading, without any threading in the abstract base class.

    Extra attributes (``circuits``, ``shots``) and internal helpers
    (``_set_result``, ``_set_error``, ``_set_cancelled``, ``add_callback``)
    are additions of this concrete class. That is, they are not part of the
    base interface.
    """

    def __init__(self, backend, circuits, shots, job_id=None):
        super().__init__(backend=backend, job_id=job_id)
        self.circuits = circuits
        self.shots = shots
        self._status = JobStatus.INITIALIZING
        self._result_data = None
        self._error = None
        self._done_event = threading.Event()
        self._cancel_requested = threading.Event()
        self._callbacks = []

    ##########################
    ### Abstract interface
    ##########################

    def submit(self) -> None:
        pass  # TestBackend triggers execution directly in run().

    def result(self, timeout=None) -> JobResult | None:
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
        if self.done():
            return False
        self._cancel_requested.set()
        return True

    def status(self) -> JobStatus:
        return self._status

    ###########################
    ### Convenience helpers
    ###########################

    def add_callback(self, fn) -> None:
        """Register a callback invoked when the job reaches a terminal state."""
        if self.done():
            fn(self)
        else:
            self._callbacks.append(fn)

    def _set_status(self, status: JobStatus) -> None:
        self._status = status

    def _set_result(self, result: JobResult) -> None:
        self._result_data = result
        self._status = JobStatus.DONE
        self._done_event.set()
        for cb in self._callbacks:
            try:
                cb(self)
            except Exception:
                pass

    def _set_error(self, error: Exception) -> None:
        self._error = error
        self._status = JobStatus.ERROR
        self._done_event.set()
        for cb in self._callbacks:
            try:
                cb(self)
            except Exception:
                pass

    def _set_cancelled(self) -> None:
        self._status = JobStatus.CANCELLED
        self._done_event.set()
        for cb in self._callbacks:
            try:
                cb(self)
            except Exception:
                pass


class TestBackend(Backend):
    """Full-featured fake backend used by the integration tests."""

    def __init__(
        self,
        mode: str = ExecutionMode.SYNC,
        num_qubits: int = 2,
        async_delay: float = 1.0,
        seed: int | None = None,
        name: str | None = None,
        options=None,
    ):
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
        return {"shots": 1024}

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def backend_info(self):
        return {"provider": "Qrisp Test Suite", "version": "0.1.0"}

    @property
    def gate_set(self):
        return ["H", "X", "Y", "Z", "CNOT", "CZ", "RZ", "RX", "RY", "T", "S"]

    def run(self, circuits, shots=None) -> TestJob:
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
        n = 2**self._num_qubits
        raw = self._rng.multinomial(shots, np.ones(n) / n)
        return {
            format(i, f"0{self._num_qubits}b"): int(c)
            for i, c in enumerate(raw)
            if c > 0
        }

    def _run_sync(self, job: TestJob) -> None:
        job._set_status(JobStatus.RUNNING)
        counts = [self._fake_counts(job.shots) for _ in job.circuits]
        job._set_result(JobResult(counts, metadata={"mode": "sync"}))

    def _run_async(self, job: TestJob) -> None:
        job._set_status(JobStatus.RUNNING)
        time.sleep(self.async_delay)
        counts = [self._fake_counts(job.shots) for _ in job.circuits]
        job._set_result(JobResult(counts, metadata={"mode": "async"}))

    def _run_error_sync(self, job: TestJob) -> None:
        job._set_status(JobStatus.RUNNING)
        job._set_error(RuntimeError("Simulated hardware fault."))

    def _run_error_async(self, job: TestJob) -> None:
        job._set_status(JobStatus.RUNNING)
        time.sleep(self.async_delay)
        job._set_error(RuntimeError("Simulated hardware fault."))

    def _run_cancellable(self, job: TestJob) -> None:
        step = 0.5
        total_steps = max(1, int(self.async_delay / step))
        job._set_status(JobStatus.RUNNING)
        for _ in range(total_steps):
            if job._cancel_requested.is_set():
                job._set_cancelled()
                return
            time.sleep(step)
        counts = [self._fake_counts(job.shots) for _ in job.circuits]
        job._set_result(JobResult(counts, metadata={"mode": "cancel"}))


# ===========================================================================
# Minimal dummy backends used only for option / name tests
# ===========================================================================


class DummyBackend(Backend):
    """Minimal subclass of Backend for testing."""

    @classmethod
    def _default_options(cls):
        return {"shots": 1000, "flag": False}

    def run(self, circuits, shots: int | None = None):
        return {"circuits": circuits, "provided_shots": shots, "options": self.options}


class BackendNoDefaultOptions(Backend):
    """Backend instantiated without options — must use Backend._default_options()."""

    def run(self, circuits, shots: int | None = None):
        pass


class BackendWithExplicitOptions(Backend):
    """Backend instantiated with explicit options — must override defaults entirely."""

    def __init__(self, options=None):
        super().__init__(options=options)

    def run(self, circuits, shots: int | None = None):
        pass


class BackendWithChildDefaultOptions(Backend):
    """Backend with a custom _default_options() at the child level."""

    @classmethod
    def _default_options(cls):
        return {"shots": 1024, "custom_default": 42}

    def run(self, circuits, shots: int | None = None):
        pass


# ===========================================================================
# Test suite
# ===========================================================================


class TestBackendAbstractInterface:
    """Tests that Backend enforces its abstract contract."""

    def test_backend_cannot_be_instantiated_directly(self):
        """Test that Backend is an abstract base class and cannot be instantiated."""
        with pytest.raises(TypeError):
            Backend()

    def test_run_is_abstract(self):
        """Test that Backend.run is an abstract method that must be implemented by subclasses."""
        assert getattr(Backend.run, "__isabstractmethod__", False)

    def test_concrete_subclass_can_be_instantiated(self):
        """Test that a concrete subclass of Backend can be instantiated."""
        assert MinimalBackend() is not None


class TestBackendName:
    """Tests for backend name handling."""

    def test_name_defaults_to_class_name(self):
        """Test that the default name of a backend is its class name."""
        assert MinimalBackend().name == "MinimalBackend"

    def test_name_can_be_set_explicitly(self):
        """Test that the name of a backend can be set explicitly."""
        assert MinimalBackend(name="my_backend").name == "my_backend"

    def test_name_is_mutable(self):
        """Test that the name of a backend can be changed after instantiation."""
        b = MinimalBackend(name="first")
        b.name = "second"
        assert b.name == "second"


class TestBackendOptions:
    """Tests for backend options handling."""

    def test_default_options_are_shots_1024(self):
        """Test that the default options of a backend are {'shots': 1024}."""
        assert MinimalBackend().options == {"shots": 1024}

    def test_child_default_options_override_base(self):
        """Test that child default options override base default options."""
        b = BackendWithChildDefaultOptions()
        assert b.options["shots"] == 1024
        assert b.options["custom_default"] == 42

    def test_explicit_options_override_defaults_entirely(self):
        """Test that providing explicit options in the constructor overrides defaults entirely."""
        b = BackendWithExplicitOptions(options={"shots": 2048, "extra": "value"})
        assert b.options == {"shots": 2048, "extra": "value"}

    def test_no_options_uses_default_options(self):
        """Test that if no options are provided, the backend uses the default options."""
        b = BackendNoDefaultOptions()
        assert b.options == {"shots": 1024}

    def test_options_must_be_mapping_string_rejected(self):
        """Test that providing a non-mapping (e.g. string) as options raises a TypeError."""
        with pytest.raises(TypeError, match="Mapping"):
            MinimalBackend(options="not_a_mapping")

    def test_options_must_be_mapping_list_rejected(self):
        """Test that providing a non-mapping (e.g. list) as options raises a TypeError."""
        with pytest.raises(TypeError):
            MinimalBackend(options=[1, 2, 3])

    def test_options_are_copied_from_constructor(self):
        """Mutating the original mapping after construction must not affect the backend."""
        original = {"shots": 1024, "flag": False}
        b = BackendWithExplicitOptions(options=original)
        original["flag"] = True
        original["shots"] = 9999
        assert b.options["flag"] is False
        assert b.options["shots"] == 1024

    def test_update_options_valid_key(self):
        """Test that update_options successfully updates a valid option key."""
        b = MinimalBackend()
        b.update_options(shots=4096)
        assert b.options["shots"] == 4096

    def test_update_options_multiple_valid_keys(self):
        """Test that update_options can update multiple valid option keys at once."""

        class MultiBackend(Backend):
            """Backend with multiple default options for testing update_options."""

            @classmethod
            def _default_options(cls):
                return {"shots": 1024, "flag": False, "level": 1}

            def run(self, circuits, shots=None):
                pass

        b = MultiBackend()
        b.update_options(shots=2048, flag=True, level=3)
        assert b.options["shots"] == 2048
        assert b.options["flag"] is True
        assert b.options["level"] == 3

    def test_update_options_invalid_key_raises_attribute_error(self):
        """Test that update_options raises an AttributeError for invalid keys."""
        b = MinimalBackend()
        with pytest.raises(AttributeError, match="not a valid backend option"):
            b.update_options(unknown_key=42)

    def test_update_options_invalid_key_does_not_add_new_key(self):
        """Test that update_options with an invalid key does not add that key to the options."""
        b = MinimalBackend()
        try:
            b.update_options(new_key="value")
        except AttributeError:
            pass
        assert "new_key" not in b.options

    def test_options_are_independent_between_instances(self):
        """Test that options are independent between different backend instances."""
        b1 = MinimalBackend()
        b2 = MinimalBackend()
        b1.update_options(shots=2048)
        assert b2.options["shots"] == 1024

    def test_backend_with_explicit_options_update(self):
        """Test that update_options works correctly on a backend instantiated with explicit options."""
        options = {"shots": 1024, "custom_option": "old_value"}
        b = BackendWithExplicitOptions(options=options)
        b.update_options(custom_option="new_value")
        assert b.options["custom_option"] == "new_value"
        with pytest.raises(AttributeError):
            b.update_options(unknown_field=123)

    def test_backend_with_child_defaults_update(self):
        """Test that update_options works correctly on a backend with child default options."""
        b = BackendWithChildDefaultOptions()
        b.update_options(shots=2048, custom_default="updated")
        assert b.options["shots"] == 2048
        assert b.options["custom_default"] == "updated"
        with pytest.raises(AttributeError):
            b.update_options(new_param=999)


class TestBackendHardwareMetadata:
    """Tests that all hardware metadata properties return None by default."""

    @pytest.fixture
    def backend(self):
        return MinimalBackend()

    def test_num_qubits_is_none(self, backend):
        """Test that num_qubits property returns None by default."""
        assert backend.num_qubits is None

    def test_connectivity_is_none(self, backend):
        """Test that connectivity property returns None by default."""
        assert backend.connectivity is None

    def test_gate_set_is_none(self, backend):
        """Test that gate_set property returns None by default."""
        assert backend.gate_set is None

    def test_error_rates_is_none(self, backend):
        """Test that error_rates property returns None by default."""
        assert backend.error_rates is None

    def test_backend_health_is_none(self, backend):
        """Test that backend_health property returns None by default."""
        assert backend.backend_health is None

    def test_backend_info_is_none(self, backend):
        """Test that backend_info property returns None by default."""
        assert backend.backend_info is None

    def test_backend_queue_is_none(self, backend):
        """Test that backend_queue property returns None by default."""
        assert backend.backend_queue is None

    def test_capabilities_is_empty_dict(self, backend):
        """Test that capabilities property returns an empty dict by default."""
        assert backend.capabilities == {}

    def test_num_qubits_can_be_overridden(self):
        """Test that num_qubits can be overridden by a concrete backend."""
        assert TestBackend(num_qubits=5).num_qubits == 5

    def test_backend_info_can_be_overridden(self):
        """Test that backend_info can be overridden by a concrete backend."""
        info = TestBackend().backend_info
        assert isinstance(info, dict)
        assert "provider" in info

    def test_gate_set_can_be_overridden(self):
        """Test that gate_set can be overridden by a concrete backend."""
        gs = TestBackend().gate_set
        assert isinstance(gs, list)
        assert len(gs) > 0


class TestDummyBackend:
    """Tests for the minimal DummyBackend helper."""

    def test_backend_is_abstract(self):
        """Test that the Backend class is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Backend()

    def test_dummy_backend_instantiation_with_name(self):
        """Test that DummyBackend can be instantiated with a custom name."""
        b = DummyBackend(name="test_backend")
        assert b.name == "test_backend"

    def test_dummy_backend_instantiation_without_name(self):
        """Test that DummyBackend defaults to the class name when no name is provided."""
        b = DummyBackend()
        assert b.name == "DummyBackend"

    def test_dummy_backend_run_returns_dict_with_correct_values(self):
        """Test that DummyBackend.run returns a dict containing the provided circuits, shots, and options."""
        b = DummyBackend()
        result = b.run(circuits=None, shots=1024)
        assert b.options == {"shots": 1000, "flag": False}
        assert result["provided_shots"] == 1024
        assert result["circuits"] is None
        assert result["options"]["shots"] == 1000
        assert result["options"]["flag"] is False


# ===========================================================================
# Integration tests: synchronous execution
# ===========================================================================


class TestSyncExecution:
    """Integration tests for synchronous (blocking) execution."""

    def test_single_circuit_job_is_done_before_run_returns(self):
        """Test that a single-circuit job is already done by the time run() returns in SYNC mode."""
        job = TestBackend(mode=ExecutionMode.SYNC, seed=0).run("c")
        assert job.status() == JobStatus.DONE

    def test_single_circuit_result_immediately_available(self):
        """Test that the result of a single-circuit job is immediately available after run() returns in SYNC mode."""
        backend = TestBackend(mode=ExecutionMode.SYNC, num_qubits=2, seed=0)
        result = backend.run("c", shots=256).result()
        assert result.num_circuits == 1
        assert sum(result.get_counts().values()) == 256

    def test_bitstring_width_matches_num_qubits(self):
        """Test that the bitstrings in the counts dict have the correct width matching num_qubits."""
        for n in [1, 2, 3, 4]:
            backend = TestBackend(mode=ExecutionMode.SYNC, num_qubits=n, seed=0)
            result = backend.run("c", shots=128).result()
            for bitstring in result.get_counts():
                assert len(bitstring) == n

    def test_batch_of_five_is_done_immediately(self):
        """Test that a batch job with five circuits is already done by the time run() returns in SYNC mode."""
        job = TestBackend(mode=ExecutionMode.SYNC, seed=1).run(
            ["c0", "c1", "c2", "c3", "c4"]
        )
        assert job.status() == JobStatus.DONE

    def test_batch_result_has_one_dict_per_circuit(self):
        """Test that the JobResult of a batch job contains one counts dict per circuit."""
        result = (
            TestBackend(mode=ExecutionMode.SYNC, seed=1)
            .run(["c0", "c1", "c2", "c3", "c4"], shots=100)
            .result()
        )
        assert result.num_circuits == 5

    def test_batch_each_counts_dict_sums_to_shots(self):
        """Test that the counts dict for each circuit in a batch job sums to the total number of shots."""
        backend = TestBackend(mode=ExecutionMode.SYNC, num_qubits=2, seed=2)
        result = backend.run(["c0", "c1", "c2"], shots=200).result()
        for i in range(3):
            assert sum(result.get_counts(i).values()) == 200

    def test_batch_index_out_of_range_raises(self):
        """Test that requesting counts for a circuit index that is out of range in a batch job raises an IndexError."""
        result = TestBackend(mode=ExecutionMode.SYNC, seed=0).run(["c0", "c1"]).result()
        with pytest.raises(IndexError):
            result.get_counts(2)

    def test_single_circuit_normalised_to_one_result(self):
        """Test that running a single circuit (not in a list) is normalized to a batch of one circuit in the result."""
        result = (
            TestBackend(mode=ExecutionMode.SYNC, seed=0)
            .run("single", shots=512)
            .result()
        )
        assert result.num_circuits == 1

    def test_shots_override_respected(self):
        """Test that providing a shots argument to run() overrides the default shots option."""
        backend = TestBackend(mode=ExecutionMode.SYNC, num_qubits=2, seed=0)
        result = backend.run("c", shots=333).result()
        assert sum(result.get_counts().values()) == 333

    def test_default_shots_used_when_none(self):
        """Test that the default shots option is used when no shots argument is provided."""
        backend = TestBackend(mode=ExecutionMode.SYNC, num_qubits=2, seed=0)
        backend.update_options(shots=128)
        result = backend.run("c").result()
        assert sum(result.get_counts().values()) == 128

    def test_done_and_in_final_state_true_after_sync(self):
        """Test that done() and in_final_state() both return True immediately after run() in SYNC mode."""
        job = TestBackend(mode=ExecutionMode.SYNC, seed=0).run("c")
        assert job.done() is True
        assert job.in_final_state() is True

    def test_metadata_contains_sync_mode(self):
        """Test that the metadata of the JobResult contains 'mode': 'sync' for a SYNC job."""
        result = TestBackend(mode=ExecutionMode.SYNC, seed=0).run("c").result()
        assert result.metadata.get("mode") == "sync"

    def test_different_seeds_give_different_counts(self):
        """Test that using different seeds for the random number generator produces different counts distributions."""
        counts_a = (
            TestBackend(mode=ExecutionMode.SYNC, num_qubits=3, seed=0)
            .run("c", shots=1024)
            .result()
            .get_counts()
        )
        counts_b = (
            TestBackend(mode=ExecutionMode.SYNC, num_qubits=3, seed=99)
            .run("c", shots=1024)
            .result()
            .get_counts()
        )
        assert counts_a != counts_b

    def test_same_seed_gives_same_counts(self):
        """Test that using the same seed for the random number generator produces the same counts distribution."""
        counts_a = (
            TestBackend(mode=ExecutionMode.SYNC, num_qubits=2, seed=42)
            .run("c", shots=512)
            .result()
            .get_counts()
        )
        counts_b = (
            TestBackend(mode=ExecutionMode.SYNC, num_qubits=2, seed=42)
            .run("c", shots=512)
            .result()
            .get_counts()
        )
        assert counts_a == counts_b

    def test_accepts_arbitrary_circuit_types(self):
        """Test that the backend accepts arbitrary circuit representations without error."""
        backend = TestBackend(mode=ExecutionMode.SYNC, seed=0)
        for circuit in ["string", 42, None, object(), {"name": "qft"}, 3.14]:
            assert backend.run(circuit, shots=10).done()

    def test_heterogeneous_batch_accepted(self):
        """Test that the backend accepts a batch of heterogeneous circuit representations."""
        backend = TestBackend(mode=ExecutionMode.SYNC, seed=0)
        batch = ["bell", {"depth": 3}, None, 42, object()]
        assert backend.run(batch, shots=64).result().num_circuits == 5

    def test_cancel_on_finished_job_returns_false(self):
        """Test that calling cancel() on a job that is already finished returns False."""
        job = TestBackend(mode=ExecutionMode.SYNC, seed=0).run("c")
        assert job.cancel() is False

    def test_cancel_on_finished_job_does_not_change_status(self):
        """Test that calling cancel() on a job that is already finished does not change its status."""
        job = TestBackend(mode=ExecutionMode.SYNC, seed=0).run("c")
        job.cancel()
        assert job.status() == JobStatus.DONE

    def test_result_still_accessible_after_late_cancel(self):
        """Test that calling cancel() on a job that is already finished does not prevent access to the result."""
        job = TestBackend(mode=ExecutionMode.SYNC, num_qubits=2, seed=0).run(
            "c", shots=256
        )
        job.cancel()
        assert job.result().num_circuits == 1

    def test_multiple_sequential_jobs_from_same_backend(self):
        """Test that multiple sequential jobs from the same backend produce correct results."""
        backend = TestBackend(mode=ExecutionMode.SYNC, num_qubits=2, seed=0)
        for _ in range(5):
            result = backend.run("c", shots=64).result()
            assert sum(result.get_counts().values()) == 64


# ===========================================================================
# Integration tests: asynchronous execution
# ===========================================================================


class TestAsyncExecution:
    """Integration tests for asynchronous execution."""

    def test_run_returns_before_job_is_done(self):
        """Test that run() returns immediately before the job is done in ASYNC mode."""
        job = TestBackend(mode=ExecutionMode.ASYNC, async_delay=2.0, seed=0).run("c")
        assert not job.done()

    def test_status_is_queued_or_running_immediately_after_run(self):
        """Test that the job status is either QUEUED or RUNNING immediately after run() in ASYNC mode."""
        job = TestBackend(mode=ExecutionMode.ASYNC, async_delay=2.0, seed=0).run("c")
        assert job.status() in (JobStatus.QUEUED, JobStatus.RUNNING)

    def test_result_blocks_and_returns(self):
        """Test that result() blocks until the job is done and then returns a JobResult in ASYNC mode."""
        backend = TestBackend(
            mode=ExecutionMode.ASYNC, num_qubits=2, async_delay=0.3, seed=0
        )
        result = backend.run("c", shots=256).result()
        assert result.num_circuits == 1
        assert sum(result.get_counts().values()) == 256

    def test_status_is_done_after_result(self):
        """Test that the job status is DONE after result() returns in ASYNC mode."""
        backend = TestBackend(mode=ExecutionMode.ASYNC, async_delay=0.3, seed=0)
        job = backend.run("c")
        job.result()
        assert job.status() == JobStatus.DONE

    def test_async_batch_all_circuits_in_result(self):
        """Test that the JobResult of a batch job contains all circuits in ASYNC mode."""
        backend = TestBackend(
            mode=ExecutionMode.ASYNC, num_qubits=2, async_delay=0.3, seed=1
        )
        result = backend.run(["c0", "c1", "c2", "c3", "c4"], shots=100).result()
        assert result.num_circuits == 5
        for i in range(5):
            assert sum(result.get_counts(i).values()) == 100

    def test_metadata_contains_async_mode(self):
        """Test that the metadata of the JobResult contains 'mode': 'async' for an ASYNC job."""
        result = (
            TestBackend(mode=ExecutionMode.ASYNC, async_delay=0.3, seed=0)
            .run("c")
            .result()
        )
        assert result.metadata.get("mode") == "async"

    def test_callback_fires_when_job_completes(self):
        """Test that a callback registered with add_callback() is fired when the job completes in ASYNC mode."""
        backend = TestBackend(mode=ExecutionMode.ASYNC, async_delay=0.3, seed=0)
        job = backend.run("c")
        fired = []
        job.add_callback(lambda j: fired.append(j.status()))
        job.result()
        assert fired == [JobStatus.DONE]

    def test_callback_on_already_done_job_fires_immediately(self):
        """Test that a callback registered with add_callback() on an already done job fires immediately."""
        job = TestBackend(mode=ExecutionMode.SYNC, seed=0).run("c")
        fired = []
        job.add_callback(lambda j: fired.append(True))
        assert len(fired) == 1

    def test_multiple_callbacks_all_fire(self):
        """Test that multiple callbacks registered with add_callback() all fire when the job completes in ASYNC mode."""
        backend = TestBackend(mode=ExecutionMode.ASYNC, async_delay=0.3, seed=0)
        job = backend.run("c")
        results = []
        job.add_callback(lambda j: results.append("cb1"))
        job.add_callback(lambda j: results.append("cb2"))
        job.result()
        assert "cb1" in results
        assert "cb2" in results

    def test_misbehaving_callback_does_not_crash_job(self):
        """Test that a callback that raises an exception does not crash the job or prevent other callbacks from firing in ASYNC mode."""
        backend = TestBackend(mode=ExecutionMode.ASYNC, async_delay=0.3, seed=0)
        job = backend.run("c")
        job.add_callback(lambda j: 1 / 0)
        result = job.result()
        assert result is not None

    def test_multiple_concurrent_jobs(self):
        """Test that multiple concurrent jobs from the same backend execute correctly in ASYNC mode."""
        backend = TestBackend(
            mode=ExecutionMode.ASYNC, num_qubits=2, async_delay=0.3, seed=0
        )
        jobs = [backend.run(f"c{i}", shots=100) for i in range(4)]
        results = [job.result() for job in jobs]
        for result in results:
            assert result.num_circuits == 1


# ===========================================================================
# Integration tests: timeout behaviour
# ===========================================================================


class TestTimeoutBehaviour:
    """Tests for result() timeout behaviour."""

    def test_timeout_raises_timeout_error(self):
        """Test that result() raises a TimeoutError if the job does not complete within the specified timeout."""
        job = TestBackend(mode=ExecutionMode.ASYNC, async_delay=5.0, seed=0).run("c")
        with pytest.raises(TimeoutError):
            job.result(timeout=0.1)

    def test_job_still_running_after_timeout(self):
        """Test that the job is still running and not marked as done if result() times out."""
        job = TestBackend(mode=ExecutionMode.ASYNC, async_delay=5.0, seed=0).run("c")
        try:
            job.result(timeout=0.1)
        except TimeoutError:
            pass
        assert not job.done()

    def test_result_accessible_after_timeout_once_complete(self):
        """Test that the result of a job is still accessible after a TimeoutError once the job eventually completes."""
        backend = TestBackend(
            mode=ExecutionMode.ASYNC, num_qubits=2, async_delay=0.4, seed=0
        )
        job = backend.run("c", shots=128)
        try:
            job.result(timeout=0.05)  # too short
        except TimeoutError:
            pass
        result = job.result()  # now wait properly
        assert result.num_circuits == 1


# ===========================================================================
# Integration tests: error behaviour
# ===========================================================================


class TestErrorBehaviour:
    """Integration tests for error mode."""

    def test_sync_error_job_status_is_error(self):
        """Test that a job run in ERROR mode with zero async_delay has status ERROR immediately after run()."""
        job = TestBackend(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run("c")
        assert job.status() == JobStatus.ERROR

    def test_sync_error_result_raises_runtime_error(self):
        """Test that calling result() on a job run in ERROR mode with zero async_delay raises the expected RuntimeError."""
        job = TestBackend(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run("c")
        with pytest.raises(RuntimeError):
            job.result()

    def test_async_error_job_not_done_immediately(self):
        """Test that a job run in ERROR mode with nonzero async_delay is not immediately marked as done after run()."""
        job = TestBackend(mode=ExecutionMode.ERROR, async_delay=0.5, seed=0).run("c")
        assert not job.done()

    def test_async_error_result_raises_runtime_error(self):
        """Test that calling result() on a job run in ERROR mode with nonzero async_delay raises the expected RuntimeError."""
        job = TestBackend(mode=ExecutionMode.ERROR, async_delay=0.3, seed=0).run("c")
        with pytest.raises(RuntimeError):
            job.result()

    def test_async_error_status_is_error_after_result(self):
        """Test that a job run in ERROR mode with nonzero async_delay has status ERROR after calling result()."""
        job = TestBackend(mode=ExecutionMode.ERROR, async_delay=0.3, seed=0).run("c")
        try:
            job.result()
        except RuntimeError:
            pass
        assert job.status() == JobStatus.ERROR

    def test_error_job_is_in_final_state(self):
        """Test that a job run in ERROR mode is in a final state (done() and in_final_state() both return True) after the error is set."""
        job = TestBackend(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run("c")
        assert job.done() is True
        assert job.in_final_state() is True

    def test_error_job_cancel_returns_false(self):
        """Test that calling cancel() on a job run in ERROR mode returns False and does not change the status."""
        job = TestBackend(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run("c")
        assert job.cancel() is False

    def test_error_in_batch_affects_whole_job(self):
        """Test that if a job is run in ERROR mode, the entire batch is marked as ERROR and not just individual circuits."""
        job = TestBackend(mode=ExecutionMode.ERROR, async_delay=0, seed=0).run(
            ["c0", "c1", "c2"]
        )
        assert job.status() == JobStatus.ERROR

    def test_callback_fires_on_error(self):
        """Test that a callback registered with add_callback() is fired when the job encounters an error in ERROR mode."""
        backend = TestBackend(mode=ExecutionMode.ERROR, async_delay=0.3, seed=0)
        job = backend.run("c")
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
    """Integration tests for job cancellation."""

    def test_cancel_returns_true_on_running_job(self):
        """Test that cancel() returns True when called on a job that is currently running."""
        job = TestBackend(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run("c")
        time.sleep(0.6)
        assert job.cancel() is True

    def test_cancel_job_reaches_cancelled_status(self):
        """Test that a job that is cancelled while running eventually reaches the CANCELLED status."""
        job = TestBackend(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run("c")
        time.sleep(0.6)
        job.cancel()
        try:
            job.result()
        except RuntimeError:
            pass
        assert job.status() == JobStatus.CANCELLED

    def test_cancel_result_raises_runtime_error(self):
        """Test that calling result() on a job that was cancelled while running raises a RuntimeError indicating cancellation."""
        job = TestBackend(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run("c")
        time.sleep(0.6)
        job.cancel()
        with pytest.raises(RuntimeError):
            job.result()

    def test_cancelled_job_is_in_final_state(self):
        """Test that a job that was cancelled while running is in a final state (done() and in_final_state() both return True) after cancellation."""
        job = TestBackend(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run("c")
        time.sleep(0.6)
        job.cancel()
        try:
            job.result()
        except RuntimeError:
            pass
        assert job.done() is True
        assert job.in_final_state() is True

    def test_cancel_returns_false_on_finished_job(self):
        """Test that cancel() returns False when called on a job that has already finished (either successfully or with an error)."""
        job = TestBackend(mode=ExecutionMode.CANCEL, async_delay=0.3, seed=0).run("c")
        job.result()
        assert job.cancel() is False

    def test_cancel_returns_false_on_already_cancelled_job(self):
        """Test that cancel() returns False when called on a job that has already been cancelled."""
        job = TestBackend(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run("c")
        time.sleep(0.6)
        job.cancel()
        try:
            job.result()
        except RuntimeError:
            pass
        assert job.cancel() is False

    def test_natural_completion_without_cancellation(self):
        """Test that if cancel() is not called, a job in CANCEL mode eventually completes successfully with the expected results."""
        backend = TestBackend(
            mode=ExecutionMode.CANCEL, num_qubits=2, async_delay=0.5, seed=3
        )
        result = backend.run("c", shots=128).result()
        assert sum(result.get_counts().values()) == 128

    def test_cancel_batch_job(self):
        """Test that cancel() can be called on a batch job that is currently running and returns True."""
        job = TestBackend(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run(
            ["c0", "c1", "c2", "c3", "c4"]
        )
        time.sleep(0.6)
        assert job.cancel() is True

    def test_callback_fires_on_cancellation(self):
        """Test that a callback registered with add_callback() is fired when the job is cancelled in CANCEL mode."""
        job = TestBackend(mode=ExecutionMode.CANCEL, async_delay=5.0, seed=0).run("c")
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
    """BackendClient must raise a deprecation warning upon instantiation."""
    with pytest.warns(QrispDeprecationWarning):
        _ = BackendClient(api_endpoint="not_used")


def test_backend_server_deprecation_warning():
    """BackendServer must raise a deprecation warning upon instantiation."""

    def dummy_run():
        pass

    with pytest.warns(QrispDeprecationWarning):
        _ = BackendServer(run_func=dummy_run)


def test_virtual_backend_deprecation_warning():
    """VirtualBackend must raise a deprecation warning upon instantiation."""

    def dummy_run():
        pass

    with pytest.warns(QrispDeprecationWarning):
        _ = VirtualBackend(run_func=dummy_run)
