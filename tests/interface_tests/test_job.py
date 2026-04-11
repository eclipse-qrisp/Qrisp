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

"""Tests for the Job-related classes defined in qrisp.interface.job"""

from __future__ import annotations

import pytest
from conftest import MinimalBackend, MinimalJob

from qrisp.interface.job import (
    JOB_FINAL_STATES,
    Job,
    JobCancelledError,
    JobFailureError,
    JobResult,
    JobStatus,
)


class TestJobStatusEnum:
    """Unit tests for the JobStatus enumeration."""

    def test_all_six_states_exist(self):
        """Test that all expected JobStatus states are defined."""
        assert JobStatus.INITIALIZING
        assert JobStatus.QUEUED
        assert JobStatus.RUNNING
        assert JobStatus.DONE
        assert JobStatus.CANCELLED
        assert JobStatus.ERROR

    def test_all_states_are_distinct(self):
        """Test that all JobStatus states are unique."""
        states = list(JobStatus)
        assert len(states) == len(set(states))

    def test_final_states_are_done_cancelled_error(self):
        """Test that JOB_FINAL_STATES contains exactly DONE, CANCELLED, and ERROR."""
        assert JobStatus.DONE in JOB_FINAL_STATES
        assert JobStatus.CANCELLED in JOB_FINAL_STATES
        assert JobStatus.ERROR in JOB_FINAL_STATES

    def test_non_final_states_not_in_final_states(self):
        """Test that non-final states are not included in JOB_FINAL_STATES."""
        assert JobStatus.INITIALIZING not in JOB_FINAL_STATES
        assert JobStatus.QUEUED not in JOB_FINAL_STATES
        assert JobStatus.RUNNING not in JOB_FINAL_STATES

    def test_final_states_tuple_has_exactly_three_elements(self):
        """Test that JOB_FINAL_STATES contains exactly three states."""
        assert len(JOB_FINAL_STATES) == 3


class TestJobResultConstruction:
    """Unit tests for JobResult construction and validation."""

    @pytest.mark.parametrize(
        "counts,expected_num_circuits",
        [
            ([{"00": 512, "11": 512}], 1),
            ([{"0": 800, "1": 224}, {"00": 500, "11": 524}], 2),
        ],
    )
    def test_job_result_construction(self, counts, expected_num_circuits):
        """Test that JobResult is constructed correctly with various inputs."""
        r = JobResult(counts)
        assert r.num_circuits == expected_num_circuits

    def test_empty_counts_raises_value_error(self):
        """Test that an empty counts list raises a ValueError."""
        with pytest.raises(ValueError):
            JobResult([])

    def test_metadata_defaults_to_empty_dict(self):
        """Test that the metadata attribute defaults to an empty dictionary."""
        r = JobResult([{"0": 1024}])
        assert r.metadata == {}

    def test_non_list_raises_type_error(self):
        """Test that passing a dict directly to JobResult raises a TypeError."""
        with pytest.raises(TypeError):
            JobResult({"00": 512})

    def test_metadata_kwargs_stored_correctly(self):
        """Test that keyword arguments are stored in the metadata dict.

        JobResult accepts metadata as **kwargs, so JobResult(counts, mode="async")
        stores {"mode": "async"} in self.metadata — not {"metadata": {...}}.
        """
        r = JobResult([{"0": 1024}], mode="async", backend="test")
        assert r.metadata["mode"] == "async"
        assert r.metadata["backend"] == "test"

    def test_metadata_old_dict_style_is_nested(self):
        """Test that passing metadata=dict nests it under the key 'metadata'.

        This documents the expected behaviour after the **kwargs API change:
        callers that use the old metadata={"key": val} style will find their
        dict stored at r.metadata["metadata"], not at r.metadata["key"].
        """
        r = JobResult([{"0": 1024}], metadata={"mode": "sync"})
        assert "metadata" in r.metadata
        assert r.metadata["metadata"] == {"mode": "sync"}


class TestJobResultAccess:
    """Unit tests for JobResult data access."""

    @pytest.fixture
    def single_result(self):
        """Fixture providing a JobResult with a single circuit's counts."""
        return JobResult([{"00": 512, "11": 512}])

    @pytest.fixture
    def batch_result(self):
        """Fixture providing a JobResult with three circuits' counts."""
        return JobResult(
            [
                {"0": 800, "1": 224},
                {"00": 500, "11": 524},
                {"000": 200, "111": 824},
            ]
        )

    def test_get_counts_default_index(self, single_result):
        """Test that get_counts() with no index returns the first circuit's counts."""
        assert single_result.get_counts() == {"00": 512, "11": 512}

    def test_get_counts_explicit_zero(self, single_result):
        """Test that get_counts(0) returns the first circuit's counts."""
        assert single_result.get_counts(0) == {"00": 512, "11": 512}

    def test_get_counts_negative_valid_index(self, single_result):
        """Test that get_counts(-1) returns the last circuit's counts."""
        assert single_result.get_counts(-1) == {"00": 512, "11": 512}

    def test_get_counts_out_of_range_raises_index_error(self, single_result):
        """Test that get_counts() raises IndexError for an out-of-range index."""
        with pytest.raises(IndexError, match="index 1 is out of range"):
            single_result.get_counts(1)

    def test_get_counts_batch_default_index(self, batch_result):
        """Test that get_counts() with no index returns the first circuit's counts in a batch."""
        assert batch_result.get_counts() == {"0": 800, "1": 224}

    def test_get_counts_batch_index_1(self, batch_result):
        """Test that get_counts(1) returns the second circuit's counts in a batch."""
        assert batch_result.get_counts(1) == {"00": 500, "11": 524}

    def test_get_counts_batch_index_2(self, batch_result):
        """Test that get_counts(2) returns the third circuit's counts in a batch."""
        assert batch_result.get_counts(2) == {"000": 200, "111": 824}

    def test_all_counts_returns_full_list(self, batch_result):
        """Test that all_counts returns the complete list of counts dictionaries."""
        all_c = batch_result.all_counts
        assert len(all_c) == 3
        assert all_c[1] == {"00": 500, "11": 524}

    def test_num_circuits_matches_input(self, batch_result):
        """Test that num_circuits equals the number of counts dictionaries provided."""
        assert batch_result.num_circuits == 3

    def test_counts_sum_preserved(self):
        """Test that the total count sum equals the number of shots."""
        r = JobResult([{"00": 600, "01": 100, "10": 200, "11": 124}])
        assert sum(r.get_counts().values()) == 1024


class TestJobAbstractInterface:
    """Tests that Job enforces its abstract contract."""

    def test_job_cannot_be_instantiated_directly(self):
        """Test that instantiating Job directly raises a TypeError."""
        with pytest.raises(TypeError):
            Job(backend=None)

    def test_submit_is_abstract(self):
        """Test that submit() is declared as an abstract method."""
        assert getattr(Job.submit, "__isabstractmethod__", False)

    def test_result_is_abstract(self):
        """Test that result() is declared as an abstract method."""
        assert getattr(Job.result, "__isabstractmethod__", False)

    def test_cancel_is_abstract(self):
        """Test that cancel() is declared as an abstract method."""
        assert getattr(Job.cancel, "__isabstractmethod__", False)

    def test_status_is_abstract(self):
        """Test that status() is declared as an abstract method."""
        assert getattr(Job.status, "__isabstractmethod__", False)

    def test_concrete_subclass_with_all_methods_can_be_instantiated(self):
        """Test that a concrete subclass implementing all abstract methods can be instantiated."""
        job = MinimalJob(backend=MinimalBackend())
        assert job is not None

    def test_partial_subclass_missing_status_cannot_be_instantiated(self):
        """Test that a subclass missing status() cannot be instantiated."""

        class IncompleteJob(Job):
            """A Job subclass missing status()."""

            def submit(self):
                """No-op submit."""

            def result(self):
                """No-op result."""

            def cancel(self):
                """No-op cancel."""
                return False

            # status() is missing

        with pytest.raises(TypeError):
            IncompleteJob(backend=None)

    def test_partial_subclass_missing_result_cannot_be_instantiated(self):
        """Test that a subclass missing result() cannot be instantiated."""

        class IncompleteJob(Job):
            """A Job subclass missing result()."""

            def submit(self):
                """No-op submit."""

            def cancel(self):
                """No-op cancel."""
                return False

            def status(self):
                """Return a fixed status."""
                return JobStatus.INITIALIZING

            # result() is missing

        with pytest.raises(TypeError):
            IncompleteJob(backend=None)

    def test_job_metadata_attribute_exists(self):
        """Test that a Job instance exposes a metadata attribute.

        Metadata is passed as **kwargs to the Job constructor and stored
        for backend-specific use.
        """
        backend = MinimalBackend()
        job = MinimalJob(backend=backend)
        assert hasattr(job, "metadata")

    def test_job_metadata_stores_kwargs(self):
        """Test that **kwargs passed to the Job constructor are accessible via metadata."""
        backend = MinimalBackend()
        job = MinimalJob(backend=backend, source="test_suite", run_id=42)
        assert job.metadata.get("source") == "test_suite"
        assert job.metadata.get("run_id") == 42


class TestJobConcreteHelpers:
    """
    Tests for the non-abstract helpers on Job.

    All helpers are derived from status() and contain no concurrency logic.
    """

    @pytest.fixture
    def backend(self):
        """Fixture providing a MinimalBackend instance."""
        return MinimalBackend()

    def _job_with_status(self, backend, status):
        """Helper to create a MinimalJob with a specific status."""
        return MinimalJob(backend=backend, initial_status=status)

    def test_done_returns_true_only_for_done_status(self, backend):
        """Test that done() returns True only for JobStatus.DONE."""
        assert self._job_with_status(backend, JobStatus.DONE).done() is True

    def test_done_returns_false_for_cancelled_and_error(self, backend):
        """Test that done() returns False for CANCELLED and ERROR.

        done() means 'completed successfully' — it is not a synonym for
        in_final_state(). Use in_final_state() to test for any terminal state.
        """
        assert self._job_with_status(backend, JobStatus.CANCELLED).done() is False
        assert self._job_with_status(backend, JobStatus.ERROR).done() is False

    @pytest.mark.parametrize(
        "status", [JobStatus.INITIALIZING, JobStatus.QUEUED, JobStatus.RUNNING]
    )
    def test_done_returns_false_for_non_terminal_states(self, backend, status):
        """Test that done() returns False for all non-terminal states."""
        assert self._job_with_status(backend, status).done() is False

    @pytest.mark.parametrize(
        "status", [JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR]
    )
    def test_in_final_state_returns_true_for_all_terminal_states(self, backend, status):
        """Test that in_final_state() returns True for all three terminal states."""
        assert self._job_with_status(backend, status).in_final_state() is True

    @pytest.mark.parametrize(
        "status", [JobStatus.INITIALIZING, JobStatus.QUEUED, JobStatus.RUNNING]
    )
    def test_in_final_state_returns_false_for_non_terminal_states(
        self, backend, status
    ):
        """Test that in_final_state() returns False for all non-terminal states."""
        assert self._job_with_status(backend, status).in_final_state() is False

    def test_running_true_only_for_running(self, backend):
        """Test that running() returns True only for the RUNNING state."""
        assert self._job_with_status(backend, JobStatus.RUNNING).running() is True
        assert self._job_with_status(backend, JobStatus.QUEUED).running() is False
        assert self._job_with_status(backend, JobStatus.DONE).running() is False

    def test_queued_true_only_for_queued(self, backend):
        """Test that queued() returns True only for the QUEUED state."""
        assert self._job_with_status(backend, JobStatus.QUEUED).queued() is True
        assert self._job_with_status(backend, JobStatus.RUNNING).queued() is False
        assert self._job_with_status(backend, JobStatus.DONE).queued() is False

    def test_cancelled_true_only_for_cancelled(self, backend):
        """Test that cancelled() returns True only for the CANCELLED state."""
        assert self._job_with_status(backend, JobStatus.CANCELLED).cancelled() is True
        assert self._job_with_status(backend, JobStatus.DONE).cancelled() is False
        assert self._job_with_status(backend, JobStatus.ERROR).cancelled() is False
        assert self._job_with_status(backend, JobStatus.RUNNING).cancelled() is False

    def test_job_id_none_by_default(self, backend):
        """Test that job_id is None when not explicitly provided."""
        assert MinimalJob(backend=backend).job_id is None

    def test_job_id_set_explicitly(self, backend):
        """Test that an explicitly provided job_id is stored correctly."""
        assert MinimalJob(backend=backend, job_id="abc-123").job_id == "abc-123"

    def test_backend_property_returns_creating_backend(self, backend):
        """Test that the backend property returns the backend passed at construction."""
        job = MinimalJob(backend=backend)
        assert job.backend is backend


class TestJobResultContract:
    """Tests verifying the result() blocking contract and terminal-state exceptions."""

    @pytest.fixture
    def backend(self):
        """Fixture providing a MinimalBackend instance."""
        return MinimalBackend()

    def test_result_returns_job_result_on_success(self, backend):
        """result() returns the JobResult when the job completed successfully."""
        job = MinimalJob(backend=backend)
        expected = JobResult([{"0": 1024}])
        job._resolve(expected)
        assert job.result(timeout=1) is expected

    def test_result_raises_job_failure_error_on_error_state(self, backend):
        """result() raises JobFailureError when the job terminated in ERROR state."""
        job = MinimalJob(backend=backend)
        job._fail(RuntimeError("hardware error"))
        with pytest.raises(JobFailureError):
            job.result(timeout=1)

    def test_result_raises_job_cancelled_error_on_cancelled_state(self, backend):
        """result() raises JobCancelledError when the job was cancelled."""
        job = MinimalJob(backend=backend)
        job.cancel()
        with pytest.raises(JobCancelledError):
            job.result(timeout=1)

    def test_result_raises_immediately_if_already_in_error_state(self, backend):
        """result() raises immediately without blocking if the job is already in ERROR."""
        job = MinimalJob(backend=backend)
        job._fail(RuntimeError("pre-existing failure"))
        # _fail() sets the done event, so result() must not block
        with pytest.raises(JobFailureError):
            job.result(timeout=0)

    def test_result_raises_immediately_if_already_cancelled(self, backend):
        """result() raises immediately without blocking if the job is already CANCELLED."""
        job = MinimalJob(backend=backend)
        job.cancel()
        # cancel() sets the done event, so result() must not block
        with pytest.raises(JobCancelledError):
            job.result(timeout=0)

    def test_job_failure_error_is_runtime_error(self, backend):
        """JobFailureError is a RuntimeError subclass, so existing bare-except callers still work."""
        job = MinimalJob(backend=backend)
        job._fail(RuntimeError("oops"))
        with pytest.raises(RuntimeError):
            job.result(timeout=1)

    def test_job_cancelled_error_is_runtime_error(self, backend):
        """JobCancelledError is a RuntimeError subclass, so existing bare-except callers still work."""
        job = MinimalJob(backend=backend)
        job.cancel()
        with pytest.raises(RuntimeError):
            job.result(timeout=1)

    def test_job_failure_error_message_contains_job_id(self, backend):
        """JobFailureError message includes the job_id to aid debugging."""
        job = MinimalJob(backend=backend, job_id="test-job-42")
        job._fail(RuntimeError("bad calibration"))
        with pytest.raises(JobFailureError, match="test-job-42"):
            job.result(timeout=1)

    def test_job_cancelled_error_message_contains_job_id(self, backend):
        """JobCancelledError message includes the job_id to aid debugging."""
        job = MinimalJob(backend=backend, job_id="test-job-99")
        job.cancel()
        with pytest.raises(JobCancelledError, match="test-job-99"):
            job.result(timeout=1)


class TestJobStatusCaching:
    """Tests for last_known_status and refresh()."""

    @pytest.fixture
    def backend(self):
        """Fixture providing a MinimalBackend instance."""
        return MinimalBackend()

    def test_last_known_status_initialises_to_initializing(self, backend):
        """last_known_status is INITIALIZING before any refresh() call."""
        job = MinimalJob(backend=backend)
        assert job.last_known_status == JobStatus.INITIALIZING

    def test_refresh_updates_last_known_status(self, backend):
        """refresh() updates last_known_status to reflect the current status."""
        job = MinimalJob(backend=backend)
        job._set_status(JobStatus.QUEUED)
        job.refresh()
        assert job.last_known_status == JobStatus.QUEUED

    def test_refresh_returns_current_status(self, backend):
        """refresh() returns the freshly fetched status."""
        job = MinimalJob(backend=backend)
        job._set_status(JobStatus.RUNNING)
        returned = job.refresh()
        assert returned == JobStatus.RUNNING

    def test_refresh_tracks_status_changes_across_calls(self, backend):
        """Successive refresh() calls track status transitions correctly."""
        job = MinimalJob(backend=backend)
        job._set_status(JobStatus.QUEUED)
        job.refresh()
        assert job.last_known_status == JobStatus.QUEUED

        job._set_status(JobStatus.RUNNING)
        job.refresh()
        assert job.last_known_status == JobStatus.RUNNING

        job._resolve(JobResult([{"0": 1024}]))
        job.refresh()
        assert job.last_known_status == JobStatus.DONE

    def test_status_call_does_not_update_last_known_status(self, backend):
        """Calling status() directly does not mutate last_known_status."""
        job = MinimalJob(backend=backend)
        job._set_status(JobStatus.RUNNING)
        job.status()  # live query — must not update the cache
        assert job.last_known_status == JobStatus.INITIALIZING

    def test_last_known_status_reflects_final_state_after_refresh(self, backend):
        """last_known_status correctly caches terminal states."""
        job = MinimalJob(backend=backend)
        job._fail(RuntimeError("error"))
        job.refresh()
        assert job.last_known_status == JobStatus.ERROR

    def test_last_known_status_is_independent_across_job_instances(self, backend):
        """Each Job instance has its own independent last_known_status cache."""
        job_a = MinimalJob(backend=backend)
        job_b = MinimalJob(backend=backend)
        job_a._set_status(JobStatus.RUNNING)
        job_a.refresh()
        assert job_a.last_known_status == JobStatus.RUNNING
        assert job_b.last_known_status == JobStatus.INITIALIZING
