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

"""Tests for the Job-related classes."""

from __future__ import annotations

import pytest
from conftest import MinimalBackend, MinimalJob

from qrisp.interface.job import JOB_FINAL_STATES, Job, JobResult, JobStatus


# TODO: add tests for `cancelled` new method in the `Job` class

# TODO: update tests to reflect the new behaviour of

# TODO: add test for `metadata` attribute in the `Job` class
# (should we combine them with the `metadata` tests in `JobResult` or keep them separate?)


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
            ([], 0),
        ],
    )
    def test_job_result_construction(self, counts, expected_num_circuits):
        """Test that JobResult is constructed correctly with various inputs."""
        r = JobResult(counts)
        assert r.num_circuits == expected_num_circuits

    def test_metadata_defaults_to_empty_dict(self):
        """Test that the metadata defaults to an empty dictionary."""
        r = JobResult([{"0": 1024}])
        assert r.metadata == {}

    def test_non_list_raises_type_error(self):
        """Test that passing a non-list to JobResult raises a TypeError."""
        with pytest.raises(TypeError, match="'counts' must be a list"):
            JobResult({"00": 512})

    def test_tuple_raises_type_error(self):
        """Test that passing a tuple to JobResult raises a TypeError."""
        with pytest.raises(TypeError):
            JobResult(({"00": 512},))

    def test_metadata_custom_value_preserved(self):
        """Test that custom metadata values are preserved."""

        meta = {"mode": "async", "backend": "test"}
        r = JobResult([{"0": 1024}], metadata=meta)
        assert r.metadata["mode"] == "async"
        assert r.metadata["backend"] == "test"


class TestJobResultAccess:
    """Unit tests for JobResult data access."""

    def test_counts_sum_preserved(self):
        """Test that the sum of counts is preserved in JobResult."""
        r = JobResult([{"00": 600, "01": 100, "10": 200, "11": 124}])
        assert sum(r.get_counts().values()) == 1024

    @pytest.fixture
    def single_result(self):
        """Fixture that provides a JobResult with a single circuit's counts."""
        return JobResult([{"00": 512, "11": 512}])

    @pytest.fixture
    def batch_result(self):
        """Fixture that provides a JobResult with multiple circuits' counts."""
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
        """Test that all_counts returns the full list of counts."""
        all_c = batch_result.all_counts
        assert len(all_c) == 3
        assert all_c[1] == {"00": 500, "11": 524}

    def test_num_circuits_matches_input(self, batch_result):
        """Test that num_circuits matches the number of counts dictionaries provided."""
        assert batch_result.num_circuits == 3


class TestJobAbstractInterface:
    """Tests that Job enforces its abstract contract."""

    def test_job_cannot_be_instantiated_directly(self):
        """Test that instantiating Job directly raises a TypeError."""
        with pytest.raises(TypeError):
            Job(backend=None)

    def test_submit_is_abstract(self):
        """Test that submit() is an abstract method."""
        assert getattr(Job.submit, "__isabstractmethod__", False)

    def test_result_is_abstract(self):
        """Test that result() is an abstract method."""
        assert getattr(Job.result, "__isabstractmethod__", False)

    def test_cancel_is_abstract(self):
        """Test that cancel() is an abstract method."""
        assert getattr(Job.cancel, "__isabstractmethod__", False)

    def test_status_is_abstract(self):
        """Test that status() is an abstract method."""
        assert getattr(Job.status, "__isabstractmethod__", False)

    def test_concrete_subclass_with_all_methods_can_be_instantiated(self):
        """Test that a concrete subclass of Job that implements all abstract methods can be instantiated."""
        job = MinimalJob(backend=MinimalBackend())
        assert job is not None

    def test_partial_subclass_missing_result_cannot_be_instantiated(self):
        """Test that a subclass of Job that does not implement result() cannot be instantiated."""

        class IncompleteJob(Job):
            """A Job subclass that does not implement result()."""

            def submit(self):
                pass

            def cancel(self):
                return False

            def status(self):
                return JobStatus.INITIALIZING

            # result() is missing

        with pytest.raises(TypeError):
            IncompleteJob(backend=None)


class TestJobConcreteHelpers:
    """
    Tests for the non-abstract helpers on Job.

    All helpers are derived from status() and contain no concurrency logic.
    """

    @pytest.fixture
    def backend(self):
        """Fixture that provides a minimal backend."""
        return MinimalBackend()

    def _job_with_status(self, backend, status):
        """Helper to create a MinimalJob with a specific status."""
        return MinimalJob(backend=backend, initial_status=status)

    @pytest.mark.parametrize(
        "status", [JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR]
    )
    def test_done_returns_true_for_terminal_states(self, backend, status):
        """Test that done() returns True for all terminal states."""
        assert self._job_with_status(backend, status).done() is True

    @pytest.mark.parametrize(
        "status", [JobStatus.INITIALIZING, JobStatus.QUEUED, JobStatus.RUNNING]
    )
    def test_done_returns_false_for_non_terminal_states(self, backend, status):
        """Test that done() returns False for all non-terminal states."""
        assert self._job_with_status(backend, status).done() is False

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

    @pytest.mark.parametrize("status", list(JobStatus))
    def test_in_final_state_equals_done_for_all_statuses(self, backend, status):
        """Test that in_final_state() returns the same value as done() for all statuses."""
        job = self._job_with_status(backend, status)
        assert job.in_final_state() == job.done()

    def test_job_id_none_by_default(self, backend):
        """Test that job_id is None by default."""
        assert MinimalJob(backend=backend).job_id is None

    def test_job_id_set_explicitly(self, backend):
        """Test that job_id can be set explicitly."""
        assert MinimalJob(backend=backend, job_id="abc-123").job_id == "abc-123"

    def test_backend_property_returns_creating_backend(self, backend):
        """Test that the backend property returns the backend passed to the constructor."""
        job = MinimalJob(backend=backend)
        assert job.backend is backend
