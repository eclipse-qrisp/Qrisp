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

"""Tests for DefaultBackend and DefaultJob."""

import pytest

from qrisp import QuantumFloat
from qrisp.default_backend import DefaultBackend, DefaultJob
from qrisp.interface.job import JobResult, JobStatus


def _simple_computation():
    """Return a deterministic QuantumFloat computation (3 * 3 = 9)."""
    qf = QuantumFloat(4)
    qf[:] = 3
    return qf * qf


class TestDefaultJobInterface:
    """Tests that run() returns a proper DefaultJob and the Job interface is correct."""

    def test_run_returns_default_job(self):
        """Ensure run() returns a DefaultJob instance, not a raw dict."""
        backend = DefaultBackend()
        res = _simple_computation()
        qc = res.qs.compile()
        job = backend.run(qc)
        assert isinstance(job, DefaultJob)

    def test_job_is_done_before_run_returns(self):
        """Ensure the job is already DONE when run() returns to the caller."""
        backend = DefaultBackend()
        res = _simple_computation()
        qc = res.qs.compile()
        job = backend.run(qc)
        assert job.status() == JobStatus.DONE

    def test_job_done_returns_true(self):
        """Ensure done() is True after synchronous execution."""
        backend = DefaultBackend()
        res = _simple_computation()
        qc = res.qs.compile()
        job = backend.run(qc)
        assert job.done() is True

    def test_job_in_final_state_returns_true(self):
        """Ensure in_final_state() is True after synchronous execution."""
        backend = DefaultBackend()
        res = _simple_computation()
        qc = res.qs.compile()
        job = backend.run(qc)
        assert job.in_final_state() is True

    def test_result_returns_job_result_instance(self):
        """Ensure job.result() returns a JobResult object."""
        backend = DefaultBackend()
        res = _simple_computation()
        qc = res.qs.compile()
        job = backend.run(qc)
        assert isinstance(job.result(), JobResult)

    def test_cancel_returns_false(self):
        """Ensure cancel() always returns False for a synchronous job."""
        backend = DefaultBackend()
        res = _simple_computation()
        qc = res.qs.compile()
        job = backend.run(qc)
        assert job.cancel() is False

    def test_status_unchanged_after_cancel(self):
        """Ensure cancel() does not change the job status."""
        backend = DefaultBackend()
        res = _simple_computation()
        qc = res.qs.compile()
        job = backend.run(qc)
        job.cancel()
        assert job.status() == JobStatus.DONE


class TestDefaultBackendAnalytic:
    """Tests for analytic (exact probability) execution with shots=None."""

    def test_analytic_result_is_correct(self):
        """Verify that a deterministic computation yields probability 1.0 for the correct outcome."""
        backend = DefaultBackend()  # shots=None by default
        res = _simple_computation()
        assert res.get_measurement(backend=backend) == {9: 1.0}

    def test_analytic_probabilities_sum_to_one(self):
        """Verify that analytic result probabilities sum to 1.0."""
        backend = DefaultBackend()
        res = _simple_computation()
        qc = res.qs.compile()
        result = backend.run(qc).result()
        total = sum(result.get_counts().values())
        assert abs(total - 1.0) < 1e-9

    def test_analytic_result_contains_floats(self):
        """Verify that analytic results are floating-point probabilities, not integer counts."""
        backend = DefaultBackend()
        res = _simple_computation()
        qc = res.qs.compile()
        result = backend.run(qc).result()
        for value in result.get_counts().values():
            assert isinstance(value, float)


# We could add some shots-based tests here as well


class TestDefaultBackendOptions:
    """Tests for DefaultBackend options handling."""

    def test_default_shots_is_none(self):
        """Ensure the default shots option is None (analytic execution)."""
        assert DefaultBackend().options["shots"] is None

    def test_default_token_is_empty_string(self):
        """Ensure the default token option is an empty string."""
        assert DefaultBackend().options["token"] == ""

    def test_update_options_shots(self):
        """Ensure shots can be updated via update_options."""
        backend = DefaultBackend()
        backend.update_options(shots=512)
        assert backend.options["shots"] == 512

    def test_update_options_invalid_key_raises(self):
        """Ensure update_options raises AttributeError for an unknown key."""
        backend = DefaultBackend()
        with pytest.raises(AttributeError):
            backend.update_options(nonexistent=42)

    def test_options_independent_between_instances(self):
        """Ensure updating one instance does not affect another."""
        b1 = DefaultBackend()
        b2 = DefaultBackend()
        b1.update_options(shots=512)
        assert b2.options["shots"] is None

    def test_module_level_def_backend_is_default_backend(self):
        """Ensure the module-level def_backend singleton is a DefaultBackend instance."""
        from qrisp.default_backend import def_backend

        assert isinstance(def_backend, DefaultBackend)
