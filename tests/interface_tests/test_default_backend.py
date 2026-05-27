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

"""Tests for QrispSimulator and QrispSimulatorJob."""

import pytest

from qrisp import QuantumFloat, h
from qrisp.default_backend import QrispSimulator, QrispSimulatorJob, def_backend
from qrisp.interface import BatchedBackend
from qrisp.interface.job import JobResult, JobStatus
from qrisp.interface.measurement_result import LazyDict
from conftest import CountingWrapper


def _simple_computation():
    """Return a deterministic QuantumFloat computation (3 * 3 = 9)."""
    qf = QuantumFloat(4)
    qf[:] = 3
    return qf * qf


class TestQrispSimulatorJobInterface:
    """Tests that run_async() returns a proper QrispSimulatorJob and the Job interface is correct.

    Note: run_async() is used here (rather than run()) because these tests need the
    Job handle. run() returns a plain dict for backward compatibility and does not
    expose the Job object.
    """

    def test_run_async_returns_qrisp_simulator_job(self):
        """Ensure run_async() returns a QrispSimulatorJob instance, not a raw dict."""
        backend = QrispSimulator()
        res = _simple_computation()
        job = backend.run_async(res.qs.compile())
        assert isinstance(job, QrispSimulatorJob)

    def test_job_is_done_before_run_async_returns(self):
        """Ensure the job is already DONE when run_async() returns to the caller."""
        backend = QrispSimulator()
        res = _simple_computation()
        job = backend.run_async(res.qs.compile())
        assert job.status() == JobStatus.DONE

    def test_job_done_returns_true(self):
        """Ensure done() is True after synchronous execution."""
        backend = QrispSimulator()
        res = _simple_computation()
        job = backend.run_async(res.qs.compile())
        assert job.done() is True

    def test_job_in_final_state_returns_true(self):
        """Ensure in_final_state() is True after synchronous execution."""
        backend = QrispSimulator()
        res = _simple_computation()
        job = backend.run_async(res.qs.compile())
        assert job.in_final_state() is True

    def test_result_returns_job_result_instance(self):
        """Ensure job.result() returns a JobResult object."""
        backend = QrispSimulator()
        res = _simple_computation()
        job = backend.run_async(res.qs.compile())
        assert isinstance(job.result(), JobResult)

    def test_cancel_returns_false(self):
        """Ensure cancel() always returns False for a synchronous job."""
        backend = QrispSimulator()
        res = _simple_computation()
        job = backend.run_async(res.qs.compile())
        assert job.cancel() is False

    def test_status_unchanged_after_cancel(self):
        """Ensure cancel() does not change the job status."""
        backend = QrispSimulator()
        res = _simple_computation()
        job = backend.run_async(res.qs.compile())
        job.cancel()
        assert job.status() == JobStatus.DONE


class TestQrispSimulatorAnalytic:
    """Tests for analytic (exact probability) execution with shots=None."""

    def test_analytic_result_is_correct(self):
        """Verify that a deterministic computation yields probability 1.0."""
        backend = QrispSimulator()  # shots=None by default
        res = _simple_computation()
        assert res.get_measurement(backend=backend) == {9: 1.0}

    def test_analytic_probabilities_sum_to_one(self):
        """Verify that analytic result probabilities sum to 1.0."""
        backend = QrispSimulator()
        res = _simple_computation()
        result = backend.run_async(res.qs.compile()).result()
        total = sum(result.get_counts().values())
        assert abs(total - 1.0) < 1e-9

    def test_analytic_result_contains_floats(self):
        """Verify that analytic results are floating-point probabilities, not integer counts."""
        backend = QrispSimulator()
        res = _simple_computation()
        result = backend.run_async(res.qs.compile()).result()
        for value in result.get_counts().values():
            assert isinstance(value, float)


class TestQrispSimulatorShots:
    """Tests for shot-based (sampled) execution."""

    def test_shot_based_differs_from_analytic(self):
        """Verify that shot-based execution gives a different result than analytic execution."""
        qv = QuantumFloat(1)
        h(qv[0])  # equal superposition of 0 and 1

        backend = QrispSimulator()  # shots=None
        analytic_result = qv.get_measurement(backend=backend)
        assert analytic_result == {0: 0.5, 1: 0.5}

        backend.update_options(shots=1)
        sampled_result = qv.get_measurement(backend=backend)

        # With 1 shot the outcome is definite: one key with probability 1.0.
        assert len(sampled_result) == 1
        assert list(sampled_result.values())[0] == 1.0
        assert sampled_result != analytic_result

    def test_backend_shots_option_used_when_not_overridden(self):
        """Verify that the backend's shots option is used and produces a sampled result."""
        qv = QuantumFloat(1)
        h(qv[0])

        backend = QrispSimulator()
        backend.update_options(shots=1)
        result = qv.get_measurement(backend=backend)

        # With 1 shot the outcome must be a single definite value.
        assert len(result) == 1
        assert list(result.values())[0] == 1.0


class TestQrispSimulatorOptions:
    """Tests for QrispSimulator options handling."""

    def test_default_shots_is_none(self):
        """Ensure the default shots option is None (analytic execution)."""
        assert QrispSimulator().options["shots"] is None

    def test_default_token_is_empty_string(self):
        """Ensure the default token option is an empty string."""
        assert QrispSimulator().options["token"] == ""

    def test_update_options_shots(self):
        """Ensure shots can be updated via update_options."""
        backend = QrispSimulator()
        backend.update_options(shots=512)
        assert backend.options["shots"] == 512

    def test_update_options_invalid_key_raises(self):
        """Ensure update_options raises AttributeError for an unknown key."""
        backend = QrispSimulator()
        with pytest.raises(AttributeError):
            backend.update_options(nonexistent=42)

    def test_options_independent_between_instances(self):
        """Updating one instance must not affect another independent instance."""
        b1 = QrispSimulator()
        b2 = QrispSimulator()
        b1.update_options(shots=512)
        assert b2.options["shots"] is None

    def test_module_level_def_backend_is_qrisp_simulator(self):
        """Ensure the module-level def_backend is a QrispSimulator instance."""
        assert isinstance(def_backend, QrispSimulator)


class TestQrispSimulatorBatched:
    """Verify that QrispSimulator.batched() produces correct lazy results."""

    def test_batched_returns_batched_backend(self):
        """batched() must return a BatchedBackend wrapping the original backend."""
        backend = QrispSimulator()
        bb = backend.batched()
        assert isinstance(bb, BatchedBackend)
        assert bb._backend is backend

    def test_result_is_lazy_before_dispatch(self):
        """get_measurement via batched QrispSimulator must raise before dispatch."""
        bb = QrispSimulator().batched()
        res = _simple_computation().get_measurement(backend=bb)
        assert isinstance(res, LazyDict)
        with pytest.raises(RuntimeError, match="dispatch"):
            len(res)
        bb.dispatch()
        assert res == {9: 1.0}

    def test_dispatch_populates_two_results(self):
        """dispatch() must populate both results correctly for two independent circuits."""
        qf1 = QuantumFloat(3)
        qf1[:] = 3
        qf2 = QuantumFloat(3)
        qf2[:] = 5

        bb = QrispSimulator().batched()
        res1 = qf1.get_measurement(backend=bb)
        res2 = qf2.get_measurement(backend=bb)
        bb.dispatch()

        assert res1 == {3: 1.0}
        assert res2 == {5: 1.0}

    def test_n_circuits_one_run_async_call(self):
        """Two circuits queued with QrispSimulator must produce exactly one run_async call."""
        counting = CountingWrapper(QrispSimulator())
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

    def test_batched_inherits_shots_from_wrapped_backend(self):
        """BatchedBackend created via batched() inherits the wrapped backend's options."""
        backend = QrispSimulator(options={"shots": 256, "token": ""})
        bb = backend.batched()
        assert bb.options["shots"] == 256
