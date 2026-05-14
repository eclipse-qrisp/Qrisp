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

"""Minimal tests for BatchedBackend."""

import threading

import pytest

from qrisp import QuantumFloat
from qrisp.interface import BatchedBackend
from qrisp.interface.job import JobStatus


def _make_batch_run_func():
    """Return a batch_run_func that delegates to each circuit's own run method."""

    def run_func_batch(batch):
        """Execute each (circuit, shots) pair and return the list of result dicts."""
        return [qc.run(shots=shots) for qc, shots in batch]

    return run_func_batch


def test_batched_backend_docstring_example():
    """Reproduce the docstring example: two threads measuring two QuantumFloats.

    Verifies that both results are correct after a manual dispatch with min_calls=2.
    This is the primary integration test for BatchedBackend.
    """
    a = QuantumFloat(4)
    b = QuantumFloat(3)
    a[:] = 1
    b[:] = 2
    c = a + b  # expected result: 3

    d = QuantumFloat(4)
    e = QuantumFloat(3)
    d[:] = 2
    e[:] = 2
    f = d + e  # expected result: 4

    bb = BatchedBackend(_make_batch_run_func())

    results = []

    def eval_measurement(qv):
        """Measure qv and append the result to the shared list."""
        results.append(qv.get_measurement(backend=bb))

    thread_0 = threading.Thread(target=eval_measurement, args=(c,))
    thread_1 = threading.Thread(target=eval_measurement, args=(f,))

    thread_0.start()
    thread_1.start()

    bb.dispatch(min_calls=2)

    thread_0.join()
    thread_1.join()

    assert {3: 1.0} in results
    assert {4: 1.0} in results


def test_batched_backend_main_thread_auto_dispatch():
    """Verify that run() from the main thread auto-dispatches without a manual call.

    This ensures that single-threaded use (e.g. a plain script) works correctly
    without the user having to call dispatch() explicitly.
    """
    a = QuantumFloat(4)
    b = QuantumFloat(3)
    a[:] = 1
    b[:] = 2
    c = a + b  # expected result: 3

    bb = BatchedBackend(_make_batch_run_func())

    # Called from the main thread — dispatch happens automatically.
    result = c.get_measurement(backend=bb)

    assert result == {3: 1.0}


def test_batched_backend_exception_propagates_to_all_jobs():
    """Verify that an exception raised by batch_run_func reaches every waiting thread.

    If the batch call fails, all threads blocked in result() must receive a
    RuntimeError rather than hanging indefinitely.
    """

    def failing_batch_run(batch):
        """Always raises to simulate a hardware or network fault."""
        raise RuntimeError("Simulated hardware fault")

    bb = BatchedBackend(failing_batch_run)

    a = QuantumFloat(4)
    b = QuantumFloat(3)
    a[:] = 1
    b[:] = 2
    c = a + b

    d = QuantumFloat(4)
    e = QuantumFloat(3)
    d[:] = 2
    e[:] = 2
    f = d + e

    errors = []

    def eval_measurement(qv):
        """Attempt to measure qv and record any RuntimeError."""
        try:
            qv.get_measurement(backend=bb)
        except RuntimeError as exc:
            cause_str = str(exc.__cause__) if exc.__cause__ is not None else ""
            errors.append(str(exc) + " " + cause_str)

    thread_0 = threading.Thread(target=eval_measurement, args=(c,))
    thread_1 = threading.Thread(target=eval_measurement, args=(f,))

    thread_0.start()
    thread_1.start()

    bb.dispatch(min_calls=2)

    thread_0.join()
    thread_1.join()

    # Both threads must have received the error.
    assert len(errors) == 2
    assert all("Simulated hardware fault" in e for e in errors)


def test_batched_backend_timeout():
    """Verify that result(timeout=...) raises TimeoutError when dispatch is never called.

    This guards against the case where a thread submits a circuit but the
    coordinating dispatch call is accidentally omitted.
    """
    bb = BatchedBackend(_make_batch_run_func())

    a = QuantumFloat(4)
    b = QuantumFloat(3)
    a[:] = 1
    b[:] = 2

    errors = []

    def submit_without_dispatch():
        """Submit a job from a background thread and expect a TimeoutError."""
        c = a + b
        job = bb.run_async(c.qs.compile())
        try:
            job.result(timeout=0.5)
        except TimeoutError as exc:
            errors.append(str(exc))

    t = threading.Thread(target=submit_without_dispatch)
    t.start()
    t.join()

    assert len(errors) == 1
    assert "dispatch" in errors[0].lower()


def test_dispatch_timeout_raises_timeout_error():
    """dispatch(min_calls=N, timeout=T) must raise TimeoutError if T expires before N jobs arrive.

    Without a timeout, a dispatch() waiting for a thread that never arrives
    would block forever. This guards against that deadlock.
    """
    bb = BatchedBackend(_make_batch_run_func())
    with pytest.raises(TimeoutError, match="timed out"):
        bb.dispatch(min_calls=999, timeout=0.1)


def test_batched_job_cancel_always_returns_false():
    """BatchedJob.cancel() must always return False — cancellation is not supported.

    Since circuits are registered in a shared batch before execution starts,
    there is no safe point at which an individual job can be withdrawn.
    Callers must not rely on cancel() to prevent execution.
    """
    bb = BatchedBackend(_make_batch_run_func())

    # Submit and complete a job via auto-dispatch (main-thread call).
    a = QuantumFloat(4)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 2
    c = a + b

    job = bb.run_async(c.qs.compile(), shots=32)
    job.result()

    assert job.status() == JobStatus.DONE
    assert job.cancel() is False


def test_batched_job_cancel_returns_false_while_queued():
    """BatchedJob.cancel() must return False even when the job is still in QUEUED state."""
    bb = BatchedBackend(_make_batch_run_func())

    job_holder = []
    ready = threading.Event()

    def worker():
        a = QuantumFloat(4)
        a[:] = 1
        b = QuantumFloat(3)
        b[:] = 2
        c = a + b
        job = bb.run_async(c.qs.compile(), shots=32)
        job_holder.append(job)
        ready.set()
        try:
            job.result(timeout=5.0)
        except Exception:
            pass

    t = threading.Thread(target=worker)
    t.start()
    ready.wait(timeout=5.0)

    job = job_holder[0]
    assert job.status() == JobStatus.QUEUED
    assert job.cancel() is False

    # Let the worker finish cleanly.
    bb.dispatch()
    t.join(timeout=5.0)
