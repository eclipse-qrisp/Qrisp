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

"""
This module defines :class:`BatchedBackend` and its associated :class:`BatchedJob`,
which allow multiple concurrent Qrisp threads to share a single backend call.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, cast

from qrisp.interface.backend import Backend
from qrisp.interface.job import Job, JobResult, JobStatus

if TYPE_CHECKING:
    from qrisp.circuit.quantum_circuit import QuantumCircuit


class BatchedJob(Job):
    """
    A :class:`Job` produced by :class:`BatchedBackend`.

    Each call to ``BatchedBackend.run_async`` creates one ``BatchedJob`` and
    registers it in the backend's pending queue.  The job stays in
    ``QUEUED`` state until ``BatchedBackend.dispatch`` is called, at
    which point all pending jobs are resolved atomically.

    Callers block in ``result`` via a private ``threading.Event`` that is
    set by the backend when the batch result for this job becomes available.
    """

    def __init__(self, backend: BatchedBackend, circuits: Sequence, shots: int):
        """Initialise the job with the backend, normalised circuit list, and shot count."""
        super().__init__(backend=backend)
        self._circuits = circuits
        self._shots = shots
        self._status = JobStatus.INITIALIZING
        self._result_data = None
        self._error = None
        # Each job owns its private event so it can be unblocked independently
        # of all other pending jobs.
        self._done_event = threading.Event()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def submit(self) -> None:
        """Mark the job as QUEUED after it has been added to the batch."""
        self._status = JobStatus.QUEUED

    def result(self, timeout: float | None = None) -> JobResult:
        """
        Block until this job's circuits have been executed by a dispatch call.

        Parameters
        ----------
        timeout : float or None
            Maximum time to wait in seconds.  ``None`` waits indefinitely.

        Returns
        -------
        JobResult

        Raises
        ------
        TimeoutError
            If ``timeout`` expires before the job is resolved.

        RuntimeError
            If the batch execution raised an exception, or if the job was
            cancelled.
        """
        if not self._done_event.wait(timeout=timeout):
            raise TimeoutError(
                f"BatchedJob did not complete within {timeout}s. "
                "Has dispatch() been called?"
            )
        if self._status == JobStatus.ERROR:
            raise RuntimeError(
                f"Batch execution failed: {self._error}"
            ) from self._error
        if self._status == JobStatus.CANCELLED:
            raise RuntimeError("BatchedJob was cancelled.")
        return cast(JobResult, self._result_data)

    def cancel(self) -> bool:
        """
        Attempt to cancel the job.

        Cancellation is not supported for queued batch jobs because the
        circuit has already been registered in the shared batch.  Returns
        ``False`` in all cases.
        """
        return False

    def status(self) -> JobStatus:
        """Return the current :class:`JobStatus` of this job."""
        return self._status

    # ------------------------------------------------------------------
    # Internal helpers — called only by BatchedBackend.dispatch()
    # ------------------------------------------------------------------

    def _resolve(self, result: JobResult) -> None:
        """Mark the job as DONE, store the result, and unblock result()."""
        self._result_data = result
        self._status = JobStatus.DONE
        self._done_event.set()

    def _fail(self, error: Exception) -> None:
        """Mark the job as ERROR, store the exception, and unblock result()."""
        self._error = error
        self._status = JobStatus.ERROR
        self._done_event.set()


class BatchedBackend(Backend):
    """
    A backend that collects circuits submitted from multiple concurrent threads
    and forwards them to the underlying hardware or simulator in a single batch call.

    Many physical backends have high per-call overhead from network latency,
    authentication, and compilation.  This overhead is typically amortised by
    submitting several circuits together. ``BatchedBackend`` bridges the gap
    between Qrisp's per-variable programming model and this batched execution
    pattern.

    In other words, this Backend does not parallelize the execution of circuits
    within ``batch_run_func``. What it parallelizes is the preparation phase:
    multiple threads can each be building Qrisp state, compiling circuits,
    and doing post-processing concurrently. The benefit of batching is not circuit-level
    parallelism but rather reduced overhead.

    **How it works**

    Each call to ``run_async`` (typically one per thread running Qrisp code)
    registers the circuit(s) in a shared pending queue and returns a
    ``BatchedJob`` immediately.  The caller then blocks in
    :meth:`Job.result` waiting for its result.

    When ``dispatch`` is called (either explicitly from the main thread,
    or automatically when ``run_async`` is called from the main thread) all
    pending jobs are collected, their circuits are forwarded to
    ``batch_run_func`` as a flat list, and the results are distributed back
    to each waiting job via private ``threading.Event`` objects.

    .. note::

        Calling ``run_async`` from the *main thread* will automatically
        dispatch all currently pending queries, including the one just added.
        This ensures that single-threaded use works without any manual
        dispatch call.

    Parameters
    ----------
    batch_run_func : callable
        A function that receives a list of ``(QuantumCircuit, int)`` tuples
        (circuit and shot count) and returns a list of measurement
        result dictionaries.

    name : str or None
        Optional name for the backend.

    options : Mapping or None
        Runtime options.  Defaults to ``{"shots": 1024}``.

    Examples
    --------

    We first define a simple ``batch_run_func`` that simulates the execution of a batch of circuits.
    Then, we provide it to :class:`BatchedBackend` and demonstrate how to use the backend in both multithreaded
    and single-threaded contexts.

    .. code-block:: python

        from qrisp import *
        from qrisp.interface.batched_backend import BatchedBackend


        def run_func_batch(batch):
            results = []
            for qc, shots in batch:
                results.append(qc.run(shots=shots))
            return results


        batched_backend = BatchedBackend(run_func_batch)

    At this point, we can use ``batched_backend`` in place of any normal Qrisp backend.

    **Multithreading Example**

    In this example, we create two threads that each measure a single QuantumFloat.

    .. code-block:: python

        import threading

        a = QuantumFloat(4)
        b = QuantumFloat(3)
        a[:] = 1
        b[:] = 2
        c = a + b

        d = QuantumFloat(4)
        e = QuantumFloat(3)
        d[:] = 2
        e[:] = 3
        f = d + e

        results = []

        def eval_measurement(qv):
            results.append(qv.get_measurement(backend=batched_backend))

        thread_0 = threading.Thread(target=eval_measurement, args=(c,))
        thread_1 = threading.Thread(target=eval_measurement, args=(f,))

        thread_0.start()
        thread_1.start()

        # At this point, both threads have been started. They will each call get_measurement,
        # which internally calls run_async() and registers their circuits as BatchedJobs
        # in the backend's pending queue, where each BatchedJob is in QUEUED state.
        # Each thread then blocks waiting for its result.

        # Now we call dispatch to execute the batch of pending jobs.
        # This will unblock both threads and deliver their results once the batch_run_func completes.
        batched_backend.dispatch(min_calls=2)

        thread_0.join()
        thread_1.join()


    Each thread calls ``get_measurement`` with the ``batched_backend``,
    which internally calls ``run_async``. This registers the circuit
    as a ``BatchedJob`` in ``JobStatus.QUEUED`` state and returns immediately.

    When the main thread calls ``dispatch``, the batch of pending jobs is executed,
    and each job's result is resolved and delivered to the waiting threads.

    We can check the results:

    >>> results
    [{3: 1.0}, {5: 1.0}]

    This workflow is also automated by :func:`qrisp.batched_measurement`:

    >>> batched_measurement([c, f], backend=batched_backend)
    [{3: 1.0}, {5: 1.0}]


    **Single-threaded Example**

    In this example, we call ``run_async`` directly from the main thread.
    This will automatically trigger a dispatch without needing to call it manually.

    .. code-block:: python

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1])

        # When run_async() is called from the main thread, it automatically dispatches
        # the batch of pending jobs, so we don't need to call dispatch() manually here.
        batched_job = batched_backend.run_async(qc, shots=1024)

        job_result = batched_job.result()

    We can check the status of the job and its results:

    >>> print(batched_job.status())
    done

    >>> job_result.get_counts(0)
    {'00': 553, '11': 471}

    """

    def __init__(
        self,
        batch_run_func: Callable[[list[tuple["QuantumCircuit", int]]], list[Mapping]],
        name: str | None = None,
        options: Mapping | None = None,
    ):
        """Initialise the backend with a batch execution function."""
        super().__init__(name=name, options=options)
        self.batch_run_func = batch_run_func
        # Pending jobs waiting to be dispatched.
        self._pending: list[BatchedJob] = []
        # Lock that protects _pending against concurrent modification.
        self._lock = threading.Lock()

    @classmethod
    def _default_options(cls):
        """Return the default runtime options (shots=1024)."""
        return {"shots": 1024}

    def run_async(self, circuits, shots: int | None = None) -> BatchedJob:
        """
        Register one or more circuits in the batch and return a ``BatchedJob``.

        This method returns immediately.  The job stays in ``QUEUED`` state
        until :meth:`dispatch` resolves it.  Call :meth:`Job.result` on the
        returned job to block until the result is available.

        When called from the main thread, a dispatch is started
        automatically in a background thread so that single-threaded usage
        requires no manual dispatch call.

        Parameters
        ----------
        circuits : QuantumCircuit or Sequence[QuantumCircuit]
            One circuit or a sequence of circuits to include in the batch.

        shots : int or None
            Number of shots.  Defaults to the backend's ``shots`` option.

        Returns
        -------
        BatchedJob
        """
        circuits = [circuits] if not isinstance(circuits, Sequence) else circuits
        n_shots = shots if shots is not None else self._options["shots"]

        job = BatchedJob(backend=self, circuits=circuits, shots=n_shots)

        with self._lock:
            self._pending.append(job)

        job.submit()

        # Auto-dispatch when called from the main thread so that the
        # single-circuit / single-threaded use case works without manual dispatch.
        if threading.current_thread() is threading.main_thread():
            threading.Thread(target=self.dispatch, daemon=True).start()

        return job

    def dispatch(self, min_calls: int = 0) -> None:
        """
        Execute all pending batch jobs and resolve their ``BatchedJob`` objects.

        This method may be called from any thread.  It waits until at least
        ``min_calls`` jobs are queued before proceeding, which is useful when
        you know exactly how many threads will submit circuits.

        Parameters
        ----------
        min_calls : int, optional
            Minimum number of pending jobs to wait for before dispatching.
            Defaults to ``0`` (dispatch whatever is currently queued).

        Notes
        -----
        The ``batch_run_func`` receives a *flat* list of
        ``(QuantumCircuit, int)`` tuples, one entry per circuit (even when a
        single :meth:`run_async` call submitted multiple circuits).  Results are
        re-assembled into :class:`JobResult` objects and delivered to each
        ``BatchedJob`` via its private ``threading.Event``.
        """
        # Wait until the required number of jobs has accumulated.
        while True:
            with self._lock:
                if len(self._pending) >= min_calls:
                    break
            time.sleep(0.01)

        # Atomically snapshot and clear the pending queue so that new calls
        # to run_async() during dispatch do not interfere with this batch.
        with self._lock:
            pending = list(self._pending)
            self._pending = []

        if not pending:
            return

        # Build the flat (circuit, shots) list expected by batch_run_func.
        # A single run_async() call may have submitted multiple circuits, so we
        # expand each job's circuit list individually.
        flat_batch: list[tuple] = []
        for job in pending:
            for circuit in job._circuits:
                flat_batch.append((circuit, job._shots))

        try:
            flat_results = self.batch_run_func(flat_batch)
            flat_results = [
                result if isinstance(result, dict) else dict(result)
                for result in flat_results
            ]

            # NOTE: It might be worth validating the results here,
            # since `batch_run_func` is user-provided and could return malformed data.

            # Re-assemble flat results back into per-job JobResult objects.
            # Each job contributed len(job._circuits) consecutive entries.
            idx = 0
            for job in pending:
                n = len(job._circuits)
                job._resolve(JobResult(flat_results[idx : idx + n]))
                idx += n

        except Exception as exc:
            # Propagate the exception to every waiting job so that their
            # result() calls raise rather than block indefinitely.
            for job in pending:
                job._fail(exc)
