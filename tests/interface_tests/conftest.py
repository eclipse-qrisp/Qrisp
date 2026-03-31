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

"""Shared test utilities for the Backend abstraction layer test suite."""

from __future__ import annotations

import threading

from qrisp.interface.backend import Backend
from qrisp.interface.job import Job, JobResult, JobStatus


class MinimalJob(Job):
    """
    The smallest possible concrete `Job` class that can be instantiated.

    Used to unit-test the base class behaviour in isolation.
    All state is managed directly via private attributes.
    Threading is supported internally via a ``threading.Event`` class
    so that ``result`` can block correctly, but this is an implementation
    detail of this test helper (not a requirement of the base class).
    """

    def __init__(
        self, backend, job_id=None, initial_status=JobStatus.INITIALIZING, **kwargs
    ):
        super().__init__(backend=backend, job_id=job_id, **kwargs)
        self._status = initial_status
        self._result_data = None
        self._error = None
        self._done_event = threading.Event()

    # ── Abstract interface ────────────────────────────────────────────

    def submit(self) -> None:
        pass

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
        if self.in_final_state():
            return False
        self._status = JobStatus.CANCELLED
        self._done_event.set()
        return True

    def status(self) -> JobStatus:
        return self._status

    # ── Internal helpers (not part of the abstract contract) ─────────

    def _resolve(self, result: JobResult) -> None:
        """Mark the job as successfully done."""
        self._result_data = result
        self._status = JobStatus.DONE
        self._done_event.set()

    def _fail(self, error: Exception) -> None:
        """Mark the job as failed."""
        self._error = error
        self._status = JobStatus.ERROR
        self._done_event.set()

    def _set_status(self, status: JobStatus) -> None:
        self._status = status


class MinimalBackend(Backend):
    """
    The smallest possible concrete `Backend` class that can be instantiated.

    Executes synchronously and returns a `MinimalJob` that is
    already `JobStatus.DONE` before `run` returns.
    Counts are trivially ``{"0": n_shots}`` for each circuit.
    """

    def run_async(self, circuits, shots=None) -> MinimalJob:
        if not isinstance(circuits, list):
            circuits = [circuits]
        n_shots = shots if shots is not None else self.options["shots"]
        job = MinimalJob(backend=self)
        job._set_status(JobStatus.RUNNING)
        counts = [{"0": n_shots} for _ in circuits]
        job._resolve(JobResult(counts))
        return job
