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

"""This module defines :class:`DefaultBackend` and its associated :class:`DefaultJob`."""

from __future__ import annotations

from typing import Sequence, cast

from qrisp.interface.backend import Backend
from qrisp.interface.job import Job, JobResult, JobStatus
from qrisp.simulator.simulator import run as default_run


class DefaultJob(Job):
    """
    A synchronous :class:`~qrisp.interface.Job` produced by :class:`DefaultBackend`.

    Because the built-in Qrisp simulator executes circuits inline,
    the result is fully available before :meth:`result` is ever called.
    :meth:`submit` performs the actual execution and transitions the job
    to :attr:`~qrisp.interface.JobStatus.DONE` (or
    :attr:`~qrisp.interface.JobStatus.ERROR`) synchronously.
    """

    def __init__(self, backend: "DefaultBackend", circuits: Sequence, shots):
        """Initialise the job with the backend, normalised circuit list, and shot count."""
        super().__init__(backend=backend)
        self._circuits = circuits
        self._shots = shots
        self._status = JobStatus.INITIALIZING
        self._result_data = None
        self._error = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def submit(self) -> None:
        """Execute all circuits synchronously and store the result."""
        token = self._backend.options.get("token", "")
        self._status = JobStatus.RUNNING
        try:
            counts_list = [
                default_run(circuit, self._shots, token) for circuit in self._circuits
            ]
            self._result_data = JobResult(counts_list)
            self._status = JobStatus.DONE
        except Exception as exc:
            self._error = exc
            self._status = JobStatus.ERROR

    def result(self) -> JobResult:
        """Return the :class:`~qrisp.interface.JobResult`.

        Because the simulator is synchronous the result is already available
        as soon as :meth:`submit` has been called by
        :meth:`DefaultBackend.run_async`.

        Returns
        -------
        JobResult

        Raises
        ------
        RuntimeError
            If the simulation raised an exception.
        """
        if self._status == JobStatus.ERROR:
            raise RuntimeError(f"DefaultJob failed: {self._error}") from self._error
        return cast(JobResult, self._result_data)

    def cancel(self) -> bool:
        """Return ``False``: synchronous jobs cannot be cancelled after submission."""
        return False

    def status(self) -> JobStatus:
        """Return the current :class:`~qrisp.interface.JobStatus` of the job."""
        return self._status


class DefaultBackend(Backend):
    """
    The default Qrisp backend, backed by the built-in statevector simulator.

    This is the simplest concrete :class:`~qrisp.interface.Backend`
    implementation.  It executes circuits synchronously — the
    :class:`DefaultJob` returned by :meth:`run_async` is already
    :attr:`~qrisp.interface.JobStatus.DONE` before :meth:`run_async` returns
    to the caller.

    Parameters
    ----------
    options : Mapping or None, optional
        Runtime options.  Defaults to ``{"shots": None, "token": ""}``.

    Examples
    --------

    **Analytic execution (default)**

    We first create a :class:`DefaultBackend`:

    >>> from qrisp import QuantumFloat
    >>> from qrisp.default_backend import DefaultBackend
    >>> backend = DefaultBackend()

    When ``get_measurement`` is called, Qrisp compiles the computation into a
    circuit and passes it to the built-in simulator via :meth:`run_async`. A
    :class:`DefaultJob` is returned immediately. And, because the simulator
    is synchronous the job is already :attr:`~qrisp.interface.JobStatus.DONE`
    before ``get_measurement`` even calls :meth:`~qrisp.interface.Job.result`:

    >>> qf = QuantumFloat(3)
    >>> qf[:] = 4
    >>> res = qf * qf
    >>> res.get_measurement(backend=backend)
    {16: 1.0}

    With ``shots=None`` the result is an exact probability distribution, so
    ``{16: 1.0}`` means the outcome ``16`` has probability ``1.0``.

    **Updating options after construction**

    Runtime options can be updated via :meth:`~qrisp.interface.Backend.update_options`.
    Only keys that were present at construction time may be modified:

    >>> backend.update_options(shots=512)
    >>> print(backend.options["shots"])
    512

    **Using the Job interface directly**

    :meth:`run_async` returns a :class:`DefaultJob` that supports the full
    :class:`~qrisp.interface.Job` interface, even though the result is
    already available synchronously:

    >>> from qrisp import QuantumFloat
    >>> from qrisp.default_backend import DefaultBackend
    >>> backend = DefaultBackend()
    >>> qf3 = QuantumFloat(2)
    >>> qf3[:] = 3
    >>> res3 = qf3 * qf3
    >>> qc = res3.qs.compile()
    >>> qc.measure(qc.qubits)
    >>> job = backend.run_async(qc)
    >>> print(job.status())
    done
    >>> result = job.result()
    >>> print(result.get_counts())
    {'0100111': 1.0}
    """

    @classmethod
    def _default_options(cls):
        """Return the default runtime options.

        ``shots=None`` enables analytic (exact probability) execution.
        ``token`` is passed through to the simulator for authenticated backends.
        """
        return {"shots": None, "token": ""}

    def run_async(self, circuits, shots: int | None = None) -> DefaultJob:
        """Submit one or more circuits to the built-in simulator.

        This method returns a :class:`DefaultJob` that is already
        :attr:`~qrisp.interface.JobStatus.DONE` before :meth:`run_async`
        returns, because the simulator executes synchronously inside
        :meth:`~DefaultJob.submit`.

        Parameters
        ----------
        circuits : QuantumCircuit or list[QuantumCircuit]
            One circuit or a list of circuits to simulate.

        shots : int or None, optional
            Number of shots.  ``None`` selects analytic execution.
            If not provided, the backend's ``shots`` option is used.

        Returns
        -------
        DefaultJob
        """
        circuits = [circuits] if not isinstance(circuits, list) else circuits
        n_shots = shots if shots is not None else self.options.get("shots")
        job = DefaultJob(backend=self, circuits=circuits, shots=n_shots)
        job.submit()
        return job


def_backend = DefaultBackend()
