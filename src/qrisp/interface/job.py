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

"""
This module defines the abstract :class:`Job` interface and related types for representing and managing backend executions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from .backend import Backend


class JobStatus(Enum):
    """
    Enumeration of possible lifecycle states for a :class:`Job`.

    The typical progression is:

    .. code-block:: text

        INITIALIZING → QUEUED → RUNNING → DONE
                                        ↘ ERROR
                       ↘ CANCELLED (at any point before DONE)

    Attributes
    ----------
    INITIALIZING :
        The job has been created but not yet submitted to the backend.
    QUEUED :
        The job has been submitted and is waiting for execution resources.
    RUNNING :
        The job is currently being executed.
    DONE :
        The job completed successfully. Results are available.
    CANCELLED :
        The job was cancelled before or during execution.
    ERROR :
        The job failed due to an error during execution.
    """

    INITIALIZING = auto()
    QUEUED = auto()
    RUNNING = auto()
    DONE = auto()
    CANCELLED = auto()
    ERROR = auto()


#: Terminal states: once a job reaches one of these, its outcome is final.
JOB_FINAL_STATES = (JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR)


class JobResult:
    """
    Wraps the outcome of one or more circuit executions.

    Counts are stored as a list with one dictionary per input circuit,
    preserving the order of submission. For single-circuit executions,
    index ``0`` is used.

    Parameters
    ----------
    counts : List[Dict[str, int]]
        A list of measurement-outcome dictionaries, one per circuit.
        Keys are bitstring representations of measurement outcomes;
        values are the number of times that outcome was observed.

    metadata : Dict[str, Any] | None
        Optional backend-specific metadata associated with the execution
        (e.g. timing information, circuit identifiers, hardware calibration
        snapshot). Defaults to an empty dict.

    Examples
    --------
    Single-circuit result:

    >>> result = JobResult([{"00": 512, "11": 512}])
    >>> result.get_counts()
    {"00": 512, "11": 512}

    Batch result:

    >>> result = JobResult([{"0": 800, "1": 224}, {"00": 500, "11": 524}])
    >>> result.get_counts(0)
    {"0": 800, "1": 224}
    >>> result.get_counts(1)
    {"00": 500, "11": 524}
    """

    def __init__(
        self,
        counts: List[Dict[str, int]],
        metadata: Dict[str, Any] | None = None,
    ):
        if not isinstance(counts, list):
            raise TypeError(
                f"'counts' must be a list of dicts, got {type(counts).__name__}"
            )
        self._counts = counts
        self.metadata = metadata or {}

    def get_counts(self, index: int = 0) -> Dict[str, int]:
        """
        Return the measurement-outcome counts for a single circuit.

        Parameters
        ----------
        index : int
            Index of the circuit in the batch. Defaults to ``0``.

        Returns
        -------
        Dict[str, int]
            Mapping from bitstring outcome to observed count.

        Raises
        ------
        IndexError
            If ``index`` is out of range for the number of circuits in this
            result.
        """
        try:
            return self._counts[index]
        except IndexError as exc:
            raise IndexError(
                f"Result contains {len(self._counts)} circuit(s); "
                f"index {index} is out of range."
            ) from exc

    @property
    def all_counts(self) -> List[Dict[str, int]]:
        """All measurement-outcome counts, one dict per circuit."""
        return self._counts

    @property
    def num_circuits(self) -> int:
        """Number of circuits whose results are stored in this object."""
        return len(self._counts)

    def __repr__(self) -> str:
        return (
            f"JobResult(num_circuits={self.num_circuits}, "
            f"metadata={self.metadata!r})"
        )


class Job(ABC):
    """
    Abstract handle for a (potentially asynchronous) backend execution.

    A :class:`Job` is returned by :meth:`Backend.run` immediately after
    submission. The caller can then:

    * call :meth:`result` to wait for the outcome and retrieve it, or
    * poll :attr:`status` / :meth:`done` without blocking, or
    * call :meth:`cancel` to attempt cancellation.

    This follows the **Future** (or **Promise**) pattern: execution happens
    independently of the caller, and the caller decides *when* to wait for
    the outcome. This is the same philosophy adopted by Qiskit — calling
    ``backend.run()`` returns a ``Job`` object, and the caller invokes
    ``job.result()`` when it is ready to receive the outcome.

    .. rubric:: Design contract

    The base class defines **only the observable contract** — the four
    abstract methods that every concrete job must implement. It deliberately
    prescribes no internal synchronisation mechanism (no threading, no
    asyncio, no polling loop). Each concrete subclass chooses whatever
    concurrency model suits its execution environment:

    * A synchronous simulator may compute the result inline and make it
      available before :meth:`result` is ever called.
    * An asynchronous hardware backend may submit to a remote queue and
      poll in a background thread or coroutine.
    * An asyncio-based backend may override :meth:`result` with a coroutine.

    From the caller's perspective, all of the above are used identically.
    This is the same approach taken by Qiskit's ``JobV1``.

    .. rubric:: Relationship to design patterns

    ``Job`` is a **Virtual Proxy** for the execution result: it is a local
    placeholder that controls access to an outcome that may be remote,
    deferred, or not yet computed. Together with :class:`Backend`, it forms
    a **Bridge**: the two abstract hierarchies (submission interface and
    execution handle) can be varied and extended independently.

    .. rubric:: Subclassing

    Concrete subclasses **must** implement:

    * :meth:`submit` — trigger the actual execution.
    * :meth:`result` — block (or await) until the result is available.
    * :meth:`cancel` — attempt to cancel the job.
    * :meth:`status` — return the current :class:`JobStatus`.

    Parameters
    ----------
    backend : Backend
        The backend that created this job.

    job_id : str or None
        Optional caller-supplied identifier (e.g. a remote job ID assigned
        by a vendor API). If ``None``, the concrete subclass is responsible
        for assigning one.
    """

    def __init__(self, backend: "Backend", job_id: str | None = None):
        self._backend = backend
        self._job_id = job_id

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def submit(self) -> None:
        """
        Submit the job to the backend for execution.

        This method triggers the actual execution. It is called by the
        backend after the job object has been constructed.
        """

    @abstractmethod
    def result(self) -> JobResult:
        """
        Block until the job finishes and return its :class:`JobResult`.

        The internal mechanism used to wait for completion is left entirely
        to the concrete implementation. It may use threading primitives,
        asyncio, polling, or any other strategy appropriate for the backend.

        Returns
        -------
        JobResult

        Raises
        ------
        RuntimeError
            If the job failed or was cancelled.
        """

    @abstractmethod
    def cancel(self) -> bool:
        """
        Attempt to cancel the job.

        Returns
        -------
        bool
            ``True`` if the cancellation request was accepted;
            ``False`` if the job has already reached a terminal state or
            the backend does not support cancellation.

        Notes
        -----
        Cancellation is best-effort. A return value of ``True`` indicates
        that the request was accepted, not necessarily that execution has
        already stopped.
        """

    @abstractmethod
    def status(self) -> JobStatus:
        """
        Return the current :class:`JobStatus` of the job.

        This method must be non-blocking. It should reflect the most
        recently known state without waiting for any transition.

        Returns
        -------
        JobStatus
        """

    # ------------------------------------------------------------------
    # Concrete helpers derived from status()
    # ------------------------------------------------------------------

    def done(self) -> bool:
        """
        Return ``True`` if the job has reached a terminal state.

        Terminal states are :attr:`~JobStatus.DONE`,
        :attr:`~JobStatus.CANCELLED`, and :attr:`~JobStatus.ERROR`.
        """
        return self.status() in JOB_FINAL_STATES

    def running(self) -> bool:
        """Return ``True`` if the job is currently executing."""
        return self.status() == JobStatus.RUNNING

    def queued(self) -> bool:
        """Return ``True`` if the job is waiting in the backend queue."""
        return self.status() == JobStatus.QUEUED

    def in_final_state(self) -> bool:
        """Alias for :meth:`done`. Provided for Qiskit naming compatibility."""
        return self.done()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def job_id(self) -> str | None:
        """Identifier for this job, or ``None`` if not yet assigned."""
        return self._job_id

    @property
    def backend(self) -> "Backend":
        """The :class:`Backend` that created this job."""
        return self._backend

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"job_id={self._job_id!r}, "
            f"status={self.status().name})"
        )
