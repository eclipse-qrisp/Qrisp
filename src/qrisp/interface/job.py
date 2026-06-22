# """
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
# """

"""
Abstract :class:`Job` interface and related types for representing and managing backend executions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backend import Backend


class JobStatus(StrEnum):
    """
    Enumeration of possible lifecycle states for a :class:`Job`.

    Attributes
    ----------

    INITIALIZING :
        The job object has been created but execution has not yet been handed off to the
        backend. This is a transient state: a correctly implemented backend must ensure
        that every job exits ``INITIALIZING`` before :meth:`~qrisp.interface.Backend.run_async`
        returns.
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

    @classmethod
    def final_states(cls) -> tuple[JobStatus, ...]:
        """Terminal states: once a Job reaches one of these, its state won't change anymore."""
        return (cls.DONE, cls.CANCELLED, cls.ERROR)


JOB_FINAL_STATES = JobStatus.final_states()


class JobFailureError(RuntimeError):
    """
    Raised by :meth:`Job.result` when the job terminated in
    :attr:`~JobStatus.ERROR` state.
    """


class JobCancelledError(RuntimeError):
    """
    Raised by :meth:`Job.result` when the job was cancelled
    (:attr:`~JobStatus.CANCELLED`).
    """


class JobResult:
    """
    Wraps the outcome of one or more circuit executions.

    Counts are stored as a list with one dictionary per input circuit,
    preserving the input order.

    Parameters
    ----------
    counts : list[dict]
        A list of measurement-outcome dictionaries, one per circuit.
        Keys and values can be of any type, as determined by the user.

    **kwargs
        Optional backend-specific metadata associated with the execution.

    Examples
    --------
    Let's start with a simple example of a single circuit result:

    >>> from qrisp.interface.job import JobResult
    >>> result = JobResult([{"00": 512, "11": 512}])
    >>> result.get_counts()
    {"00": 512, "11": 512}

    For batch executions, we can store results for multiple circuits in a single object:

    >>> result = JobResult([{"0": 800, "1": 224}, {"00": 500, "11": 524}])
    >>> result.all_counts
    [{'0': 800, '1': 224}, {'00': 500, '11': 524}]

    Even in this case, we can access individual circuit results using the :meth:`get_counts` method:

    >>> result.get_counts(1)
    {'00': 500, '11': 524}

    Note that we can retrieve the number of circuits in the result using the :attr:`num_circuits` property:

    >>> result.num_circuits
    2

    Finally, we can print the result object to see a summary of its contents:

    >>> print(result)
    JobResult(num_circuits=2, metadata={})
    """

    def __init__(self, counts: Sequence[dict], **kwargs):
        """Initialize a JobResult instance."""

        if not isinstance(counts, Sequence) or isinstance(counts, (str, bytes)):
            raise TypeError(
                f"'counts' must be a sequence of dicts, got {type(counts).__name__}"
            )

        if len(counts) == 0:
            raise ValueError("'counts' must contain at least one dict")

        for i, count in enumerate(counts):
            if not isinstance(count, dict):
                raise TypeError(
                    f"Each item in 'counts' must be a dict, got {type(count).__name__} at index {i}"
                )

        self._counts = list(counts)
        self.metadata = kwargs

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def all_counts(self) -> list[dict]:
        """All measurement-outcome counts, one dict per circuit."""
        return self._counts

    @property
    def num_circuits(self) -> int:
        """Number of circuits whose results are stored in this object."""
        return len(self._counts)

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def get_counts(self, index: int = 0) -> dict:
        """
        Return the measurement-outcome counts for a single circuit.

        Parameters
        ----------
        index : int
            Index of the circuit in the batch. Defaults to ``0``.

        Returns
        -------
        dict
            Mapping from bitstring outcome to observed count.

        """
        try:
            return self._counts[index]
        except IndexError as exc:
            raise IndexError(
                f"Result contains {len(self._counts)} circuit(s); "
                f"index {index} is out of range."
            ) from exc

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"JobResult(num_circuits={self.num_circuits}, metadata={self.metadata!r})"
        )


class Job(ABC):
    """
    Abstract handle for a (potentially asynchronous) backend execution.

    A ``Job`` is returned by :meth:`Backend.run_async` immediately after
    submission. The caller can then:

    * call :meth:`result` to wait for the outcome and retrieve it, or
    * poll :attr:`status` / :meth:`done` / :meth:`running` / :meth:`queued` / :meth:`cancelled`
      to check the current state, or
    * call :meth:`cancel` to attempt cancellation.

    This follows the *Future* (or *Promise*) pattern: execution happens
    independently of the caller, and the caller decides *when* to wait for
    the outcome.

    .. rubric:: Design contract

    The ``Job`` base class defines *only the observable contract*. That is, the three
    abstract methods that every concrete job must implement. It deliberately
    prescribes no internal synchronisation mechanism (no threading, no
    asyncio, no polling loop). Each concrete subclass chooses whatever
    concurrency model suits its execution environment:

    * A synchronous simulator may compute the result inline and make it
      available before :meth:`result` is ever called.

    * An asynchronous hardware backend may submit to a remote queue and
      poll in a background thread until the result is ready.

    From the caller's perspective, all of the above are used identically.

    Parameters
    ----------
    backend : Backend
        The backend that created this job.

    job_id : str or None
        Optional caller-supplied identifier (e.g. a remote job ID assigned
        by a vendor API). If ``None``, the concrete subclass is responsible
        for assigning one.

    **kwargs
        Additional backend-specific metadata to associate with this job.
        It can be used by any backend implementation as needed.

    """

    def __init__(self, backend: Backend, job_id: str | None = None, **kwargs):
        """Initialize a Job instance."""

        self._backend: Backend = backend
        self._job_id: str | None = job_id
        self._last_known_status: JobStatus = JobStatus.INITIALIZING
        self._failure_cause: BaseException | None = None
        self.metadata: dict = kwargs

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

    @property
    def last_known_status(self) -> JobStatus:
        """
        The most recently cached :class:`JobStatus` for this job.

        This value is initialised to :attr:`~JobStatus.INITIALIZING` and is
        updated at the following points:

        * :meth:`status`: every live query updates the cache as a side effect.
        * :meth:`result`: transitions to ``DONE``, ``CANCELLED``, or
          ``ERROR`` when the job reaches a terminal state.
        * :meth:`refresh`: explicitly fetches the live status and caches it
          (semantically identical to calling :meth:`status`, but signals intent
          clearly when the sole purpose is to refresh the cache).
        """
        return self._last_known_status

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def result(self, timeout: float | None = None) -> JobResult:
        """
        Block until the job reaches a terminal state and return its result.

        This is a blocking call: it does not return until the job is in one
        of the terminal states: :attr:`~JobStatus.DONE`,
        :attr:`~JobStatus.CANCELLED`, or :attr:`~JobStatus.ERROR`. The
        internal synchronisation mechanism (threading, polling, asyncio,
        etc.) is left entirely to the concrete implementation.

        If the job is already in a terminal state when this method is
        called, it must return (or raise) immediately without blocking.

        Concrete implementations should call :meth:`_raise_for_status`
        once the terminal state has been reached, before returning.

        Parameters
        ----------
        timeout : float or None, optional
            Maximum number of seconds to wait for the job to finish.
            ``None`` (default) waits indefinitely. Implementations that
            support this parameter should raise ``TimeoutError`` when
            the deadline expires.

        Returns
        -------
        JobResult
            The measurement results, if the job completed successfully
            (:attr:`~JobStatus.DONE`).

        Raises
        ------
        JobFailureError
            If the job terminated in :attr:`~JobStatus.ERROR` state.

        JobCancelledError
            If the job was cancelled (:attr:`~JobStatus.CANCELLED`).

        TimeoutError
            If *timeout* is specified and the job does not finish in time.
        """

        raise NotImplementedError

    @abstractmethod
    def cancel(self) -> bool:
        """
        Attempt to cancel the job.

        Returns
        -------
        bool
            ``True`` if the cancellation was initiated: either the job is
            already in :attr:`~JobStatus.CANCELLED` state (for synchronous
            or in-process backends), or the cancel request has been
            dispatched to the remote backend (for asynchronous hardware
            backends, the transition to ``CANCELLED`` may not yet be
            visible in :meth:`status`).

            ``False`` if the job is already in a terminal state and no
            action was taken.

            .. note::

                For asynchronous remote backends, ``True`` means the
                cancel *request* was accepted by the server, not that
                the job has confirmed cancellation.  Callers should check
                :meth:`status` (or call :meth:`result`) to determine the
                final outcome.

        """

        raise NotImplementedError

    @abstractmethod
    def status(self) -> JobStatus:
        """
        Query and return the current :class:`JobStatus` of the job.

        This is a *live query*: every call fetches the most up-to-date
        status available, which for remote backends may involve a network
        call. As a side effect, concrete implementations should store the
        result in ``_last_known_status`` before returning, so that
        :attr:`last_known_status` always reflects the most recently
        observed state.

        Callers that want to avoid the cost of a live query should read
        :attr:`last_known_status` directly. Call :meth:`refresh` (or
        :meth:`status`) when an explicit update is needed.

        Returns
        -------
        JobStatus

        """

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Methods (derived from status)
    #
    # NOTE: every method in this block calls :meth:`status` on each
    # invocation, which for remote backends may involve a network
    # round-trip. Callers that need to minimise round-trips should read
    # :attr:`last_known_status` directly after a single :meth:`status`
    # or :meth:`refresh` call rather than using these helpers repeatedly.
    # ------------------------------------------------------------------

    def done(self) -> bool:
        """
        Return ``True`` if the job has completed successfully and results are available.
        """
        return self.status() == JobStatus.DONE

    def running(self) -> bool:
        """Return ``True`` if the job is currently executing."""
        return self.status() == JobStatus.RUNNING

    def queued(self) -> bool:
        """Return ``True`` if the job is waiting in the backend queue."""
        return self.status() == JobStatus.QUEUED

    def cancelled(self) -> bool:
        """Return ``True`` if the job was cancelled."""
        return self.status() == JobStatus.CANCELLED

    def in_final_state(self) -> bool:
        """
        Return ``True`` if the job has reached a terminal state.

        Terminal states are :attr:`~JobStatus.DONE`,
        :attr:`~JobStatus.CANCELLED`, and :attr:`~JobStatus.ERROR`.
        """
        return self.status() in JOB_FINAL_STATES

    def refresh(self) -> JobStatus:
        """
        Fetch the latest status from the backend and return it.

        Delegates to :meth:`status`, which already updates
        :attr:`last_known_status` as a side effect.  This method exists as
        a named alias that makes the intent explicit: the caller wants to
        refresh the cached status, not merely query it.

        Returns
        -------
        JobStatus
            The freshly fetched status.
        """
        return self.status()

    # ------------------------------------------------------------------
    # Helper for concrete implementations
    # ------------------------------------------------------------------

    def _raise_for_status(self, status: JobStatus | None = None) -> None:
        """
        Raise the appropriate exception if the job did not complete successfully.

        Concrete implementations should call this inside :meth:`result`
        *after* the job has reached a terminal state (i.e. after any
        blocking wait).

        Parameters
        ----------
        status : JobStatus or None, optional
            The terminal :class:`JobStatus` that was already determined by
            the caller's wait mechanism. When provided, this value is used
            directly and no additional :meth:`status` call is made: which
            avoids a redundant network round-trip on remote backends.
            If omitted (or ``None``), :meth:`status` is called to obtain
            the current state.

        Raises
        ------
        JobFailureError
            If *status* is ``ERROR``.

        JobCancelledError
            If *status* is ``CANCELLED``.
        """
        if status is None:
            status = self.status()
        if status == JobStatus.ERROR:
            if self._failure_cause is not None:
                raise JobFailureError(
                    f"Job {self._job_id!r} failed: {self._failure_cause}"
                ) from self._failure_cause
            raise JobFailureError(f"Job {self._job_id!r} terminated with status ERROR.")
        if status == JobStatus.CANCELLED:
            raise JobCancelledError(f"Job {self._job_id!r} was cancelled.")

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        # Use last_known_status rather than calling status() here.
        # status() is a live query that may involve a network round-trip
        # on remote backends, and triggering it inside __repr__ (e.g. during
        # logging, debugging, or pytest failure messages) would be
        # surprising and potentially expensive. Callers that want the
        # current status should call refresh() before printing.
        return (
            f"{self.__class__.__name__}("
            f"job_id={self._job_id!r}, "
            f"status={self._last_known_status.name})"
        )
