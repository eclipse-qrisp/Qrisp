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
Abstract :class:`Job` interface and related types for representing and managing backend executions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
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

    @classmethod
    def final_states(cls) -> tuple[JobStatus, ...]:
        """Terminal states: once a Job reaches one of these, its state won't change anymore."""
        return (cls.DONE, cls.CANCELLED, cls.ERROR)


JOB_FINAL_STATES = JobStatus.final_states()


class JobResult:
    """
    Wraps the outcome of one or more circuit executions.

    Counts are stored as a list with one dictionary per input circuit,
    preserving the input order.

    Parameters
    ----------
    counts : List[Dict]
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

    def __init__(self, counts: Sequence[Mapping], **kwargs):
        """Initialize a JobResult instance."""

        if not isinstance(counts, list):
            raise TypeError(
                f"'counts' must be a list of dicts, got {type(counts).__name__}"
            )

        self._counts = counts
        self.metadata = kwargs

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def all_counts(self) -> Sequence[Mapping]:
        """All measurement-outcome counts, one dict per circuit."""
        return self._counts

    @property
    def num_circuits(self) -> int:
        """Number of circuits whose results are stored in this object."""
        return len(self._counts)

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def get_counts(self, index: int = 0) -> Mapping:
        """
        Return the measurement-outcome counts for a single circuit.

        Parameters
        ----------
        index : int
            Index of the circuit in the batch. Defaults to ``0``.

        Returns
        -------
        Dict
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
            f"JobResult(num_circuits={self.num_circuits}, "
            f"metadata={self.metadata!r})"
        )


class Job(ABC):
    """
    Abstract handle for a (potentially asynchronous) backend execution.

    A ``Job`` is returned by :meth:`Backend.run` immediately after
    submission. The caller can then:

    * call :meth:`result` to wait for the outcome and retrieve it, or
    * poll :attr:`status` / :meth:`done` / :meth:`running` / :meth:`queued` / :meth:`cancelled`
      to check the current state, or
    * call :meth:`cancel` to attempt cancellation.

    This follows the *Future* (or *Promise*) pattern: execution happens
    independently of the caller, and the caller decides *when* to wait for
    the outcome.

    .. rubric:: Design contract

    The ``Job`` base class defines *only the observable contract*. That is, the four
    abstract methods that every concrete job must implement. It deliberately
    prescribes no internal synchronisation mechanism (no threading, no
    asyncio, no polling loop). Each concrete subclass chooses whatever
    concurrency model suits its execution environment:

    * A synchronous simulator may compute the result inline and make it
      available before :meth:`result` is ever called.

    * An asynchronous hardware backend may submit to a remote queue and
      poll in a background thread until the result is ready.

    From the caller's perspective, all of the above are used identically.

    .. rubric:: Relationship to design patterns

    ``Job`` is a `Virtual Proxy <https://refactoring.guru/design-patterns/proxy>`_ for
    the execution result: it is a local placeholder that controls access to
    an outcome that may be remote, deferred, or not yet computed.
    Together with :class:`Backend`, it forms a `Bridge <https://refactoring.guru/design-patterns/bridge>`_: the
    two abstract hierarchies (submission interface and execution handle)
    can be varied and extended independently.


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
        Included for qiskit compatibility, but may be used by any backend implementation as needed.

    """

    def __init__(self, backend: "Backend", job_id: str | None = None, **kwargs):
        """Initialize a Job instance."""

        self._backend = backend
        self._job_id = job_id
        self.metadata = kwargs

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
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def submit(self) -> None:
        """
        Submit the job to the backend for execution.

        This method triggers the actual execution. It is called by the
        backend after the job object has been constructed.
        """

        raise NotImplementedError

    @abstractmethod
    def result(self) -> JobResult:
        """
        Returns the :class:`JobResult` of this job.

        The internal mechanism used to wait for completion is left entirely
        to the concrete implementation.

        Returns
        -------
        JobResult

        """

        raise NotImplementedError

    @abstractmethod
    def cancel(self) -> bool:
        """
        Attempt to cancel the job.

        Returns
        -------
        bool
            True iff the job was successfully cancelled.

        """

        raise NotImplementedError

    @abstractmethod
    def status(self) -> JobStatus:
        """
        Return the current :class:`JobStatus` of the job.

        The returned status should reflect the most up-to-date information available to the backend,
        and it should be among the values of the :class:`JobStatus` enumeration.

        Returns
        -------
        JobStatus

        """

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Methods (derived from status)
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

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"job_id={self._job_id!r}, "
            f"status={self.status().name})"
        )
