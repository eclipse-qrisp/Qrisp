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

"""This module defines the abstract :class:`Backend` interface for Qrisp-compatible backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, overload, runtime_checkable

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.interface.measurement_result import MeasurementResult

from .job import Job

if TYPE_CHECKING:
    from qrisp.interface.batched_backend import BatchedBackend


# This protocol is required because `BatchedBackend` intentionally
# does not inherit from `Backend`, as this would violate the Liskov
# Substitution Principle (the `run` method in `BatchedBackend`
# has different behaviour and return type than in `Backend`).
@runtime_checkable
class BackendLike(Protocol):
    """Structural protocol satisfied by both :class:`Backend` and
    :class:`~qrisp.interface.BatchedBackend`.

    Use this as the type hint for parameters that accept either a concrete
    backend or a :class:`~qrisp.interface.BatchedBackend`.
    Both :class:`Backend` and :class:`~qrisp.interface.BatchedBackend`
    satisfy this protocol structurally (i.e. without explicit inheritance),
    so type checkers accept either wherever :class:`BackendLike` is required.

    """

    @property
    def name(self) -> str:
        """Human-readable name of the backend."""
        ...

    @overload
    def run(self, circuits: QuantumCircuit, shots: int | None = None) -> MeasurementResult: ...

    @overload
    def run(self, circuits: Sequence[QuantumCircuit], shots: int | None = None) -> list[MeasurementResult]: ...

    def run(
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        shots: int | None = None,
    ) -> MeasurementResult | list[MeasurementResult]:
        """Submit circuits and return measurement result(s)."""
        ...

    @property
    def options(self) -> Mapping:
        """Current runtime options (read-only)."""
        ...

    def update_options(self, **kwargs) -> None:
        """Update existing runtime options."""
        ...


class Backend(ABC):
    """Abstract base class for Qrisp-compatible backends.

    This class provides a minimal, hardware-agnostic interface that all
    backends (simulators or quantum hardware clients) must follow.

    The only mandatory method for child classes is :meth:`run_async`, which
    submits one or more circuits for execution and returns a :class:`Job`
    handle immediately. The caller then decides when to wait for the result
    by calling :meth:`Job.result <qrisp.interface.Job.result>`.

    A synchronous :meth:`run` method is also provided. It calls
    :meth:`run_async` internally, blocks until the job completes, and returns
    the results as :class:`~qrisp.interface.MeasurementResult` objects. This
    is the standard execution path for most users, and is what
    :meth:`~qrisp.QuantumVariable.get_measurement` uses internally. Use
    :meth:`run_async` directly only when you need the :class:`Job` handle
    itself (e.g. to poll status, cancel, or wait concurrently).

    .. rubric:: Design contract

    The ``Backend`` class defines the minimal execution interface required by
    Qrisp. It does not assume universality of the gate set, specific
    connectivity constraints, or the presence of calibration data. Any such
    assumptions must be made explicit by concrete backend implementations or
    higher-level compilation policies.

    This class intentionally avoids prescribing compilation, scheduling, or
    execution policies, which are delegated to concrete backends or external
    components.

    For example, simulators are expected to implement :meth:`run_async` but may
    return ``None`` for all hardware metadata properties.

    .. rubric:: Execution model

    :meth:`run_async` accepts a single circuit *or* a sequence of circuits
    and always returns a :class:`Job` instance immediately.
    This follows the *Future* pattern: execution happens independently of the
    caller, and the caller decides *when* to block by calling
    :meth:`Job.result <qrisp.interface.Job.result>`.

    :meth:`run` is the standard synchronous execution path.
    It blocks until execution completes and returns the measurement results
    (one per submitted circuit).

    .. rubric:: Runtime options

    Runtime options describe *how* the backend executes circuits (e.g. the
    number of shots). These can be provided:

    * by overriding :meth:`_default_options` at class level, or
    * by passing a custom ``options`` mapping to :meth:`__init__`.

    The number of shots may also be overridden per-execution by passing
    the ``shots`` argument to :meth:`run_async` or :meth:`run`. It is
    treated as a conventional execution parameter for gate-based, shot-based
    backends, and may be ignored by backends for which it is not meaningful.

    Options are updated through :meth:`update_options`, but only keys that
    were present at initialisation may be modified.

    .. rubric:: Hardware metadata

    The backend may optionally expose hardware metadata such as:

    * backend health or diagnostics,
    * backend queue status,
    * number of qubits,
    * connectivity information,
    * supported native gates,
    * gate errors or calibration data.

    These properties are intentionally left loosely typed at the base-class
    level. Their purpose is to *signal the presence of a capability*, not to
    enforce a universal hardware schema.

    The absence of a value for a given property (``None``) explicitly means
    *"capability not exposed by this backend"*. This is common for simulators,
    abstract backends, or backends that choose not to provide hardware details.

    Concrete backend implementations (e.g. vendor-specific backends) are
    expected to override these properties and may return structured,
    vendor-defined objects with richer semantics. Transpilers or compilation
    passes may then either:

    * require specific capabilities and raise if they are missing, or
    * ignore hardware metadata entirely and operate in a backend-agnostic mode.

    Backend-specific capabilities that are not covered by the base interface
    should be exposed as concrete typed properties on the subclass.

    Parameters
    ----------
    name : str or None
        Optional user-defined name for the backend.
        Defaults to the class name.

    options : Mapping or None
        Runtime execution options for the backend.
        If omitted, :meth:`_default_options` is used.

    **kwargs :
        Additional backend-specific parameters. These are not defined at the
        base-class level but may be accepted by concrete backend
        implementations (e.g. for authentication, provider selection, or
        other configuration).

    """

    def __init__(self, name: str | None = None, options: Mapping | None = None, **kwargs):
        """Initialise the backend."""
        self.name = name or self.__class__.__name__

        if options is None:
            options = self._default_options()

        if not isinstance(options, Mapping):
            raise TypeError(f"'options' must be a dict-like Mapping, got {type(options).__name__}")

        # Shallow-copy and convert to dict so that:
        # (a) external mutations of the original mapping do not affect the
        #     backend's internal state, and
        # (b) update_options can use __setitem__ regardless of the original
        #     Mapping type.
        self._options = dict(options)

        self.metadata = kwargs

    # ------------------------------------------------------------------
    # Abstract method
    # ------------------------------------------------------------------

    @abstractmethod
    def run_async(
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        shots: int | list[int] | None = None,
    ) -> Job:
        """Submit one or more circuits for execution and return a :class:`Job`.

        The returned job must not be in
        :attr:`~qrisp.interface.JobStatus.INITIALIZING` state: by the time
        ``run_async`` returns, execution must have been handed off to the
        backend. The exact state the job enters depends on the backend type:

        * :attr:`~qrisp.interface.JobStatus.QUEUED`: for asynchronous or remote
          backends where the job waits in a queue before execution begins.

        * :attr:`~qrisp.interface.JobStatus.RUNNING` or
          :attr:`~qrisp.interface.JobStatus.DONE`: for synchronous simulators
          that begin (or complete) execution before returning.

        Parameters
        ----------
        circuits : QuantumCircuit or Sequence[QuantumCircuit]
            A single circuit or a sequence of circuits to execute.
            No validation or introspection is performed at the base-class level.

            When a sequence is provided, the backend decides internally how to
            handle the circuit execution. Hardware backends may impose a limit
            on how many circuits a single job may contain. This is a
            backend-defined constraint.

        shots : int or list[int] or None, optional
            Number of shots (repetitions) for the execution. If ``None``,
            the value from the backend's runtime options should be used.
            When a ``list[int]`` is provided, each entry specifies the shot
            count for the circuit at the corresponding index.

            Backends whose SDK does not natively support per-circuit shot
            counts should fall back to ``max(shots)`` and issue a
            ``UserWarning``.

        Returns
        -------
        Job
            A handle to the submitted execution. Call
            :meth:`Job.result <qrisp.interface.Job.result>` to wait for
            completion and retrieve the
            :class:`~qrisp.interface.JobResult`.

        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Synchronous convenience method
    # ------------------------------------------------------------------

    @overload
    def run(self, circuits: QuantumCircuit, shots: int | None = None) -> MeasurementResult: ...

    @overload
    def run(self, circuits: Sequence[QuantumCircuit], shots: int | None = None) -> list[MeasurementResult]: ...

    def run(
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        shots: int | None = None,
    ) -> MeasurementResult | list[MeasurementResult]:
        """Submit one or more circuits, block until completion, and return results.

        This is a synchronous convenience wrapper around :meth:`run_async`.
        It calls :meth:`run_async`, waits for the :class:`Job` to finish, and
        returns the measurement results wrapped in
        :class:`~qrisp.interface.MeasurementResult` objects.  The result type
        mirrors the input: a single :class:`~qrisp.interface.MeasurementResult`
        for a single circuit, or a ``list`` of them for a sequence.

        Parameters
        ----------
        circuits : QuantumCircuit or Sequence[QuantumCircuit]
            A single circuit or a sequence of circuits to execute.

        shots : int or None, optional
            Number of shots. If ``None``, the backend's ``shots`` option
            is used by default.

        Returns
        -------
        MeasurementResult or list[MeasurementResult]
            A pre-populated :class:`~qrisp.interface.MeasurementResult` when
            one circuit is submitted, or a list of them for multiple circuits.

        Raises
        ------
        TypeError
            If *shots* is not an integer.

        ValueError
            If *shots* is not a positive integer.

        """
        self._validate_shots(shots)
        self._check_circuit_limit(circuits)
        batch = not isinstance(circuits, QuantumCircuit)
        shots = shots if shots is not None else self.options.get("shots")
        all_counts = self.run_async(circuits, shots).result().all_counts

        results = []
        for counts in all_counts:
            raw = MeasurementResult()
            raw._inject(counts)
            results.append(raw)

        return results if batch else results[0]

    # ------------------------------------------------------------------
    # Optional job retrieval and utility methods
    # ------------------------------------------------------------------

    def batched(self) -> "BatchedBackend":
        """Return a :class:`BatchedBackend` that wraps this backend.

        The returned object buffers circuits submitted via :meth:`run` and
        executes them by forwarding each to this backend's :meth:`run` when
        :meth:`BatchedBackend.dispatch` is called.

        Returns
        -------
        BatchedBackend

        """
        from qrisp.interface.batched_backend import BatchedBackend

        return BatchedBackend(self)

    def retrieve_job(self, job_id: str) -> Job:
        """Reconnect to a previously submitted job by its identifier.

        This method allows users to recover a :class:`Job` handle after a
        process restart or network interruption, provided the backend
        stores job history server-side.

        This is an *optional capability*. The default implementation
        raises :exc:`NotImplementedError`. Backends that support job
        recovery must override this method.

        Parameters
        ----------
        job_id : str
            The identifier of the job to retrieve. This is the value
            returned by :attr:`Job.job_id <qrisp.interface.Job.job_id>`
            after a previous call to :meth:`run_async`.

        Returns
        -------
        Job
            A handle to the previously submitted job, in whatever state
            it currently is.

        Raises
        ------
        NotImplementedError
            If this backend does not support job recovery (the default).

        LookupError
            If no job with the given *job_id* can be found on the backend.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support job recovery. Override retrieve_job() to enable this feature."
        )

    def _check_circuit_limit(
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
    ) -> None:
        """Raise :exc:`ValueError` if the number of submitted circuits exceeds
        :attr:`max_circuits`.

        This helper is called automatically by :meth:`run`. Implementations
        of :meth:`run_async` can call it before submitting to the
        hardware so that users who bypass :meth:`run` get the same early
        error.

        Parameters
        ----------
        circuits : QuantumCircuit or Sequence[QuantumCircuit]
            The circuit(s) about to be submitted.

        Raises
        ------
        ValueError
            If *circuits* is a sequence whose length exceeds
            :attr:`max_circuits`.

        """
        limit = self.max_circuits
        if limit is None:
            return
        if isinstance(circuits, QuantumCircuit):
            return  # a single circuit is always within any positive limit
        try:
            n = len(circuits)
        except TypeError:
            return  # length unknown (e.g. a lazy iterable). We skip the check
        if n > limit:
            raise ValueError(
                f"{self.__class__.__name__} accepts at most {limit} "
                f"circuit(s) per job, but {n} were submitted. "
                "Split the batch into smaller chunks and submit separately."
            )

    @staticmethod
    def _validate_shots(shots: int | None) -> None:
        """Raise an informative error if *shots* is not a valid positive integer.

        Intended to be called at the top of :meth:`run` (and optionally
        inside :meth:`run_async` implementations) before submitting circuits
        to the backend.

        Parameters
        ----------
        shots : int or None
            The shot count to validate. ``None`` is accepted and means
            "use the backend default". No error is raised.

        Raises
        ------
        TypeError
            If *shots* is not an :class:`int` (or is a :class:`bool`,
            which is a subclass of :class:`int` but almost certainly a
            caller mistake).

        ValueError
            If *shots* is zero or negative.

        """
        if shots is None:
            return
        if isinstance(shots, bool) or not isinstance(shots, int):
            raise TypeError(f"'shots' must be a positive integer, got {type(shots).__name__!r}")
        if shots <= 0:
            raise ValueError(f"'shots' must be a positive integer, got {shots!r}")

    @staticmethod
    def _validate_shots_length(
        shots: list[int],
        circuits: "QuantumCircuit | Sequence[QuantumCircuit]",
    ) -> None:
        """Raise :exc:`ValueError` if the length of a per-circuit *shots* list
        does not match the number of submitted circuits.

        This method can be called inside :meth:`run_async` implementations
        immediately after normalising *circuits* to a list and before any execution,
        whenever *shots* is a ``list``.

        Parameters
        ----------
        shots : list[int]
            The per-circuit shot counts to validate.

        circuits : QuantumCircuit or Sequence[QuantumCircuit]
            The circuits that will be executed.  A single
            :class:`~qrisp.circuit.QuantumCircuit` counts as one circuit.

        Raises
        ------
        ValueError
            If ``len(shots)`` does not equal the number of circuits.

        """
        n = 1 if isinstance(circuits, QuantumCircuit) else len(circuits)
        if len(shots) != n:
            raise ValueError(
                f"'shots' list has {len(shots)} element(s) but {n} circuit(s) "
                "were submitted. When passing per-circuit shot counts, provide "
                "exactly one value per circuit."
            )

    # ------------------------------------------------------------------
    # Runtime options
    # ------------------------------------------------------------------

    @classmethod
    def _default_options(cls) -> Mapping[str, Any]:
        """Default runtime options for the backend.

        Child classes may override this method to provide custom default
        options, or the defaults may be overridden entirely by passing an
        ``options`` mapping to the constructor.

        Returns
        -------
        Mapping[str, Any]
            A mapping of option names to their default values.

        """
        return {"shots": 1024}

    @property
    def options(self) -> Mapping[str, Any]:
        """Current runtime options for the backend.

        These options may influence execution behaviour (e.g. the number of
        shots) and therefore may affect :meth:`run_async` and :meth:`run`.

        The returned mapping is read-only. Use :meth:`update_options` to
        change existing keys. Direct mutation (e.g.
        ``backend.options["shots"] = n``) raises :exc:`TypeError`.
        """
        return MappingProxyType(self._options)

    def update_options(self, **kwargs) -> None:
        """Update existing runtime options for the backend.

        Only keys that were present at initialisation (i.e. defined in
        :meth:`_default_options` or the ``options`` argument passed to the
        constructor) may be updated. Attempting to set an unknown key raises
        an :exc:`AttributeError`.

        This method is *atomic*: all keys are validated before any value
        is written. If one key is invalid, no options are changed.

        Parameters
        ----------
        **kwargs :
            Key-value pairs to update in the backend's runtime options.

        """
        unknown = [k for k in kwargs if k not in self._options]
        if unknown:
            raise AttributeError(
                f"'{unknown[0]}' is not a valid backend option for "
                f"{self.__class__.__name__}. "
                f"Valid options: {list(self._options.keys())}"
            )
        if "shots" in kwargs:
            self._validate_shots(kwargs["shots"])
        self._options.update(kwargs)

    # ------------------------------------------------------------------
    # Hardware / backend-specific status information
    # ------------------------------------------------------------------

    @property
    def health(self):
        """Current health status or diagnostics of the backend.

        This may include uptime statistics or other backend-specific health
        indicators.

        Returns ``None`` if the backend does not expose health information.
        """
        return None

    @property
    def info(self):
        """General information about the backend.

        This may include backend version, provider details, or other
        backend-specific information.

        Returns ``None`` if the backend does not expose such information.
        """
        return None

    @property
    def queue(self):
        """Current queue status or job backlog of the backend.

        This may include estimated wait times, number of pending jobs, or
        other backend-specific queue indicators.

        Returns ``None`` if the backend does not expose queue information.
        """
        return None

    # ------------------------------------------------------------------
    # Hardware / backend-specific metadata
    # ------------------------------------------------------------------

    @property
    def num_qubits(self):
        """Total number of physical qubits the backend exposes.

        This reflects the full physical qubit count of the device, not a
        snapshot of currently healthy or calibrated qubits. A qubit whose
        calibration has degraded or whose gates have been removed from the
        active gate set should still be counted here.

        Returns ``None`` if the backend does not expose a fixed or meaningful
        qubit count (e.g. simulators, abstract backends, or backends where
        this information is intentionally omitted).
        """
        return None

    @property
    def max_circuits(self) -> int | None:
        """Maximum number of circuits this backend can execute in a single job.

        Many hardware backends impose a per-job circuit limit (e.g. a
        cloud provider may accept at most 300 circuits per submission).
        Submitting a batch larger than this limit causes an opaque error
        from the vendor SDK. Exposing the limit here lets Qrisp provide an
        early, clear :exc:`ValueError` via :meth:`run` and
        :meth:`_check_circuit_limit` before any network call is made.

        Concrete backends should override this property and return the
        actual limit. Implementations of :meth:`run_async` should also
        call :meth:`_check_circuit_limit` before submitting to the
        hardware so that users who bypass :meth:`run` still get the same
        early error.

        Returns ``None`` if the backend imposes no limit (simulators) or
        does not expose this information.
        """
        return None

    @property
    def connectivity(self):
        """Currently executable qubit connectivity for the backend.

        This property describes which qubit pairs currently
        have at least one multi-qubit gate available for execution. It
        reflects the active gate set at the time the property is queried
        (not the physical wiring of the device, not a theoretical maximum,
        and not a snapshot of all pairs that have ever been calibrated).

        If the physical device consists of multiple disconnected components,
        the backend is expected to expose a single connected component.
        The choice of which component to expose (e.g. largest component,
        highest-fidelity subset) is left to the concrete backend
        implementation.

        Gates involving more than two qubits are not representable as edges
        in a connectivity graph. Backends that support such operations should
        document them separately (e.g. via :attr:`gate_set`).

        The returned object may encode qubit adjacency, coupling constraints,
        or other topology information in a backend-specific format.

        Returns ``None`` if the backend does not expose connectivity
        information. Concrete backend implementations may return structured,
        vendor-defined objects.
        """
        return None

    @property
    def gate_set(self):
        """Native gate set supported by the backend.

        This property describes which operations the backend can execute
        natively. The gate set is purely descriptive.
        Qrisp does not assume the set is universal. Compilation passes that
        require universality must verify this themselves.

        Different qubit pairs may support different gates (e.g. CZ on some
        pairs, iSWAP on others, or both). Backends are expected to encode
        this granularity in the returned object.

        Measurement is not assumed. The availability of a measurement
        operation on a given qubit is not guaranteed by this interface.
        Backends that require explicit measurement declarations should
        include them in the gate set.

        The format of the returned object is backend-specific. For hardware
        backends it typically refers to native operations. For simulators it
        may be omitted or ignored.

        Returns ``None`` if the backend does not expose gate availability.
        """
        return None

    @property
    def error_rates(self):
        """Error rates or calibration-related information for the backend.

        This property is calibration-dependent: the values it exposes are
        only meaningful relative to a specific calibration run. Concrete
        backend implementations currently must include metadata that identifies
        which calibration snapshot the data refers to. For example, a
        timestamp, a calibration ID, or a version string. Returning bare
        error values without any calibration anchor is discouraged,
        as callers have no way to determine whether the data is current.

        The format of the returned object is hardware (and vendor) specific.

        Returns ``None`` if the backend does not expose error or calibration
        data.
        """
        return None

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise string representation of the backend."""
        return f"{self.__class__.__name__}(name={self.name!r}, options={dict(self._options)!r})"
