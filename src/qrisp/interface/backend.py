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

"""This module defines the abstract :class:`Backend` interface for Qrisp-compatible backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .job import Job

if TYPE_CHECKING:
    from qrisp.circuit.quantum_circuit import QuantumCircuit


class Backend(ABC):
    """
    Abstract base class for Qrisp-compatible backends.

    This class provides a minimal, hardware-agnostic interface that all
    backends (simulators or quantum hardware clients) must follow.

    The only mandatory method for child classes is :meth:`run`, which submits
    one or more circuits for execution and returns a :class:`Job` handle
    immediately. The caller then decides when to wait for the result by
    calling :meth:`Job.result <qrisp.interface.Job.result>`.

    .. rubric:: Design contract

    The ``Backend`` class defines the minimal execution interface required by
    Qrisp. It does not assume universality of the gate set, specific
    connectivity constraints, or the presence of calibration data. Any such
    assumptions must be made explicit by concrete backend implementations or
    higher-level compilation policies.

    This class intentionally avoids prescribing compilation, scheduling, or
    execution policies, which are delegated to concrete backends or external
    components.

    For example, simulators are expected to implement ``run`` but may
    return ``None`` for all hardware metadata properties.

    .. rubric:: Execution model

    :meth:`run` accepts a single circuit *or* a sequence of circuits and always
    returns a :class:`Job` instance immediately.
    This follows the *Future* pattern: execution happens independently of the caller,
    and the caller decides *when* to block by calling :meth:`Job.result <qrisp.interface.Job.result>`.

    .. rubric:: Runtime options

    Runtime options describe *how* the backend executes circuits (e.g. the
    number of shots). These can be provided:

    * by overriding :meth:`_default_options` at class level, or
    * by passing a custom ``options`` mapping to :meth:`__init__`.

    The number of shots may also be overridden per-execution by passing
    the ``shots`` argument to :meth:`run`. It is treated as a conventional
    execution parameter for gate-based, shot-based backends, and may be
    ignored by backends for which it is not meaningful.


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

    .. rubric:: Relationship to design patterns

    ``Backend`` and :class:`Job` together form a `Bridge <https://refactoring.guru/design-patterns/bridge>`_: two
    independently varying class hierarchies (submission interface and execution handle)
    that can be extended without affecting each other. :class:`Job` is additionally
    a `Virtual Proxy <https://refactoring.guru/design-patterns/proxy>`_ for the execution result.
    Concrete vendor backends act as `Adapters <https://refactoring.guru/design-patterns/adapter>`_ that
    wrap a vendor SDK to satisfy the ``Backend`` interface expected by Qrisp.

    Parameters
    ----------
    name : str or None
        Optional user-defined name for the backend.
        Defaults to the class name.

    options : Mapping or None
        Runtime execution options for the backend.
        If omitted, :meth:`_default_options` is used.

    **kwargs :
        Additional backend-specific parameters. These are not defined at the base-class level
        but may be accepted by concrete backend implementations (e.g. for authentication,
        provider selection, or other configuration).

    """

    def __init__(
        self, name: str | None = None, options: Mapping | None = None, **kwargs
    ):
        """Initialise the backend."""
        self.name = name or self.__class__.__name__

        if options is None:
            options = self._default_options()

        if not isinstance(options, Mapping):
            raise TypeError(
                f"'options' must be a dict-like Mapping, "
                f"got {type(options).__name__}"
            )

        # We make a shallow copy to prevent external mutations from affecting the backend's internal state.
        # Furthermore, we convert to a dict to ensure that a `__setitem__`-capable mapping is stored,
        # which is required for `update_options`.
        self._options = dict(options)

        self.metadata = kwargs

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def run(
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        shots: int | None = None,
    ) -> Job:
        """
        Submit one or more circuits for execution and return a :class:`Job`.

        This method returns immediately. The caller blocks and retrieves the
        outcome by calling :meth:`Job.result <qrisp.interface.Job.result>`
        on the returned object.

        Parameters
        ----------
        circuits : QuantumCircuit or Sequence[QuantumCircuit]
            A single circuit or a sequence of circuits to execute.
            No validation or introspection is performed at the base-class level.

            When a sequence is provided, the backend decides internally whether
            to run the circuits sequentially or in parallel. Hardware
            backends may impose a limit on how many circuits a single job
            may contain. This is a backend-defined constraint.

        shots : int or None, optional
            Number of shots (repetitions) for the execution. If ``None``,
            the value from the backend's runtime options is used.

        Returns
        -------
        Job
            A handle to the submitted execution. Call :meth:`Job.result <qrisp.interface.Job.result>`
            to wait for completion and retrieve the :class:`qrisp.interface.JobResult`.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Runtime options
    # ------------------------------------------------------------------

    @classmethod
    def _default_options(cls) -> Mapping[str, Any]:
        """
        Default runtime options for the backend.

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
        """
        Current runtime options for the backend.

        These options may influence execution behaviour (e.g. the number of
        shots) and therefore may affect :meth:`run`.
        """
        return self._options

    def update_options(self, **kwargs) -> None:
        """
        Update existing runtime options for the backend.

        Only keys that were present at initialisation (i.e. defined in
        :meth:`_default_options` or the ``options`` argument passed to the
        constructor) may be updated. Attempting to set an unknown key raises
        an ``AttributeError``.

        Parameters
        ----------
        **kwargs :
            Key-value pairs to update in the backend's runtime options.

        """
        for key, val in kwargs.items():
            if key not in self._options:
                raise AttributeError(
                    f"'{key}' is not a valid backend option for "
                    f"{self.__class__.__name__}. "
                    f"Valid options: {list(self._options.keys())}"
                )
            self._options[key] = val

    # ------------------------------------------------------------------
    # Hardware / backend-specific status information
    # ------------------------------------------------------------------

    @property
    def backend_health(self):
        """
        Current health status or diagnostics of the backend.

        This may include uptime statistics or other backend-specific health
        indicators.

        Returns ``None`` if the backend does not expose health information.
        """
        return None

    @property
    def backend_info(self):
        """
        General information about the backend.

        This may include backend version, provider details, or other
        backend-specific information.

        Returns ``None`` if the backend does not expose such information.
        """
        return None

    @property
    def backend_queue(self):
        """
        Current queue status or job backlog of the backend.

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
        """
        Number of qubits available on the backend.

        Returns ``None`` if the backend does not expose a fixed or meaningful
        qubit count (e.g. simulators, abstract backends, or backends where
        this information is intentionally omitted).
        """
        return None

    @property
    def connectivity(self):
        """
        Connectivity information for the backend.

        This property describes the qubit connectivity exposed by the backend
        for compilation and execution purposes. If the physical device
        consists of multiple disconnected components, the backend is expected
        to expose **a single connected component**. The choice of which
        component to expose (e.g. largest component, highest-fidelity
        subset) is left to the concrete backend implementation.

        The returned object may encode qubit adjacency, coupling constraints,
        or other topology information in a backend-specific format.

        Returns ``None`` if the backend does not expose connectivity
        information. Concrete backend implementations may return structured,
        vendor-defined objects.
        """
        return None

    @property
    def gate_set(self):
        """
        Native gate set supported by the backend.

        The exact meaning of this property is backend-specific. For hardware
        backends this typically refers to native operations; for simulators
        it may be omitted or ignored.

        Returns ``None`` if the backend does not expose gate availability.
        """
        return None

    @property
    def error_rates(self):
        """
        Error rates or calibration-related information for the backend.

        This may depend on a specific calibration snapshot and is typically
        hardware- and vendor-specific.

        Returns ``None`` if the backend does not expose error or calibration
        data.
        """
        return None

    # ------------------------------------------------------------------
    # Extension point for backend-specific capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> Mapping[str, Any]:
        """
        Backend-specific capabilities not covered by the base interface.

        Keys and values are backend-defined. This property is intended as an
        extension point for vendor backends to expose additional structured
        information without modifying the base ``Backend`` API.

        This may include, for example, calibration schedules, advanced
        hardware features, quality metrics, or other backend-specific data.

        Returns an empty dict if the backend does not expose additional
        capabilities.
        """
        return {}

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"options={dict(self._options)!r})"
        )
