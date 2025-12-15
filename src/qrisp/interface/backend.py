"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import copy
from typing import Any

from qrisp.circuit.quantum_circuit import QuantumCircuit


class Backend(ABC):
    """
    Abstract base class for Qrisp-compatible backends.

    This class provides a minimal and hardware-agnostic interface that all
    backends (simulators or quantum hardware clients) must follow.

    The only mandatory method for child classes is :meth:`run`, which executes
    a circuit or operation on the backend.

    **Runtime Options**

    Runtime options describe *how* the backend executes circuits (e.g., the
    number of shots). These can be provided:

    * by overriding :meth:`_default_options` at class level, or
    * by passing a custom ``options`` mapping to the constructor.

    The ``options`` argument does not need to be a dict; any object satisfying
    the :class:`collections.abc.Mapping` interface is accepted. This allows
    interoperability with external frameworks such as Qiskit, which define
    their own structured ``Options`` classes.

    Options are updated through :meth:`update_options`, but only keys that were
    present at initialization may be modified.

    **Hardware Metadata**

    The backend may optionally expose hardware metadata such as:

    * backend health or diagnostics,
    * backend queue status,
    * number of qubits,
    * connectivity information,
    * supported native gates,
    * error rates or calibration data.

    These properties are intentionally left loosely typed at the base-class level.
    Their purpose is to *signal the presence of a capability*, not to enforce a
    universal hardware schema.

    A value of ``None`` for a given property explicitly means *“capability not
    exposed by this backend”*. This is common for simulators, abstract backends,
    or backends that choose not to provide hardware details.

    Concrete backend implementations (e.g., vendor-specific backends) are expected
    to override these properties and may return structured, vendor-defined objects
    with richer semantics. Transpilers or compilation passes may then either:

    * require specific capabilities and raise if they are missing, or
    * ignore hardware metadata entirely and operate in a backend-agnostic mode.

    This design keeps the base ``Backend`` interface fully hardware-agnostic while
    allowing vendor backends to expose detailed and typed hardware information.

    **Design Contract**

    The ``Backend`` class defines the minimal execution interface required by Qrisp.
    It does not assume universality of the gate set, specific connectivity
    constraints, or the presence of calibration data. Any such assumptions must be
    made explicit by concrete backend implementations or higher-level compilation
    policies.

    Simulators are expected to implement ``run`` but may return ``None``
    for all hardware metadata properties.

    The concrete type of hardware metadata might be backend-defined.


    Parameters
    ----------
    name : str or None
        Optional user-defined name for the backend.
        Defaults to the class name.

    options : Mapping or None
        Runtime execution options for the backend.
        If omitted, :meth:`_default_options` is used.

    """

    def __init__(self, name: str | None = None, options: Mapping | None = None):
        """
        Initialize the backend.

        Parameters
        ----------
        name : str or None
            Optional user-defined name for this backend.
            Defaults to the class name.

        options : Mapping or None
            Mapping of runtime execution options for the backend.
            If omitted, the backend uses the class-level default options from
            :meth:`_default_options`.

        """

        self.name = name or self.__class__.__name__

        if options is None:
            options = self._default_options()

        if not isinstance(options, Mapping):
            raise TypeError("options must be a dict-like mapping")

        self._options = copy(options)

    # ----------------------------------------------------------------------
    # Abstract interface
    # ----------------------------------------------------------------------

    # TODO: we will handle with batched execution later
    @abstractmethod
    def run(self, circuit: QuantumCircuit, *args, **kwargs) -> Any:
        """
        Execute a qrisp QuantumCircuit on the backend.

        The execution semantics of this method should not depend on the presence of optional hardware metadata.

        Parameters
        ----------
        circuit : QuantumCircuit
            The quantum circuit to be executed.
        *args :
            Additional positional arguments.
        **kwargs :
            Additional keyword arguments.

        Returns
        -------
        Any
            Backend-specific result of the execution.

        """
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Runtime options
    # ----------------------------------------------------------------------

    @classmethod
    def _default_options(cls) -> Mapping:
        """
        Default runtime options for the backend.

        Child classes may override this method to provide custom default options,
        or the defaults may be overridden entirely by passing an ``options``
        mapping to the constructor.
        """
        return {"shots": 1024}

    @property
    def options(self) -> Mapping:
        """Current runtime options."""
        return self._options

    def update_options(self, **kwargs) -> None:
        """Update existing runtime options, rejecting unknown keys."""

        for key, val in kwargs.items():
            if key not in self._options:
                raise AttributeError(
                    f"'{key}' is not a valid backend option for {self.__class__.__name__}. "
                    f"Valid options: {list(self._options.keys())}"
                )
            self._options[key] = val

    # ----------------------------------------------------------------------
    # Optional hardware/backend-specific metadata
    # ----------------------------------------------------------------------

    @property
    def backend_health(self) -> Any | None:
        """
        Current health status or diagnostics of the backend.

        This may include calibration data, uptime statistics, or other
        backend-specific health indicators.

        Returns ``None`` if the backend does not expose health information.
        """
        return None

    @property
    def backend_queue(self) -> Any | None:
        """
        Current queue status or job backlog of the backend.

        This may include estimated wait times, number of pending jobs,
        or other backend-specific queue indicators.

        Returns ``None`` if the backend does not expose queue information.
        """
        return None

    @property
    def num_qubits(self) -> int | None:
        """
        Number of qubits available on the backend.

        Returns ``None`` if the backend does not expose a fixed or meaningful qubit
        count (e.g., simulators, abstract backends, or backends where this
        information is intentionally omitted).
        """
        return None

    @property
    def connectivity(self) -> Any | None:
        """
        Connectivity information for the backend.

        This may encode qubit adjacency, coupling constraints, or other topology
        information in a backend-specific format.

        Returns ``None`` if the backend does not expose connectivity information.
        Concrete backend implementations may return structured, vendor-defined
        objects.
        """
        return None

    @property
    def gate_set(self) -> Any | None:
        """
        Native gate set supported by the backend.

        The exact meaning of this property is backend-specific. For hardware
        backends, this typically refers to native operations; for simulators, it
        may be omitted or ignored.

        Returns ``None`` if the backend does not expose gate availability.
        """
        return None

    @property
    def error_rates(self) -> Any | None:
        """
        Error rates or calibration-related information for the backend.

        This may depend on a specific calibration snapshot and is typically
        hardware- and vendor-specific.

        Returns ``None`` if the backend does not expose error or calibration data.
        """
        return None

    # ----------------------------------------------------------------------
    # Extension point for backend-specific capabilities
    # ----------------------------------------------------------------------

    @property
    def capabilities(self) -> Mapping[str, Any]:
        """
        Backend-specific capabilities not covered by the base interface.

        Keys and values are backend-defined. This property is intended as an
        extension point for vendor backends to expose additional structured
        information without modifying the base Backend API.
        """
        return {}
