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
from typing import Any, Optional


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

    * number of qubits,
    * connectivity graph,
    * gate set,
    * error rates, etc.

    These properties default to ``None`` and are intended primarily for hardware
    backends. Simulators may safely ignore them.

    Parameters
    ----------
    name : str or None
        Optional user-defined name for the backend.
        Defaults to the class name.

    options : Mapping or None
        Runtime execution options for the backend.
        If omitted, :meth:`_default_options` is used.

    """

    def __init__(self, name: Optional[str] = None, options: Optional[Mapping] = None):
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

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """
        Execute one or more circuits/operations on the backend.

        Subclasses may override this method with a more specific signature.

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
    def options(self):
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
    def num_qubits(self):
        """Number of qubits available on the backend."""
        return None

    @property
    def connectivity(self):
        """Connectivity graph of the backend."""
        return None

    @property
    def gate_set(self):
        """Set of gates supported by the backend."""
        return None

    @property
    def error_rates(self):
        """Error rates of the backend's operations."""
        return None
