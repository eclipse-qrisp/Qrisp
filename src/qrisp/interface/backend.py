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

# TODO: This class is complete state of construction.
# It should be extended with more functionality as needed.


class Backend(ABC):
    """
    Minimal abstract Backend class for Qrisp.

    - Uses a simple `dict` for backend options.
    - Inspired by Qiskit BackendV2 but intentionally simplified.
    """

    def __init__(self, name=None, description=None, options=None):
        self.name = name or self.__class__.__name__
        self.description = description

        # options is just a plain dict at the moment
        self._options = self._default_options()

        if options is not None:
            if not isinstance(options, dict):
                raise TypeError("Backend options must be a dict.")
            self._options.update(options)

    @property
    def options(self):
        """Return the backend options dictionary."""
        return self._options

    @classmethod
    @abstractmethod
    def _default_options(cls) -> dict:
        """Return a dict of default options."""
        pass

    def set_options(self, **kwargs):
        """
        Update the backend options for this instance.

        Example:
            backend.set_options(shots=5000, seed=42)

        By default, only fields that already exist in _default_options()
        can be updated. This prevents silent typos.

        Raises:
            AttributeError: If a field is not part of the allowed options.
        """

        for key, value in kwargs.items():
            if key not in self._options:
                raise AttributeError(
                    f"'{key}' is not a valid backend option. "
                    f"Valid options are: {list(self._options.keys())}"
                )
            self._options[key] = value

    @property
    @abstractmethod
    def max_circuits(self):
        """Maximum number of circuits supported in a single run."""
        pass

    @abstractmethod
    def run(self, run_input, **kwargs):
        """
        Execute a circuit or batch of circuits on the backend.

        run_input may be:
           - a single QuantumCircuit
           - a list of QuantumCircuit
           - anything the subclass supports

        kwargs override options.
        """
        pass
