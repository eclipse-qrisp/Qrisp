"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

from __future__ import annotations

from collections.abc import Callable, Iterator
from qrisp.circuit.quantum_circuit import QuantumCircuit


class PassManager:
    """
    A manager for organizing and applying quantum circuit transformation passes.
    
    The PassManager maintains an ordered list of passes (transformation functions)
    and applies them sequentially to quantum circuits. Each pass is a callable
    with signature ``QuantumCircuit -> QuantumCircuit``.
    
    Passes that require configuration parameters are factory functions that return
    the configured callable. Simple passes can be added directly.
    
    Example
    -------
    >>> from qrisp import PassManager
    >>> 
    >>> pm = PassManager()
    >>> pm.add_pass(some_layout_pass(coupling_map=[(0,1), (1,2)]))
    >>> pm.add_pass(some_routing_pass(coupling_map=[(0,1), (1,2)]))
    >>> pm.add_pass(some_direct_pass)
    >>> 
    >>> transpiled_qc = pm.run(qc)
    """

    def __init__(self, passes: list[Callable[[QuantumCircuit], QuantumCircuit]] | None = None) -> None:
        """
        Initialize a PassManager.

        Parameters
        ----------
        passes : list[Callable], optional
            Initial list of pass functions. Each function should have signature
            ``QuantumCircuit -> QuantumCircuit``.
        """
        self._passes: list[Callable[[QuantumCircuit], QuantumCircuit]] = []
        if passes is not None:
            self._passes.extend(passes)

    def add_pass(self, pass_func: Callable[[QuantumCircuit], QuantumCircuit]) -> PassManager:
        """
        Add a transformation pass to the manager.

        Parameters
        ----------
        pass_func : Callable
            A function with signature ``QuantumCircuit -> QuantumCircuit``.
            Use factory functions to configure passes with parameters.

        Returns
        -------
        PassManager
            Returns self for method chaining.
        """
        self._passes.append(pass_func)
        return self

    def insert_pass(self, index: int, pass_func: Callable[[QuantumCircuit], QuantumCircuit]) -> PassManager:
        """
        Insert a pass at a specific position.

        Parameters
        ----------
        index : int
            Position to insert the pass.
        pass_func : Callable
            The pass function to insert.

        Returns
        -------
        PassManager
            Returns self for method chaining.
        """
        self._passes.insert(index, pass_func)
        return self

    def remove_pass(self, index: int) -> PassManager:
        """
        Remove a pass at the specified index.

        Parameters
        ----------
        index : int
            Index of the pass to remove.

        Returns
        -------
        PassManager
            Returns self for method chaining.
        """
        del self._passes[index]
        return self

    def clear(self) -> PassManager:
        """
        Remove all passes from the manager.

        Returns
        -------
        PassManager
            Returns self for method chaining.
        """
        self._passes.clear()
        return self

    def run(self, qc: QuantumCircuit) -> QuantumCircuit:
        """
        Apply all passes in sequence to the quantum circuit.

        Parameters
        ----------
        qc : QuantumCircuit
            The input quantum circuit to transform.

        Returns
        -------
        QuantumCircuit
            The transformed quantum circuit after all passes have been applied.
        """
        result = qc
        for pass_func in self._passes:
            result = pass_func(result)
        return result

    def __len__(self) -> int:
        """Return the number of passes."""
        return len(self._passes)

    def __iter__(self) -> Iterator[Callable[[QuantumCircuit], QuantumCircuit]]:
        """Iterate over passes."""
        return iter(self._passes)

    def __repr__(self) -> str:
        pass_names = [getattr(p, "__name__", repr(p)) for p in self._passes]
        return f"PassManager({pass_names})"

    @classmethod
    def from_list(cls, pass_list: list[Callable[[QuantumCircuit], QuantumCircuit]]) -> PassManager:
        """
        Create a PassManager from a list of passes.

        Parameters
        ----------
        pass_list : list[Callable]
            List of pass functions with signature ``QuantumCircuit -> QuantumCircuit``.

        Returns
        -------
        PassManager
            A new PassManager instance.
        """
        return cls(pass_list)
