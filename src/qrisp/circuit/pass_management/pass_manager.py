"""********************************************************************************
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

from collections.abc import Iterator
from typing import Any

from qrisp.circuit.pass_management.circuit_pass import CircuitPass
from qrisp.circuit.quantum_circuit import QuantumCircuit


class PassManager:
    """A manager for organizing and applying quantum circuit transformation passes.

    The PassManager maintains an ordered list of :class:`CircuitPass` instances
    and applies them sequentially to quantum circuits.

    Only :class:`CircuitPass` instances are accepted — raw functions must be
    wrapped first, for example with the ``@CircuitPass`` decorator or by calling
    ``CircuitPass(my_func)``.

    Example:
    --------
    Build a pipeline with ``+=``, then inspect, insert, and remove passes::

        >>> from qrisp import PassManager
        >>> from qrisp import convert_to_cz, fuse_adjacents, remove_barriers
        >>>
        >>> pm = PassManager()
        >>> pm += convert_to_cz()
        >>> pm += fuse_adjacents
        >>> pm += remove_barriers
        >>>
        >>> print(pm)
        PassManager(['convert_to_cz', 'fuse_adjacents', 'remove_barriers'])
        >>>
        >>> # Insert a pass at a specific position
        >>> from qrisp import decompose
        >>> pm.insert_pass(1, decompose)
        >>> print(pm)
        PassManager(['convert_to_cz', 'decompose', 'fuse_adjacents', 'remove_barriers'])
        >>>
        >>> # Remove the last pass
        >>> pm.remove_pass(len(pm) - 1)
        >>> print(pm)
        PassManager(['convert_to_cz', 'decompose', 'fuse_adjacents'])
        >>>
        >>> transpiled_qc = pm.run(qc)

    """

    def __init__(self, passes: list[CircuitPass] | None = None) -> None:
        """Initialize a PassManager.

        Parameters
        ----------
        passes : list[CircuitPass], optional
            Initial list of :class:`CircuitPass` instances.

        """
        self._passes: list[CircuitPass] = []
        if passes is not None:
            for p in passes:
                self._validate_pass(p)
            self._passes.extend(passes)

    @staticmethod
    def _validate_pass(pass_obj: Any) -> None:
        """Raise :exc:`TypeError` if *pass_obj* is not a :class:`CircuitPass`."""
        if not isinstance(pass_obj, CircuitPass):
            raise TypeError(
                f"PassManager only accepts CircuitPass instances, "
                f"got {type(pass_obj).__name__}. "
                f"Wrap raw functions with CircuitPass(func) or the @CircuitPass decorator."
            )

    def add_pass(self, pass_obj: CircuitPass) -> PassManager:
        """Add a transformation pass to the manager.

        Parameters
        ----------
        pass_obj : CircuitPass
            A :class:`CircuitPass` wrapping a transformation with signature
            ``QuantumCircuit -> QuantumCircuit``.

        Returns
        -------
        PassManager
            Returns self for method chaining.

        """
        self._validate_pass(pass_obj)
        self._passes.append(pass_obj)
        return self

    def insert_pass(self, index: int, pass_obj: CircuitPass) -> PassManager:
        """Insert a pass at a specific position.

        Parameters
        ----------
        index : int
            Position to insert the pass.
        pass_obj : CircuitPass
            The :class:`CircuitPass` to insert.

        Returns
        -------
        PassManager
            Returns self for method chaining.

        """
        self._validate_pass(pass_obj)
        self._passes.insert(index, pass_obj)
        return self

    def remove_pass(self, index: int) -> PassManager:
        """Remove a pass at the specified index.

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
        """Remove all passes from the manager.

        Returns
        -------
        PassManager
            Returns self for method chaining.

        """
        self._passes.clear()
        return self

    def run(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Apply all passes in sequence to the quantum circuit.

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
        for circuit_pass in self._passes:
            result = circuit_pass(result)
        return result

    def verify(
        self,
        qc: QuantumCircuit,
        verification_type: str,
        visualize_failures: bool = False,
        **verification_kwargs: Any,
    ) -> list[tuple[str, bool]]:
        """Verify every pass in the manager against *qc* one by one.

        For each pass, the method captures the circuit state before
        applying the pass and then verifies that the pass preserves it
        using either :meth:`CircuitPass.compare_unitary` or
        :meth:`CircuitPass.compare_measurement`.

        Parameters
        ----------
        qc : QuantumCircuit
            The input quantum circuit to test the pipeline against.
        verification_type : str
            Which verification method to use — ``"unitary"`` or
            ``"measurements"``.
        visualize_failures : bool, optional
            If ``True``, call :meth:`CircuitPass.visualize` on any
            pass that fails verification. The default is ``False``.
        **verification_kwargs : Any
            Keyword arguments forwarded to the verification method
            (e.g. ``precision``, ``ignore_gphase``, ``backend``).

        Returns
        -------
        list of tuple(str, bool)
            A list of ``(pass_name, passed)`` pairs, one for each
            pass in verification order.

        Raises
        ------
        ValueError
            If *verification_type* is not ``"unitary"`` or
            ``"measurements"``.

        """
        if verification_type not in ("unitary", "measurements"):
            raise ValueError(f"Unknown verification_type {verification_type!r}. Expected 'unitary' or 'measurements'.")

        results: list[tuple[str, bool]] = []
        current = qc
        for circuit_pass in self._passes:
            if verification_type == "unitary":
                passed = circuit_pass.compare_unitary(current, **verification_kwargs)
            else:
                passed = circuit_pass.compare_measurement(current, **verification_kwargs)

            results.append((circuit_pass.__name__, passed))

            if not passed and visualize_failures:
                circuit_pass.visualize(current)

            # Always apply the pass so subsequent passes see a realistic
            # circuit state — even if this one failed verification.
            current = circuit_pass(current)

        return results

    def __iadd__(self, other: CircuitPass | PassManager) -> PassManager:
        """Add a pass or extend with the passes of another PassManager in-place.

        Usage: ``pm += circuit_pass`` or ``pm += other_pass_manager``.

        Parameters
        ----------
        other : CircuitPass or PassManager
            The pass or pass manager whose passes to add.

        Returns
        -------
        PassManager
            Returns self after modification.

        """
        if isinstance(other, CircuitPass):
            self._passes.append(other)
        elif isinstance(other, PassManager):
            self._passes.extend(other._passes)
        else:
            raise TypeError(
                f"PassManager only accepts CircuitPass or PassManager instances, got {type(other).__name__}."
            )
        return self

    def __len__(self) -> int:
        """Return the number of passes."""
        return len(self._passes)

    def __iter__(self) -> Iterator[CircuitPass]:
        """Iterate over passes."""
        return iter(self._passes)

    def __repr__(self) -> str:
        pass_names = [getattr(p, "__name__", repr(p)) for p in self._passes]
        return f"PassManager({pass_names})"

    @classmethod
    def from_list(cls, pass_list: list[CircuitPass]) -> PassManager:
        """Create a PassManager from a list of passes.

        Parameters
        ----------
        pass_list : list[CircuitPass]
            List of :class:`CircuitPass` instances.

        Returns
        -------
        PassManager
            A new PassManager instance.

        """
        return cls(pass_list)
