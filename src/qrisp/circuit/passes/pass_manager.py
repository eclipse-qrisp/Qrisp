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

import functools
from collections.abc import Callable, Iterator
from typing import Any

from qrisp.circuit.quantum_circuit import QuantumCircuit


class CircuitPass:
    """
    A decorator for quantum circuit transformation passes.

    ``CircuitPass`` wraps a callable with signature
    ``QuantumCircuit -> QuantumCircuit`` and enforces type safety by checking
    that both the argument and the return value are :class:`~qrisp.QuantumCircuit`
    instances.

    It can be used as a bare decorator on a pass function, or instantiated
    explicitly with a function reference.  Either way, the resulting
    ``CircuitPass`` object is itself callable and forwards calls to the
    wrapped function after performing the type checks.

    Examples
    --------
    Using ``CircuitPass`` as a decorator::

        @CircuitPass
        def my_pass(qc):
            # transform qc
            return qc

    Wrapping an existing function explicitly::

        from qrisp import CircuitPass, cancel_inverses
        typed_pass = CircuitPass(cancel_inverses)
        result = typed_pass(qc)

    Inside factory functions that return configured passes::

        def convert_to_cz(strict=False):
            @CircuitPass
            def _convert_to_cz(qc):
                ...
            return _convert_to_cz
    """

    def __init__(
        self, func: Callable[[QuantumCircuit], QuantumCircuit]
    ) -> None:
        """
        Parameters
        ----------
        func : Callable[[QuantumCircuit], QuantumCircuit]
            The transformation function to wrap.  The function's ``__name__``
            and ``__doc__`` are copied onto the ``CircuitPass`` instance.
        """
        self._func: Callable[[QuantumCircuit], QuantumCircuit] = func
        functools.update_wrapper(self, func, assigned=('__module__', '__doc__', '__annotations__'))
        self.__wrapped__ = func
        self.__name__ = func.__name__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Invoke the wrapped pass function with type guards.

        The argument must be a :class:`~qrisp.QuantumCircuit`.  The wrapped
        function receives it and the return value is checked to also be a
        :class:`~qrisp.QuantumCircuit`.

        Parameters
        ----------
        qc : QuantumCircuit
            The quantum circuit to transform.

        Returns
        -------
        QuantumCircuit
            The transformed quantum circuit.

        Raises
        ------
        TypeError
            If the argument is not a :class:`~qrisp.QuantumCircuit`, or if the
            wrapped function does not return a :class:`~qrisp.QuantumCircuit`.
        """
        # --- Type guard on input ---
        if len(args) != 1 or kwargs:
            raise TypeError(
                f"CircuitPass expected exactly one positional argument "
                f"(the QuantumCircuit), got {len(args)} positional and "
                f"{len(kwargs)} keyword arguments."
            )
        qc = args[0]
        if not isinstance(qc, QuantumCircuit):
            raise TypeError(
                f"CircuitPass expected a QuantumCircuit as input, "
                f"got {type(qc).__name__}."
            )

        # --- Forward to the wrapped function ---
        result = self._func(qc)

        # --- Type guard on output ---
        if not isinstance(result, QuantumCircuit):
            raise TypeError(
                f"CircuitPass expected the wrapped function to return a "
                f"QuantumCircuit, but it returned {type(result).__name__}."
            )

        return result

    def compare_unitary(
        self,
        qc: QuantumCircuit,
        precision: int = 4,
        ignore_gphase: bool = False,
    ) -> bool:
        """
        Verify that this :class:`CircuitPass` leaves the unitary invariant
        when applied to the given :class:`~qrisp.QuantumCircuit`.

        The method copies *qc*, applies the pass to the copy, and then
        compares the original and transformed unitaries using
        :meth:`QuantumCircuit.compare_unitary`.

        Measurements are not allowed because they break unitarity and would
        cause the underlying unitary computation to fail.

        Parameters
        ----------
        qc : QuantumCircuit
            The input quantum circuit to test the pass against.
        precision : int, optional
            The precision passed to
            :meth:`QuantumCircuit.compare_unitary`. The default is 4.
        ignore_gphase : bool, optional
            If ``True``, ignore global phase differences. The default is
            ``False``.

        Returns
        -------
        bool
            ``True`` if the pass preserves the unitary up to the given
            precision, ``False`` otherwise.

        Raises
        ------
        TypeError
            If *qc* is not a :class:`~qrisp.QuantumCircuit`.
        ValueError
            If either the input or the transformed circuit contains
            measurement instructions.
        """
        if not isinstance(qc, QuantumCircuit):
            raise TypeError(
                f"Expected a QuantumCircuit, got {type(qc).__name__}."
            )

        # Check that the input circuit contains no measurements
        for instr in qc.data:
            if instr.op.name == "measure":
                raise ValueError(
                    "The input circuit contains measurement instructions, "
                    "which break unitarity. Remove measurements before "
                    "calling compare_unitary."
                )

        # Apply the pass to a copy to avoid mutating the original
        transformed_qc = self(qc.copy())

        # Check that the transformed circuit contains no measurements
        for instr in transformed_qc.data:
            if instr.op.name == "measure":
                raise ValueError(
                    "The transformed circuit contains measurement "
                    "instructions, which break unitarity. The pass should "
                    "not introduce measurements."
                )

        return qc.compare_unitary(transformed_qc, precision, ignore_gphase)


class PassManager:
    """
    A manager for organizing and applying quantum circuit transformation passes.

    The PassManager maintains an ordered list of :class:`CircuitPass` instances
    and applies them sequentially to quantum circuits.

    Only :class:`CircuitPass` instances are accepted — raw functions must be
    wrapped first, for example with the ``@CircuitPass`` decorator or by calling
    ``CircuitPass(my_func)``.

    Example
    -------
    >>> from qrisp import PassManager, CircuitPass
    >>>
    >>> pm = PassManager()
    >>> pm.add_pass(CircuitPass(some_layout_pass(coupling_map=[(0,1), (1,2)])))
    >>> pm.add_pass(CircuitPass(some_routing_pass(coupling_map=[(0,1), (1,2)])))
    >>> pm.add_pass(CircuitPass(some_direct_pass))
    >>>
    >>> transpiled_qc = pm.run(qc)
    """

    def __init__(self, passes: list[CircuitPass] | None = None) -> None:
        """
        Initialize a PassManager.

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
        """
        Add a transformation pass to the manager.

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
        """
        Insert a pass at a specific position.

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
        for circuit_pass in self._passes:
            result = circuit_pass(result)
        return result

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
        """
        Create a PassManager from a list of passes.

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
