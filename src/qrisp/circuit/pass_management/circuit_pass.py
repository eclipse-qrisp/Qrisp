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
from collections.abc import Callable
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

        from qrisp import CircuitPass, fuse_adjacents
        typed_pass = CircuitPass(fuse_adjacents)
        result = typed_pass(qc)

    Inside factory functions that return configured passes::

        def convert_to_cz(strict=False):
            @CircuitPass
            def _convert_to_cz(qc):
                ...
            return _convert_to_cz
    """

    def __init__(self, func: Callable[[QuantumCircuit], QuantumCircuit]) -> None:
        """
        Parameters
        ----------
        func : Callable[[QuantumCircuit], QuantumCircuit]
            The transformation function to wrap.  The function's ``__name__``
            and ``__doc__`` are copied onto the ``CircuitPass`` instance.
        """
        self._func: Callable[[QuantumCircuit], QuantumCircuit] = func
        functools.update_wrapper(
            self, func, assigned=("__module__", "__doc__", "__annotations__")
        )
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
            raise TypeError(f"Expected a QuantumCircuit, got {type(qc).__name__}.")

        # Check that the input circuit contains no measurements
        if any(instr.op.name == "measure" for instr in qc.data):
            raise ValueError(
                "The input circuit contains measurement instructions, "
                "which break unitarity. Remove measurements before "
                "calling compare_unitary."
            )

        # Apply the pass to a copy to avoid mutating the original
        transformed_qc = self(qc.copy())

        # Check that the transformed circuit contains no measurements
        if any(instr.op.name == "measure" for instr in transformed_qc.data):
            raise ValueError(
                "The transformed circuit contains measurement "
                "instructions, which break unitarity. The pass should "
                "not introduce measurements."
            )

        return qc.compare_unitary(transformed_qc, precision, ignore_gphase)

    def visualize(self, qc: QuantumCircuit) -> None:
        """
        Print a before/after visualisation of this pass applied to *qc*.

        The method copies *qc*, applies the pass, and prints both the
        original and the transformed circuit to the console.

        Parameters
        ----------
        qc : QuantumCircuit
            The input quantum circuit to visualise.

        Examples
        --------
        >>> from qrisp import QuantumCircuit, CircuitPass
        >>> from qrisp.circuit.pass_management.passes.fuse_adjacents import fuse_adjacents
        >>> qc = QuantumCircuit(2)
        >>> qc.cx(0, 1)
        >>> qc.cx(0, 1)
        >>> fuse_adjacents.visualize(qc)
        """
        # Apply the pass to a copy to avoid mutating the original
        transformed_qc = self(qc.copy())

        pass_name = self.__name__
        width = 60

        # ── Banner ────────────────────────────────────────────────────
        print(f"  {pass_name}  ".center(width, "="))

        # ── Before ────────────────────────────────────────────────────
        print(" Before ".center(width, "─"))

        print(qc)

        # ── After ─────────────────────────────────────────────────────
        print(" After ".center(width, "─"))
        print(transformed_qc)

        # ── Footer ────────────────────────────────────────────────────
        print("=" * width)

    def compare_measurement(
        self,
        qc: QuantumCircuit,
        precision: int = 6,
        backend: Any = None,
    ) -> bool:
        """
        Verify that this :class:`CircuitPass` leaves measurement statistics
        invariant when applied to the given :class:`~qrisp.QuantumCircuit`.

        The method copies *qc*, applies the pass to the copy, and then
        compares the measurement distributions of the original and the
        transformed circuit using :meth:`QuantumCircuit.run`.

        By default, the method runs in analytic mode (no shot noise): the
        default backend's ``shots`` option is ``None``, yielding exact
        probability distributions of type ``dict[bitstring, float]``.

        Parameters
        ----------
        qc : QuantumCircuit
            The input quantum circuit to test the pass against.
        precision : int, optional
            The number of decimal places of agreement required between
            corresponding probabilities. A pair of outcomes contributes a
            mismatch when ``abs(p_original - p_transformed) >= 10 ** -precision``.
            The default is 6.
        backend : Backend or None, optional
            The backend used for simulation. If ``None``, the Qrisp default
            backend is used (which runs in analytic mode with ``shots=None``).

        Returns
        -------
        bool
            ``True`` if the pass preserves the measurement distribution up
            to the given precision, ``False`` otherwise.

        Raises
        ------
        TypeError
            If *qc* is not a :class:`~qrisp.QuantumCircuit`.
        """
        if not isinstance(qc, QuantumCircuit):
            raise TypeError(f"Expected a QuantumCircuit, got {type(qc).__name__}.")

        if backend is None:
            from qrisp.default_backend import def_backend

            backend = def_backend

        # Apply the pass to a copy to avoid mutating the original
        transformed_qc = self(qc.copy())

        # Obtain measurement statistics. When shots is None (the default
        # for DefaultBackend) the results are exact probability
        # distributions (dict[str, float]).
        original_counts = qc.run(shots=None, backend=backend)
        transformed_counts = transformed_qc.run(shots=None, backend=backend)

        # Gather all observed outcome keys
        all_keys = set(original_counts.keys()) | set(transformed_counts.keys())
        tolerance = 10.0**-precision

        for key in all_keys:
            p_orig = original_counts.get(key, 0.0)
            p_trans = transformed_counts.get(key, 0.0)

            # Handle the empty-circuit / no-measurement case where
            # the simulator may return {"": None} when shots=None.
            if p_orig is None:
                p_orig = 0.0
            if p_trans is None:
                p_trans = 0.0

            if abs(p_orig - p_trans) >= tolerance:
                return False

        return True
