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

"""
This module defines :class:`BatchedBackend`, obtained via
:meth:`Backend.batched() <qrisp.interface.Backend.batched>`.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import overload

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.interface.backend import Backend
from qrisp.interface.measurement_result import MeasurementResult


# NOTE: ``BatchedBackend`` intentionally does not inherit from
# ``Backend``. ``Backend.run`` is contractually required to return
# a fully populated result. A ``BatchedBackend`` cannot honour that
# contract because its ``run`` returns an empty placeholder that is
# only populated after ``dispatch`` is called. Inheriting from
# ``Backend`` would therefore violate the Liskov Substitution Principle.
class BatchedBackend:
    """
    A class that buffers circuit submissions and executes them on an explicit
    :meth:`dispatch` call.

    Obtain an instance via :meth:`~qrisp.interface.Backend.batched`:

    .. code-block:: python

        batched_backend = my_backend.batched()

    .. rubric:: How it works

    Each call to :meth:`run` registers the circuit in an internal queue and
    returns a :class:`~qrisp.interface.MeasurementResult` immediately.  The
    result is *lazy*: its data is not available until :meth:`dispatch` is
    called.  Accessing the result before dispatch raises ``RuntimeError``.

    After :meth:`dispatch`, every pending
    :class:`~qrisp.interface.MeasurementResult` is populated with raw
    bitstring counts by forwarding each circuit to the wrapped backend's
    :meth:`~qrisp.interface.Backend.run_async`. Higher-level decoding (performed by
    :meth:`QuantumVariable.get_measurement
    <qrisp.QuantumVariable.get_measurement>`) is deferred until the user
    actually reads the decoded result.

    .. note::

        :class:`BatchedBackend` is not thread-safe. :meth:`run` and
        :meth:`dispatch` must be called from the same thread. Concurrent
        calls from multiple threads can silently drop queued circuits.

    Parameters
    ----------
    backend : Backend
        The backend whose :meth:`~qrisp.interface.Backend.run_async` is called
        with all queued circuits when :meth:`dispatch` is invoked.

    name : str or None
        Optional name for this backend.

    Examples
    --------

    In this example, we collect two measurements, then dispatch:

    .. code-block:: python

        from qrisp import QuantumFloat
        from qrisp.default_backend import QrispSimulatorBackend

        backend = QrispSimulatorBackend()
        batched_backend = backend.batched()

        a = QuantumFloat(4); a[:] = 1
        b = QuantumFloat(3); b[:] = 2
        c = a + b

        d = QuantumFloat(4); d[:] = 2
        e = QuantumFloat(3); e[:] = 3
        f = d + e

        res_c = c.get_measurement(backend=batched_backend)  # lazy (returns immediately)
        res_f = f.get_measurement(backend=batched_backend)  # lazy (returns immediately)

        batched_backend.dispatch()  # runs both circuits, then populates results

        print(res_c)  # {3: 1.0}
        print(res_f)  # {5: 1.0}

    The same pattern is available via
    :func:`~qrisp.batched_measurement`:

    .. code-block:: python

        from qrisp import batched_measurement
        results = batched_measurement([c, f], backend=batched_backend)
        print(results)  # [{3: 1.0}, {5: 1.0}]
    """

    def __init__(
        self,
        backend: Backend,
        name: str | None = None,
    ):
        self.name = name or self.__class__.__name__
        self._backend = backend
        # Each entry: (circuit, shots, MeasurementResult)
        self._queries: list[tuple] = []

    @property
    def options(self) -> Mapping:
        """Current runtime options (read-only). Delegates to the wrapped backend."""
        return self._backend.options

    @property
    def pending_count(self) -> int:
        """Number of circuits currently queued, waiting for :meth:`dispatch`."""
        return len(self._queries)

    def update_options(self, **kwargs) -> None:
        """Update existing runtime options.

        Delegates to the wrapped backend's
        :meth:`~qrisp.interface.Backend.update_options`, which validates keys
        and raises ``AttributeError`` for unknown ones.  Changes are visible
        through both ``self.options`` and the wrapped backend's ``options``.

        Parameters
        ----------
        **kwargs
            Key-value pairs to update.
        """
        self._backend.update_options(**kwargs)

    @overload
    def run(
        self, circuits: QuantumCircuit, shots: int | None = None
    ) -> MeasurementResult: ...

    @overload
    def run(
        self, circuits: Sequence[QuantumCircuit], shots: int | None = None
    ) -> list[MeasurementResult]: ...

    def run(
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        shots: int | None = None,
    ) -> MeasurementResult | list[MeasurementResult]:
        """Register one or more circuits in the queue and return lazy result(s).

        Each circuit is stored alongside its shot count. The corresponding
        :class:`~qrisp.interface.MeasurementResult` objects are *empty* until
        :meth:`dispatch` is called.

        Parameters
        ----------
        circuits : QuantumCircuit or Sequence[QuantumCircuit]
            One circuit or a sequence of circuits to include in the batch.

        shots : int or None
            Number of shots. Defaults to the backend's ``shots`` option.

        Returns
        -------
        MeasurementResult or list[MeasurementResult]
            A single lazy result for a single circuit, or a list of lazy
            results when multiple circuits are given.
        """
        Backend._validate_shots(shots)
        batch = not isinstance(circuits, QuantumCircuit)
        if not batch:
            circuits = [circuits]

        n_shots = shots if shots is not None else self._backend.options.get("shots")

        raw_list: list[MeasurementResult] = []
        for qc in circuits:
            raw = MeasurementResult()
            self._queries.append((qc, n_shots, raw))
            raw_list.append(raw)

        return raw_list if batch else raw_list[0]

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, timeout: float | None = None) -> None:
        """Execute all pending circuits and populate their results.

        All queued circuits are submitted via
        :meth:`~qrisp.interface.Backend.run_async`, eliminating redundant
        network round-trips and queue-wait overhead regardless of whether the
        circuits were queued with the same or different shot counts.

        When all queued circuits share the same shot count a scalar is passed
        to :meth:`~qrisp.interface.Backend.run_async`.
        When circuits carry different shot counts the full per-circuit list is
        passed, enabling each circuit to be executed with its own shot budget.
        Backends whose SDK does not natively support per-circuit shot counts
        should fall back to a fixed shot count and emit a ``UserWarning``.

        If the wrapped backend's :attr:`~qrisp.interface.Backend.max_circuits`
        limit is set and the queued batch exceeds it, the batch is
        automatically split into smaller jobs and a :exc:`UserWarning` is
        emitted.

        If the wrapped backend raises, that error is stored in every
        not-yet-populated :class:`~qrisp.interface.MeasurementResult` and then
        re-raised. Accessing any of those results later will re-raise the stored
        exception.

        Parameters
        ----------
        timeout : float or None, optional
            Maximum seconds to wait for each hardware job. ``None`` (default)
            waits indefinitely. Passed through to
            :meth:`Job.result <qrisp.interface.Job.result>`.

        Raises
        ------
        UserWarning
            If the batch is split across multiple jobs due to
            :attr:`~qrisp.interface.Backend.max_circuits`.

        TimeoutError
            If *timeout* is given and a hardware job does not finish in time.
            Results for all circuits in the batch are left unpopulated.

        Exception
            Re-raises any exception thrown by the wrapped backend during
            execution, after injecting the error into all pending results.
        """
        queries = list(self._queries)
        self._queries = []

        if not queries:
            return

        circuits, shots_list, raws = map(list, zip(*queries))

        limit = self._backend.max_circuits
        chunk_size = limit if limit is not None else len(circuits)

        if limit is not None and len(circuits) > limit:
            n_jobs = -(-len(circuits) // limit)
            warnings.warn(
                f"The number of queued circuits ({len(circuits)}) exceeds the "
                f"backend's per-job circuit limit ({limit}). "
                f"Splitting into {n_jobs} jobs automatically.",
                UserWarning,
                stacklevel=2,
            )

        try:
            for start in range(0, len(circuits), chunk_size):
                self._submit_chunk(
                    circuits[start : start + chunk_size],
                    shots_list[start : start + chunk_size],
                    raws[start : start + chunk_size],
                    timeout,
                )
        except Exception as exc:
            for _, _, r in queries:
                if not r._populated:
                    r._inject_error(exc)
            raise

    def _submit_chunk(
        self,
        chunk_circuits: list,
        chunk_shots: list,
        chunk_raws: list,
        timeout: float | None,
    ) -> None:
        """Submit one chunk of circuits and inject results into the corresponding placeholders."""
        unique = set(chunk_shots)
        effective_shots = chunk_shots[0] if len(unique) == 1 else chunk_shots
        chunk_counts = (
            self._backend.run_async(chunk_circuits, shots=effective_shots)
            .result(timeout=timeout)
            .all_counts
        )
        for raw, counts in zip(chunk_raws, chunk_counts):
            raw._inject(counts)

    def clear(self) -> None:
        """Discard all queued circuits without dispatching.

        Any :class:`~qrisp.interface.MeasurementResult` objects previously
        returned by :meth:`run` for the cleared circuits remain unpopulated.
        Accessing them after ``clear()`` raises ``RuntimeError``.
        """
        self._queries = []
