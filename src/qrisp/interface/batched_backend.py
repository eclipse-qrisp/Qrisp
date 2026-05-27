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

"""
This module defines :class:`BatchedBackend`, obtained via :meth:`Backend.batched() <qrisp.interface.Backend.batched>`.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.interface.backend import Backend
from qrisp.interface.measurement_result import MeasurementResult


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

    :class:`BatchedBackend` intentionally does not inherit from
    :class:`~qrisp.interface.Backend`. :meth:`~qrisp.interface.Backend.run`
    is contractually required to return a fully populated result. A
    :class:`BatchedBackend` cannot honour that contract because its
    :meth:`run` returns an empty placeholder that is only populated after
    :meth:`dispatch` is called. Inheriting from
    :class:`~qrisp.interface.Backend` would therefore violate the Liskov
    Substitution Principle.

    .. note::

        :class:`BatchedBackend` is not thread-safe. :meth:`run` and
        :meth:`dispatch` must be called from the same thread. Concurrent
        calls from multiple threads can silently drop queued circuits.

    Parameters
    ----------
    backend : Backend
        The backend whose :meth:`~qrisp.interface.Backend.run` is called for
        each queued circuit when :meth:`dispatch` is invoked.

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

    def run(self, circuits, shots: int | None = None):
        """Register one or more circuits in the queue and return lazy result(s).

        Each circuit is stored alongside its shot count.  The corresponding
        :class:`~qrisp.interface.MeasurementResult` objects are *empty* until
        :meth:`dispatch` is called.

        Parameters
        ----------
        circuits : QuantumCircuit or Sequence[QuantumCircuit]
            One circuit or a sequence of circuits to include in the batch.

        shots : int or None
            Number of shots.  Defaults to the backend's ``shots`` option.

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

        Circuits are grouped by shot count and each group is submitted as a
        single :meth:`~qrisp.interface.Backend.run_async` call.  For the
        common case where all circuits share the same shot count this reduces
        N separate hardware jobs to a single one, eliminating redundant
        network round-trips and queue-wait overhead.  Results are injected into
        the corresponding :class:`~qrisp.interface.MeasurementResult` objects.

        Circuit-limit enforcement (see :attr:`~qrisp.interface.Backend.max_circuits`)
        is the responsibility of the wrapped backend's
        :meth:`~qrisp.interface.Backend.run_async` implementation. Backends
        that override :attr:`~qrisp.interface.Backend.max_circuits` should
        check this limit inside
        :meth:`~qrisp.interface.Backend.run_async` so that the check applies
        whether circuits are submitted directly or via :meth:`dispatch`.

        .. warning::

            If the queued circuits use more than one distinct shot count, they
            are split into multiple hardware jobs (one per shot count).
            To avoid the split, ensure all circuits are queued with the
            same shot count.

        If the wrapped backend raises for any group, that error is stored in
        every not-yet-populated :class:`~qrisp.interface.MeasurementResult`
        and then re-raised. Accessing any of those results later will re-raise
        the stored exception.

        Parameters
        ----------
        timeout : float or None, optional
            Maximum seconds to wait for each hardware job.  ``None`` (default)
            waits indefinitely.  Passed through to
            :meth:`Job.result <qrisp.interface.Job.result>`.

        Raises
        ------
        TimeoutError
            If *timeout* is given and a hardware job does not finish in time.
            Results for the timed-out group and any remaining groups are left
            unpopulated.

        Exception
            Re-raises any exception thrown by the wrapped backend during
            execution, after injecting the error into all pending results.
        """
        queries = list(self._queries)
        self._queries = []

        if not queries:
            return

        # Group by effective shot count so each group becomes one hardware job.
        groups: dict[int | None, list] = {}
        for qc, shots, raw in queries:
            key = shots if shots is not None else self._backend.options.get("shots")
            groups.setdefault(key, []).append((qc, raw))

        if len(groups) > 1:
            warnings.warn(
                f"BatchedBackend.dispatch(): circuits use {len(groups)} different "
                f"shot counts ({list(groups.keys())}), requiring {len(groups)} "
                "separate hardware jobs. For a single hardware job, ensure all "
                "queued circuits use the same shot count.",
                UserWarning,
                stacklevel=2,
            )

        for shots, items in groups.items():
            circuits = [qc for qc, _ in items]
            raws = [raw for _, raw in items]
            try:
                all_counts = (
                    self._backend.run_async(circuits, shots=shots)
                    .result(timeout=timeout)
                    .all_counts
                )
                for raw, counts in zip(raws, all_counts):
                    raw._inject(counts)
            except Exception as exc:
                for _, _, r in queries:
                    if not r._populated:
                        r._inject_error(exc)
                raise

    def clear(self) -> None:
        """Discard all queued circuits without dispatching.

        Any :class:`~qrisp.interface.MeasurementResult` objects previously
        returned by :meth:`run` for the cleared circuits remain unpopulated.
        Accessing them after ``clear()`` raises ``RuntimeError``.
        """
        self._queries = []
