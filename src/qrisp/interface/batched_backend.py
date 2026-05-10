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
This module defines :class:`BatchedBackend`, obtained via
:meth:`Backend.batched() <qrisp.interface.Backend.batched>`.  It wraps any
:class:`~qrisp.interface.Backend`, buffers circuit submissions in an explicit
request queue, and executes them all when :meth:`~BatchedBackend.dispatch` is
called.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.interface.backend import Backend
from qrisp.interface.measurement_result import MeasurementResult


class BatchedBackend(Backend):
    """
    A backend that buffers circuit submissions and executes them on an explicit
    :meth:`dispatch` call.

    Obtain an instance via :meth:`Backend.batched`:

    .. code-block:: python

        bb = my_backend.batched()

    .. rubric:: How it works

    Each call to :meth:`run` registers the circuit in an internal queue and
    returns a :class:`~qrisp.interface.MeasurementResult` immediately.  The
    result is *lazy*: its data is not available until :meth:`dispatch` is
    called.  Accessing the result before dispatch raises ``RuntimeError``.

    After :meth:`dispatch`, every pending
    :class:`~qrisp.interface.MeasurementResult` is populated with raw
    bitstring counts by forwarding each circuit to the wrapped backend's
    :meth:`~qrisp.interface.Backend.run`.  Higher-level decoding (performed by
    :meth:`QuantumVariable.get_measurement
    <qrisp.QuantumVariable.get_measurement>`) is deferred until the user
    actually reads the decoded result.

    Parameters
    ----------
    backend : Backend
        The backend whose :meth:`~qrisp.interface.Backend.run` is called for
        each queued circuit when :meth:`dispatch` is invoked.

    name : str or None
        Optional name for this backend.

    options : Mapping or None
        Runtime options.  Defaults to the wrapped backend's current options.

    Examples
    --------

    **Basic usage** — collect two measurements, then dispatch:

    .. code-block:: python

        from qrisp import QuantumFloat
        from qrisp.default_backend import DefaultBackend

        backend = DefaultBackend()
        bb = backend.batched()

        a = QuantumFloat(4); a[:] = 1
        b = QuantumFloat(3); b[:] = 2
        c = a + b  # expected: 3

        d = QuantumFloat(4); d[:] = 2
        e = QuantumFloat(3); e[:] = 3
        f = d + e  # expected: 5

        res_c = c.get_measurement(backend=bb)  # lazy — returns immediately
        res_f = f.get_measurement(backend=bb)  # same

        bb.dispatch()  # runs both circuits via backend.run(), then populates results

        print(res_c)  # {3: 1.0}
        print(res_f)  # {5: 1.0}

    The same pattern is available via
    :func:`~qrisp.batched_measurement`:

    .. code-block:: python

        from qrisp import batched_measurement
        results = batched_measurement([c, f], backend=bb)
    """

    def __init__(
        self,
        backend: Backend,
        name: str | None = None,
        options: Mapping | None = None,
    ):
        if options is None:
            options = dict(backend._options)
        super().__init__(name=name, options=options)
        self._backend = backend
        # Each entry: (circuit, shots, MeasurementResult)
        self._queries: list[tuple] = []

    @classmethod
    def _default_options(cls):
        return {"shots": 1024}

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    def run_async(self, circuits, shots=None):
        """Not supported — :class:`BatchedBackend` uses an explicit dispatch model.

        Use :meth:`run` to queue circuits and :meth:`dispatch` to execute them.

        Raises
        ------
        NotImplementedError
            Always. This method is intentionally unsupported.
        """
        raise NotImplementedError(
            "BatchedBackend does not support run_async(). "
            "Submit circuits via run() and execute them with dispatch()."
        )

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
        batch = not isinstance(circuits, QuantumCircuit)
        if not batch:
            circuits = [circuits]

        n_shots = shots if shots is not None else self._options.get("shots")

        raw_list: list[MeasurementResult] = []
        for qc in circuits:
            raw = MeasurementResult()
            self._queries.append((qc, n_shots, raw))
            raw_list.append(raw)

        return raw_list if batch else raw_list[0]

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self) -> None:
        """Execute all pending circuits and populate their results.

        Each queued circuit is forwarded to the wrapped backend's
        :meth:`~qrisp.interface.Backend.run`.  Results are injected into the
        corresponding :class:`~qrisp.interface.MeasurementResult` objects.

        If the wrapped backend raises for any circuit, that error is stored in
        every not-yet-populated :class:`~qrisp.interface.MeasurementResult`
        and then re-raised.  Accessing any of those results later will re-raise
        the stored exception.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Re-raises any exception thrown by the wrapped backend during
            execution, after injecting the error into all pending results.
        """
        queries = list(self._queries)
        self._queries = []

        if not queries:
            return

        for qc, shots, raw in queries:
            try:
                result = cast(MeasurementResult, self._backend.run(qc, shots=shots))
                raw._inject(result._data)
            except Exception as exc:
                for _, _, r in queries:
                    if not r._populated:
                        r._inject_error(exc)
                raise
