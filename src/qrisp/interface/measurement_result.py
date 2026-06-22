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

"""Lazy measurement result types used by Qrisp backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


class LazyDict(Mapping[Any, Any], ABC):
    """Abstract lazy mapping whose data is computed exactly once on first access.

    ``LazyDict`` is the common base class for all result types returned by
    Qrisp backends. Subclasses implement :meth:`_populate` to fill
    ``self._data`` the first time any ``Mapping``
    operation is called. Subsequent accesses reuse the cached data without
    re-running :meth:`_populate`.

    If ``_populate`` raises an exception, ``_populated`` stays ``False``
    and the next access will call ``_populate`` again. Concrete subclasses
    such as :class:`MeasurementResult` exploit this to implement persistent
    error propagation: they re-raise the stored exception on every call.

    All standard ``Mapping`` operations (``[]``,
    ``len``, ``iter``, ``==``) are supported and trigger population on
    first use.

    .. rubric:: Implementing a custom LazyDict

    Subclass ``LazyDict`` and implement :meth:`_populate` to fill
    ``self._data`` with a plain ``dict``::

        class MyResult(LazyDict):
            def _populate(self):
                self._data = {"answer": 42}

        r = MyResult()
        print(r._populated)  # False
        print(r["answer"])   # 42 (population happens here)
        print(r._populated)  # True
    """

    __hash__ = None  # Mappings are unhashable by convention

    def __init__(self) -> None:
        self._populated: bool = False
        self._data: dict | None = None

    def _ensure_populated(self) -> None:
        """Trigger population if it has not happened yet.

        Calls :meth:`_populate` exactly once on first access. If
        ``_populate`` raises, ``_populated`` stays ``False`` and the next
        access will call ``_populate`` again.
        """
        if not self._populated:
            self._populate()
            self._populated = True

    @abstractmethod
    def _populate(self) -> None:
        """Fill ``self._data``. Called at most once by :meth:`_ensure_populated`."""

    # ------------------------------------------------------------------
    # Mapping protocol
    # ------------------------------------------------------------------

    def __getitem__(self, key: Any) -> Any:
        self._ensure_populated()
        assert self._data is not None
        return self._data[key]

    def __iter__(self):
        self._ensure_populated()
        assert self._data is not None
        return iter(self._data)

    def __len__(self) -> int:
        self._ensure_populated()
        assert self._data is not None
        return len(self._data)

    # ------------------------------------------------------------------
    # Extras for usability
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self._populated:
            try:
                self._ensure_populated()
            except Exception:
                return f"<{type(self).__name__} pending>"
        assert self._data is not None
        return repr(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            self._ensure_populated()
            assert self._data is not None
            return self._data == dict(other)
        return NotImplemented


class MeasurementResult(LazyDict):
    """Raw bitstring counts returned by
    :meth:`Backend.run() <qrisp.interface.Backend.run>`.

    :meth:`~qrisp.interface.Backend.run` always returns a
    ``MeasurementResult``. For standard (non-batched) backends the object
    arrives pre-populated (data is immediately available without any
    further call). For :class:`~qrisp.interface.BatchedBackend` the object
    starts empty and is filled only when
    :meth:`~qrisp.interface.BatchedBackend.dispatch` is called. Any
    attempt to read the data before population raises ``RuntimeError``.

    ``MeasurementResult`` is a ``Mapping``, so all dict-style access
    works unchanged.

    Examples
    --------
    The typical way to obtain a ``MeasurementResult`` is through
    :meth:`~qrisp.interface.Backend.run`:

    .. code-block:: python

        from qrisp import QuantumCircuit
        from qrisp.default_backend import QrispSimulatorBackend

        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.measure(0)

        backend = QrispSimulatorBackend()
        result = backend.run(circuit, shots=1024)

    The object supports all standard dict-style operations:

    .. code-block:: python

        >>> print(result["0"])                      # e.g. 512
        >>> print(result["1"])                      # e.g. 512
        >>> print(len(result))                      # 2
        >>> print(set(result))                      # {'1', '0'}
        >>> for bitstring, count in result.items():
        ...     print(bitstring, count)

    Equality comparison works against a plain ``dict``:

    .. code-block:: python

        >>> result == {"0": 512, "1": 512}         # True if counts match exactly

    Membership check:

    .. code-block:: python

        >>> isinstance(result, Mapping)             # True
        >>> isinstance(result, dict)                # False

    For the *batched* (lazy) case, the object is returned before the
    circuits are executed. Accessing data before
    :meth:`~qrisp.interface.BatchedBackend.dispatch` is called raises
    ``RuntimeError``:

    .. code-block:: python

        from qrisp import QuantumFloat
        from qrisp.default_backend import QrispSimulatorBackend

        batched_backend = QrispSimulatorBackend().batched()

        qf = QuantumFloat(3)
        qf[:] = 5

        result = qf.get_measurement(backend=batched_backend)  # returns immediately (empty)

        # result["0"]   # RuntimeError: Call dispatch() first.

        batched_backend.dispatch()
        print(result)   # {5: 1.0}

    """

    def __init__(self) -> None:
        super().__init__()
        self._error: Exception | None = None

    def _populate(self) -> None:
        """Raise until the result is populated via :meth:`_inject`.

        Re-raises ``_error`` if one was stored by :meth:`_inject_error`,
        otherwise raises ``RuntimeError`` to signal that
        :meth:`~qrisp.interface.BatchedBackend.dispatch` has not been called
        yet.
        """
        if self._error is not None:
            raise self._error
        raise RuntimeError("MeasurementResult not yet populated. Call dispatch() first.")

    def _inject(self, counts: dict) -> None:
        """Populate with raw backend results.

        Called by :meth:`~qrisp.interface.Backend.run` for standard backends
        and by :meth:`~qrisp.interface.BatchedBackend.dispatch` for batched
        backends.

        Parameters
        ----------
        counts : dict
            Raw bitstring-to-count mapping produced by the backend.

        """
        self._data = counts
        self._populated = True

    def _inject_error(self, exc: Exception) -> None:
        """Store *exc* so it is raised on every subsequent data access.

        After this call ``_populated`` remains ``False``, so every subsequent
        access calls :meth:`_populate`, which immediately re-raises the stored
        exception. This ensures the error is visible no matter how many times
        the result is accessed.

        Parameters
        ----------
        exc : Exception
            The exception to store and re-raise on access.

        """
        self._error = exc


class DecodedMeasurementResult(LazyDict):
    """Human-readable measurement result produced by
    :meth:`QuantumVariable.get_measurement() <qrisp.QuantumVariable.get_measurement>`.

    Wraps a raw or intermediate :class:`LazyDict` together with a *decoder*
    callable. On first access the raw data is read, *decoder* is applied to
    every key, duplicate labels are merged by summing their counts, and the
    result is cached sorted by probability (descending).

    Because decoding is deferred, this object can be returned *before*
    :meth:`~qrisp.interface.BatchedBackend.dispatch` is called when using a
    :class:`~qrisp.interface.BatchedBackend`. The decoded data becomes
    available automatically once the underlying raw result is populated.

    Parameters
    ----------
    raw : LazyDict
        The source of integer-keyed, normalised probability data (typically
        an :class:`_IntKeyedResult` returned by ``get_measurement_from_qc``).
    decoder : Callable[[int], object]
        Maps an integer bitstring index to a user-facing label. This is
        ``QuantumVariable.decoder``.

    Examples
    --------
    ``DecodedMeasurementResult`` is returned directly by
    :meth:`~qrisp.QuantumVariable.get_measurement`:

    .. code-block:: python

        from qrisp import QuantumFloat
        from qrisp.default_backend import QrispSimulatorBackend

        backend = QrispSimulatorBackend()

        qf = QuantumFloat(4)
        qf[:] = 7
        result = qf.get_measurement(backend=backend)

        >>> print(result)               # {7: 1.0}
        >>> print(result[7])            # 1.0
        >>> print(len(result))          # 1
        >>> print(result == {7: 1.0})   # True

    When multiple outcomes are possible, the result is sorted by probability,
    highest first:

    .. code-block:: python

        from qrisp import QuantumFloat, h

        qf = QuantumFloat(2)
        h(qf[0])

        result = qf.get_measurement(backend=backend, shots=1024)
        # e.g. {0: 0.51, 1: 0.49} or {1: 0.52, 0: 0.48} (highest first)

    In the *batched* case the object is returned immediately and decoding
    happens lazily on first access after
    :meth:`~qrisp.interface.BatchedBackend.dispatch`:

    .. code-block:: python

        from qrisp import QuantumFloat
        from qrisp.default_backend import QrispSimulatorBackend

        batched_backend = QrispSimulatorBackend().batched()

        a = QuantumFloat(4); a[:] = 3
        b = QuantumFloat(4); b[:] = 5

        res_a = a.get_measurement(backend=batched_backend)
        res_b = b.get_measurement(backend=batched_backend)

        print(res_a)        # <DecodedMeasurementResult pending>
        print(res_b)        # <DecodedMeasurementResult pending>

        batched_backend.dispatch()       # populates raw results

        print(res_a)        # {3: 1.0}  decoded on first access
        print(res_b)        # {5: 1.0}

    """

    def __init__(self, raw: LazyDict, decoder: Callable[[int], object]) -> None:
        super().__init__()
        self._raw = raw
        self._decoder = decoder

    def _populate(self) -> None:

        self._raw._ensure_populated()
        assert self._raw._data is not None
        d: dict = {}
        for key, count in sorted(self._raw._data.items()):
            label = self._decoder(key)
            d[label] = d.get(label, 0) + count
        try:
            self._data = dict(sorted(d.items(), key=lambda x: -x[1]))
        except TypeError:
            self._data = d


class MultiMeasurementResult(LazyDict):
    """Lazy result for :func:`~qrisp.multi_measurement`.

    Defers decoding of the combined bitstring into per-variable labels until
    first access, so the result can be returned before
    :meth:`~qrisp.interface.BatchedBackend.dispatch` is called when using a
    :class:`~qrisp.interface.BatchedBackend`.

    Parameters
    ----------
    raw : MeasurementResult
        The raw bitstring counts returned by
        :meth:`~qrisp.interface.Backend.run`.
    qv_list : list
        The :class:`~qrisp.QuantumVariable` instances whose joint measurement
        is being decoded, in the same order as passed to
        :func:`~qrisp.multi_measurement`.
    cl_reg_list : list[list]
        Classical-bit sub-registers, one per variable in *qv_list* (reversed),
        produced by the circuit construction inside
        :func:`~qrisp.multi_measurement`.

    """

    def __init__(self, raw: MeasurementResult, qv_list: list, cl_reg_list: list) -> None:
        super().__init__()
        self._raw = raw
        self._qv_list = qv_list
        self._cl_reg_list = cl_reg_list

    def _populate(self) -> None:

        from qrisp import OutcomeArray

        self._raw._ensure_populated()
        assert self._raw._data is not None

        # Sort for deterministic decoding order.
        raw_counts = dict(sorted(self._raw._data.items()))
        # Normalise to probabilities in case the backend returned raw counts.
        total = sum(raw_counts.values())

        # The circuit in multi_measurement adds classical registers in
        # qv_list[::-1] order (last variable first), so cl_reg_list is stored
        # in that same reversed order.
        # Many simulators put the most recently added classical bits at the
        # left of the bitstring.
        # Reversing cl_reg_list here re-aligns index j with qv_list[j] and
        # with the corresponding slice in the bitstring.
        regs = self._cl_reg_list[::-1]

        new_counts: dict = {}
        for bitstring, count in raw_counts.items():
            labels = []
            offset = 0
            for j, cl_reg in enumerate(regs):
                # Slice out the bits for variable j. The [::-1] pair is an
                # artifact of the original bit-order convention: reversing and
                # then reversing back is a no-op, so outcome_int is simply
                # int(bitstring[offset : offset + len(cl_reg)], 2).
                sub = bitstring[offset : offset + len(cl_reg)][::-1]
                offset += len(cl_reg)
                outcome_int = int(sub[::-1], 2)

                # Apply the variable's decoder. If the variable has no decoder
                # (AttributeError), fall back to the raw integer.
                try:
                    label = self._qv_list[j].decoder(outcome_int)
                    if isinstance(label, np.ndarray):
                        label = OutcomeArray(label)
                except AttributeError:
                    label = outcome_int
                labels.append(label)

            outcome = tuple(labels)
            try:
                new_counts[outcome] = count / total
            except TypeError as exc:
                raise TypeError(
                    "Tried to create measurement outcome dic for QuantumVariable with unhashable labels"
                ) from exc

        # Sort descending by probability, matching the convention used by
        # QuantumVariable.get_measurement() and DecodedMeasurementResult.
        self._data = dict(sorted(new_counts.items(), key=lambda item: -item[1]))


class _IntKeyedResult(LazyDict):
    """Internal: lazily converts raw bitstring counts to normalised int-keyed probabilities.

    Created by ``get_measurement_from_qc`` after calling
    :meth:`~qrisp.interface.Backend.run`. On first access it reads the raw
    bitstring counts, strips spaces, truncates each key to *num_bits* bits,
    converts to an integer, merges colliding keys by summing, and normalises
    to probabilities when the total count differs from 1 by more than 1e-3.
    """

    def __init__(self, raw: MeasurementResult, num_bits: int) -> None:
        super().__init__()
        self._raw = raw
        self._num_bits = num_bits

    def _populate(self) -> None:

        self._raw._ensure_populated()
        assert self._raw._data is not None
        new_counts: dict[int, float] = {}
        total: float = 0
        for bits, count in self._raw._data.items():
            trimmed = bits.replace(" ", "")[: self._num_bits]
            key = int(trimmed, base=2)
            new_counts[key] = new_counts.get(key, 0) + count
            total += count
        if total > 0 and abs(1 - total) > 1e-3:
            for k in new_counts:
                new_counts[k] /= abs(total)
        self._data = new_counts
