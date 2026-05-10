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

"""Lazy measurement result types used by Qrisp backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping


class LazyDict(Mapping, ABC):
    """Abstract lazy mapping whose data is computed exactly once on first access.

    Subclasses implement :meth:`_populate` to fill ``self._data``.  All
    ``Mapping`` operations delegate to :meth:`_ensure_populated`, so data is
    never computed before it is needed.
    """

    __hash__ = None  # type: ignore[assignment]  # Mappings are unhashable by convention

    def __init__(self) -> None:
        self._populated: bool = False
        self._data: dict | None = None

    def _ensure_populated(self) -> None:
        if not self._populated:
            self._populate()
            self._populated = True

    @abstractmethod
    def _populate(self) -> None:
        """Fill ``self._data``.  Called at most once by :meth:`_ensure_populated`."""

    # --- Mapping protocol -------------------------------------------------------

    def __getitem__(self, key: object) -> object:
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

    # --- Extras for usability ---------------------------------------------------

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
            if isinstance(other, LazyDict):
                other._ensure_populated()
                return self._data == other._data
            return self._data == dict(other)
        return NotImplemented


class MeasurementResult(LazyDict):
    """Raw bitstring counts returned by :class:`~qrisp.interface.Backend`.

    :meth:`~qrisp.interface.Backend.run` always returns a ``MeasurementResult``.
    For standard (non-batched) backends the object arrives pre-populated —
    data is immediately available without any further call.  For
    :class:`~qrisp.interface.BatchedBackend` the object starts empty and is
    filled only when :meth:`~qrisp.interface.BatchedBackend.dispatch` is
    called.  Any attempt to read the data before population raises
    ``RuntimeError``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._error: Exception | None = None

    def _populate(self) -> None:
        if self._error is not None:
            raise self._error
        raise RuntimeError(
            "MeasurementResult not yet populated. Call dispatch() first."
        )

    def _inject(self, counts: dict) -> None:
        """Populate with raw backend results.

        Called by :meth:`~qrisp.interface.Backend.run` for standard backends
        and by :meth:`~qrisp.interface.BatchedBackend.dispatch` for batched
        backends.
        """
        self._data = counts
        self._populated = True

    def _inject_error(self, exc: Exception) -> None:
        """Store *exc* so it is raised on the first data access."""
        self._error = exc


class DecodedMeasurementResult(LazyDict):
    """Human-readable measurement result produced by :meth:`QuantumVariable.get_measurement`.

    Wraps a raw or intermediate :class:`LazyDict` together with a *decoder*
    callable.  On first access the raw data is read, *decoder* is applied to
    every key, duplicate labels are merged, and the result is cached sorted by
    probability (descending).

    Parameters
    ----------
    raw : LazyDict
        The source of integer-keyed, normalised probability data — typically an
        :class:`_IntKeyedResult` returned by ``get_measurement_from_qc``.
    decoder : Callable[[int], object]
        Maps an integer bitstring index to a user-facing label.  This is
        ``QuantumVariable.decoder``.
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


class _IntKeyedResult(LazyDict):
    """Internal: lazily converts raw bitstring counts to normalised int-keyed probabilities.

    Created by ``get_measurement_from_qc`` after calling
    :meth:`~qrisp.interface.Backend.run`.  On first access it reads the raw
    bitstring counts, truncates each key to *num_bits* bits, converts to an
    integer, and normalises to probabilities.
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
