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

"""Tests for BatchedBackend — lazy queue + explicit dispatch architecture."""

import threading

import pytest

from qrisp import QuantumFloat
from qrisp.default_backend import DefaultBackend
from qrisp.interface import BatchedBackend
from qrisp.interface.measurement_result import (
    DecodedMeasurementResult,
    LazyDict,
    MeasurementResult,
)


def _make_batched_backend():
    return DefaultBackend().batched()


# ---------------------------------------------------------------------------
# Core batching behaviour
# ---------------------------------------------------------------------------


def test_batched_returns_batched_backend_instance():
    """Backend.batched() must return a BatchedBackend."""
    backend = DefaultBackend()
    bb = backend.batched()
    assert isinstance(bb, BatchedBackend)
    assert bb._backend is backend


def test_explicit_dispatch_batches_two_variables():
    """Two get_measurement calls followed by dispatch() must both return correct results.

    This is the primary integration test: no threading needed because
    get_measurement() returns a lazy result immediately and dispatch() populates
    both results by calling the wrapped backend once per circuit.
    """
    a = QuantumFloat(4); a[:] = 1
    b = QuantumFloat(3); b[:] = 2
    c = a + b  # expected: 3

    d = QuantumFloat(4); d[:] = 2
    e = QuantumFloat(3); e[:] = 2
    f = d + e  # expected: 4

    bb = _make_batched_backend()

    res_c = c.get_measurement(backend=bb)
    res_f = f.get_measurement(backend=bb)

    # Results are lazy before dispatch
    assert isinstance(res_c, LazyDict)
    assert isinstance(res_f, LazyDict)
    assert not res_c._populated
    assert not res_f._populated

    bb.dispatch()

    assert res_c == {3: 1.0}
    assert res_f == {4: 1.0}


def test_dispatch_returns_correct_results_with_single_variable():
    """A single get_measurement + dispatch() works as a degenerate batch of one."""
    a = QuantumFloat(4); a[:] = 1
    b = QuantumFloat(3); b[:] = 2
    c = a + b  # expected: 3

    bb = _make_batched_backend()
    res = c.get_measurement(backend=bb)
    bb.dispatch()

    assert res == {3: 1.0}


def test_result_is_lazy_before_dispatch():
    """Accessing a result before dispatch() raises RuntimeError."""
    a = QuantumFloat(4); a[:] = 1
    b = QuantumFloat(3); b[:] = 2
    c = a + b

    bb = _make_batched_backend()
    res = c.get_measurement(backend=bb)

    with pytest.raises(RuntimeError, match="dispatch"):
        len(res)


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------


def test_exception_propagates_to_all_results():
    """An exception from the wrapped backend must surface when any result is accessed.

    All pending MeasurementResult objects receive the error via _inject_error(),
    so every subsequent access raises, preventing silent data loss.
    """
    from qrisp.interface.backend import Backend
    from qrisp.interface.job import Job, JobStatus

    class _FailJob(Job):
        def submit(self):
            self._last_known_status = JobStatus.ERROR
        def result(self, timeout=None):
            raise RuntimeError("Simulated hardware fault")
        def cancel(self):
            return False
        def status(self):
            return self._last_known_status

    class FailingBackend(Backend):
        def run_async(self, circuits, shots=None):
            raise RuntimeError("Simulated hardware fault")

    bb = FailingBackend().batched()

    a = QuantumFloat(4); a[:] = 1
    b = QuantumFloat(3); b[:] = 2
    c = a + b

    d = QuantumFloat(4); d[:] = 2
    e = QuantumFloat(3); e[:] = 2
    f = d + e

    res_c = c.get_measurement(backend=bb)
    res_f = f.get_measurement(backend=bb)

    with pytest.raises(RuntimeError, match="Simulated hardware fault"):
        bb.dispatch()

    errors = []
    for res in [res_c, res_f]:
        try:
            len(res)
        except RuntimeError as exc:
            errors.append(str(exc))

    assert len(errors) == 2
    assert all("Simulated hardware fault" in e for e in errors)


# ---------------------------------------------------------------------------
# run_async is unsupported
# ---------------------------------------------------------------------------


def test_run_async_raises_not_implemented():
    """BatchedBackend.run_async() must raise NotImplementedError."""
    bb = _make_batched_backend()
    a = QuantumFloat(4); a[:] = 1
    b = QuantumFloat(3); b[:] = 2
    c = a + b
    with pytest.raises(NotImplementedError):
        bb.run_async(c.qs.compile())


# ---------------------------------------------------------------------------
# MeasurementResult unit tests
# ---------------------------------------------------------------------------


def test_measurement_result_raises_before_inject():
    """MeasurementResult must raise RuntimeError if accessed before _inject()."""
    raw = MeasurementResult()
    with pytest.raises(RuntimeError, match="dispatch"):
        len(raw)


def test_measurement_result_is_accessible_after_inject():
    """MeasurementResult is fully readable after _inject()."""
    raw = MeasurementResult()
    raw._inject({"00": 512, "11": 512})
    assert raw["00"] == 512
    assert set(raw) == {"00", "11"}
    assert len(raw) == 2


def test_measurement_result_error_propagates_on_access():
    """After _inject_error(), every access raises the stored exception."""
    raw = MeasurementResult()
    raw._inject_error(RuntimeError("backend error"))
    with pytest.raises(RuntimeError, match="backend error"):
        len(raw)
    with pytest.raises(RuntimeError, match="backend error"):
        iter(raw)


def test_decoded_measurement_result_equals_dict():
    """DecodedMeasurementResult must compare equal to the equivalent plain dict."""
    raw = MeasurementResult()
    raw._inject({"0": 0.5, "1": 0.5})
    decoded = DecodedMeasurementResult(raw, lambda key: int(key, 2))
    assert decoded == {0: 0.5, 1: 0.5}
    assert {0: 0.5, 1: 0.5} == decoded


def test_decoded_measurement_result_repr_pending():
    """repr() on an unpopulated result must include 'pending'."""
    raw = MeasurementResult()
    decoded = DecodedMeasurementResult(raw, lambda k: k)
    assert "pending" in repr(decoded).lower()


# ---------------------------------------------------------------------------
# Docstring examples
# ---------------------------------------------------------------------------


def test_docstring_basic_usage_example():
    """Exact values from the BatchedBackend class docstring example.

    a=1, b=2 → c=3; d=2, e=3 → f=5.
    """
    backend = DefaultBackend()
    bb = backend.batched()

    a = QuantumFloat(4); a[:] = 1
    b = QuantumFloat(3); b[:] = 2
    c = a + b  # expected: 3

    d = QuantumFloat(4); d[:] = 2
    e = QuantumFloat(3); e[:] = 3
    f = d + e  # expected: 5

    res_c = c.get_measurement(backend=bb)
    res_f = f.get_measurement(backend=bb)

    bb.dispatch()

    assert res_c == {3: 1.0}
    assert res_f == {5: 1.0}


def test_batched_measurement_function():
    """batched_measurement helper matches the usage shown in its docstring."""
    from qrisp import batched_measurement

    backend = DefaultBackend()
    bb = backend.batched()

    a = QuantumFloat(4); a[:] = 1
    b = QuantumFloat(3); b[:] = 2
    c = a + b  # expected: 3

    d = QuantumFloat(4); d[:] = 2
    e = QuantumFloat(3); e[:] = 3
    f = d + e  # expected: 5

    results = batched_measurement([c, f], backend=bb)

    assert results[0] == {3: 1.0}
    assert results[1] == {5: 1.0}
