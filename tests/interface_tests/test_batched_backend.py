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

"""Tests for BatchedBackend."""


import pytest
from conftest import CountingWrapper

from qrisp import QuantumCircuit, QuantumFloat, batched_measurement
from qrisp.default_backend import QrispSimulatorBackend
from qrisp.interface import BatchedBackend
from qrisp.interface.backend import Backend
from qrisp.interface.job import Job, JobResult, JobStatus
from qrisp.interface.measurement_result import (
    DecodedMeasurementResult,
    LazyDict,
    MeasurementResult,
)


def _make_batched_backend():
    """Return a fresh BatchedBackend wrapping QrispSimulatorBackend."""
    return QrispSimulatorBackend().batched()


def test_batched_returns_batched_backend_instance():
    """Backend.batched() must return a BatchedBackend."""
    backend = QrispSimulatorBackend()
    bb = backend.batched()
    assert isinstance(bb, BatchedBackend)
    assert bb._backend is backend


def test_explicit_dispatch_batches_two_variables():
    """Two get_measurement calls followed by dispatch() must both return correct results.

    This is the primary integration test: no threading needed because
    get_measurement() returns a lazy result immediately and dispatch() populates
    both results by calling the wrapped backend once per circuit.
    """
    a = QuantumFloat(4)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 2
    c = a + b  # expected: 3

    d = QuantumFloat(4)
    d[:] = 2
    e = QuantumFloat(3)
    e[:] = 2
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
    a = QuantumFloat(4)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 2
    c = a + b  # expected: 3

    bb = _make_batched_backend()
    res = c.get_measurement(backend=bb)
    bb.dispatch()

    assert res == {3: 1.0}


def test_result_is_lazy_before_dispatch():
    """Accessing a result before dispatch() raises RuntimeError."""
    a = QuantumFloat(4)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 2
    c = a + b

    bb = _make_batched_backend()
    res = c.get_measurement(backend=bb)

    with pytest.raises(RuntimeError, match="dispatch"):
        len(res)


def _det_circuit(value: int = 1) -> QuantumCircuit:
    """Return a 1-qubit deterministic circuit: X (when value=1) then measure.

    - value=1 → always measures '1'
    - value=0 → always measures '0'.
    """
    qc = QuantumCircuit(1, 1)
    if value:
        qc.x(0)
    qc.measure(0, 0)
    return qc


def test_run_with_multiple_circuits_returns_list_of_measurement_results():
    """run([c1, c2]) must return a list of two MeasurementResult objects, not a single one."""
    bb = _make_batched_backend()
    results = bb.run([_det_circuit(), _det_circuit()])
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, MeasurementResult) for r in results)


def test_run_with_multiple_circuits_increments_pending_count_by_n():
    """run([c1, c2]) must add two entries to the queue, not one."""
    bb = _make_batched_backend()
    assert bb.pending_count == 0
    bb.run([_det_circuit(), _det_circuit()])
    assert bb.pending_count == 2


def test_run_with_multiple_circuits_results_are_lazy_before_dispatch():
    """All results from run([c1, c2]) must be inaccessible until dispatch() is called."""
    bb = _make_batched_backend()
    results = bb.run([_det_circuit(), _det_circuit()])
    for r in results:
        with pytest.raises(RuntimeError, match="dispatch"):
            len(r)


def test_run_with_multiple_circuits_all_results_populated_after_dispatch():
    """After dispatch(), every result from run([c1, c2]) contains the correct counts."""
    bb = _make_batched_backend()
    shots = 100
    results = bb.run([_det_circuit(value=0), _det_circuit(value=1)], shots=shots)
    bb.dispatch()
    assert results[0]["0"] == shots
    assert results[1]["1"] == shots


class _FailingBackend(Backend):
    """Backend whose run_async always raises RuntimeError."""

    def run_async(self, circuits, shots: int | list[int] | None = None):
        """Raise RuntimeError unconditionally."""
        raise RuntimeError("Simulated hardware fault")


def test_exception_propagates_to_all_results():
    """An exception from the wrapped backend must surface when any result is accessed.

    All pending MeasurementResult objects receive the error via _inject_error(),
    so every subsequent access raises, preventing silent data loss.
    """
    bb = _FailingBackend().batched()

    a = QuantumFloat(4)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 2
    c = a + b

    d = QuantumFloat(4)
    d[:] = 2
    e = QuantumFloat(3)
    e[:] = 2
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


def test_docstring_basic_usage_example():
    """Exact values from the BatchedBackend class docstring example.

    a=1, b=2 → c=3; d=2, e=3 → f=5.
    """
    backend = QrispSimulatorBackend()
    bb = backend.batched()

    a = QuantumFloat(4)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 2
    c = a + b  # expected: 3

    d = QuantumFloat(4)
    d[:] = 2
    e = QuantumFloat(3)
    e[:] = 3
    f = d + e  # expected: 5

    res_c = c.get_measurement(backend=bb)
    res_f = f.get_measurement(backend=bb)

    bb.dispatch()

    assert res_c == {3: 1.0}
    assert res_f == {5: 1.0}


def test_batched_measurement_function():
    """batched_measurement helper matches the usage shown in its docstring."""
    backend = QrispSimulatorBackend()
    bb = backend.batched()

    a = QuantumFloat(4)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 2
    c = a + b  # expected: 3

    d = QuantumFloat(4)
    d[:] = 2
    e = QuantumFloat(3)
    e[:] = 3
    f = d + e  # expected: 5

    results = batched_measurement([c, f], backend=bb)

    assert results[0] == {3: 1.0}
    assert results[1] == {5: 1.0}


def test_options_delegates_to_wrapped_backend():
    """bb.options must reflect the wrapped backend's options, not an independent copy.

    Changes made directly to the backend must be visible through bb.options,
    proving that no separate copy is held.
    """
    backend = QrispSimulatorBackend()
    bb = backend.batched()

    assert bb.options["shots"] == backend.options["shots"]

    backend.update_options(shots=512)
    assert bb.options["shots"] == 512


def test_update_options_valid_key():
    """update_options() with a valid key must update both bb.options and backend.options."""
    backend = QrispSimulatorBackend()
    bb = backend.batched()

    bb.update_options(shots=512)

    assert bb.options["shots"] == 512
    assert backend.options["shots"] == 512


def test_update_options_invalid_key_raises():
    """update_options() with an unknown key must raise AttributeError."""
    bb = _make_batched_backend()
    with pytest.raises(AttributeError, match="not_a_real_option"):
        bb.update_options(not_a_real_option=42)


def test_pending_count_starts_at_zero():
    """A freshly created BatchedBackend must have pending_count == 0."""
    bb = _make_batched_backend()
    assert bb.pending_count == 0


def test_pending_count_increments_on_run():
    """Each get_measurement() call must increment pending_count by one."""
    bb = _make_batched_backend()

    a = QuantumFloat(4)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 2
    c = a + b

    d = QuantumFloat(4)
    d[:] = 2
    e = QuantumFloat(3)
    e[:] = 3
    f = d + e

    c.get_measurement(backend=bb)
    assert bb.pending_count == 1

    f.get_measurement(backend=bb)
    assert bb.pending_count == 2


def test_pending_count_resets_after_dispatch():
    """dispatch() must drain the queue, leaving pending_count == 0."""
    bb = _make_batched_backend()

    a = QuantumFloat(4)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 2
    c = a + b

    c.get_measurement(backend=bb)
    assert bb.pending_count == 1

    bb.dispatch()
    assert bb.pending_count == 0


def test_dispatch_on_empty_queue_is_a_noop():
    """dispatch() on an empty queue must not raise and pending_count stays 0."""
    bb = _make_batched_backend()
    bb.dispatch()
    assert bb.pending_count == 0


def test_clear_empties_the_queue():
    """clear() must set pending_count to 0 without dispatching."""
    bb = _make_batched_backend()

    a = QuantumFloat(4)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 2
    c = a + b

    c.get_measurement(backend=bb)
    assert bb.pending_count == 1

    bb.clear()
    assert bb.pending_count == 0


def test_clear_leaves_results_unpopulated():
    """Results returned before clear() must remain unpopulated (RuntimeError on access)."""
    bb = _make_batched_backend()

    a = QuantumFloat(4)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 2
    c = a + b

    res = c.get_measurement(backend=bb)
    bb.clear()

    with pytest.raises(RuntimeError, match="dispatch"):
        len(res)


def test_dispatch_with_timeout_succeeds_for_fast_backend():
    """dispatch(timeout=30) must complete normally for a local simulator."""
    bb = _make_batched_backend()

    a = QuantumFloat(4)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 2
    c = a + b

    res = c.get_measurement(backend=bb)
    bb.dispatch(timeout=30)

    assert res == {3: 1.0}


class _RecordingJob(Job):
    """Job that records the timeout value passed to result() and returns a fixed count."""

    def __init__(self, backend, received_timeouts):
        """Initialise with a reference to the shared list that records timeouts."""
        super().__init__(backend)
        self._received_timeouts = received_timeouts

    def submit(self):
        """Mark job as running."""
        self._last_known_status = JobStatus.RUNNING

    def result(self, timeout=None):
        """Record the timeout and return a single-circuit dummy result."""
        self._received_timeouts.append(timeout)
        self._last_known_status = JobStatus.DONE
        return JobResult([{"0": 1024}])

    def cancel(self):
        """Cancel is not supported; always returns False."""
        return False

    def status(self):
        """Return the last known status."""
        return self._last_known_status


class _RecordingBackend(Backend):
    """Backend that creates _RecordingJob instances to capture Job.result() arguments."""

    def __init__(self, received_timeouts):
        """Initialise with a reference to the shared list that records timeouts."""
        super().__init__()
        self._received_timeouts = received_timeouts

    def run_async(self, circuits, shots: int | list[int] | None = None):
        """Submit circuits by creating a _RecordingJob."""
        job = _RecordingJob(self, self._received_timeouts)
        job.submit()
        return job


def test_dispatch_timeout_propagates_to_job_result():
    """dispatch(timeout=...) must pass the timeout value through to Job.result()."""
    received_timeouts = []
    bb = _RecordingBackend(received_timeouts).batched()

    a = QuantumFloat(4)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 2
    c = a + b
    c.get_measurement(backend=bb)

    bb.dispatch(timeout=99)

    assert received_timeouts == [99]


def test_dispatch_mixed_shots_uses_single_run_async_call():
    """dispatch() must issue exactly one run_async call even when circuits use different shot counts."""

    counting = CountingWrapper(QrispSimulatorBackend())
    bb = counting.batched()

    a = QuantumFloat(4)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 2
    c = a + b

    d = QuantumFloat(4)
    d[:] = 2
    e = QuantumFloat(3)
    e[:] = 3
    f = d + e

    c.get_measurement(backend=bb, shots=100)
    f.get_measurement(backend=bb, shots=200)

    assert counting.run_async_call_count == 0
    bb.dispatch()
    assert counting.run_async_call_count == 1


def test_dispatch_mixed_shots_passes_shots_list_to_run_async():
    """dispatch() must forward the per-circuit shot counts as a list to run_async."""

    counting = CountingWrapper(QrispSimulatorBackend())
    bb = counting.batched()

    a = QuantumFloat(4)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 2
    c = a + b

    d = QuantumFloat(4)
    d[:] = 2
    e = QuantumFloat(3)
    e[:] = 3
    f = d + e

    c.get_measurement(backend=bb, shots=100)
    f.get_measurement(backend=bb, shots=200)
    bb.dispatch()

    assert counting.shots_received == [[100, 200]]


def test_dispatch_uniform_shots_passes_scalar_to_run_async():
    """dispatch() must pass a scalar when all circuits share the same shot count."""

    counting = CountingWrapper(QrispSimulatorBackend())
    bb = counting.batched()

    a = QuantumFloat(4)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 2
    c = a + b

    d = QuantumFloat(4)
    d[:] = 2
    e = QuantumFloat(3)
    e[:] = 3
    f = d + e

    c.get_measurement(backend=bb, shots=100)
    f.get_measurement(backend=bb, shots=100)
    bb.dispatch()

    assert counting.shots_received == [100]


def test_dispatch_mixed_shots_returns_correct_results():
    """dispatch() with mixed shot counts must still return correct results for each circuit."""
    bb = _make_batched_backend()

    a = QuantumFloat(4)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 2
    c = a + b  # expected: 3

    d = QuantumFloat(4)
    d[:] = 2
    e = QuantumFloat(3)
    e[:] = 3
    f = d + e  # expected: 5

    res_c = c.get_measurement(backend=bb, shots=100)
    res_f = f.get_measurement(backend=bb, shots=200)
    bb.dispatch()

    assert res_c == {3: 1.0}
    assert res_f == {5: 1.0}


def test_run_async_shots_list_length_mismatch_raises():
    """run_async with a shots list whose length differs from the circuit count must raise ValueError."""
    from qrisp import QuantumCircuit

    backend = QrispSimulatorBackend()

    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure(0)

    with pytest.raises(ValueError, match="shots.*list"):
        backend.run_async([qc, qc], shots=[100])  # 2 circuits, 1 shot value


def test_hamiltonian_measurement_uses_single_run_async_call():
    """Measuring a Hermitian observable via BatchedBackend must produce exactly one run_async call."""
    from qrisp import QuantumVariable
    from qrisp.operators import X, Z

    qv = QuantumVariable(1)  # |0> state

    # Unequal coefficients → different operator variances → different shot budgets
    # per group (~799 shots for 3*X(0), ~266 shots for Z(0) at precision=0.1).
    H = 3 * X(0) + Z(0)

    counting = CountingWrapper(QrispSimulatorBackend())
    bb = counting.batched()

    H.get_measurement(qv, precision=0.1, backend=bb)

    # One run_async call regardless of how many commuting groups
    # the Hamiltonian decomposes into
    assert counting.run_async_call_count == 1
    # The shots list must be non-uniform, confirming dispatch() took the
    # per-circuit-shots path rather than the "all equal → scalar" branch.
    assert isinstance(counting.shots_received[0], list)
    assert len(set(counting.shots_received[0])) > 1
