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

"""Tests for StimBackend."""

import pytest
from conftest import CountingWrapper

from qrisp import QuantumCircuit, QuantumFloat, QuantumVariable
from qrisp.interface import BatchedBackend, StimBackend
from qrisp.interface.job import JobFailureError
from qrisp.interface.measurement_result import LazyDict


def _build_deterministic_circuit():
    """Return a 2-qubit Clifford circuit that deterministically yields '11'."""
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.x(1)
    qc.measure([0, 1])
    return qc


def test_stim_backend_run_counts():
    """run() on a Clifford circuit must return the expected bitstring counts."""
    qc = _build_deterministic_circuit()
    backend = StimBackend()
    counts = backend.run(qc, shots=200)
    assert sum(counts.values()) == 200
    assert counts == {"11": 200}


def test_stim_unused_clbit():
    """run() on a Clifford circuit with unused Clbits returns the expected bitstring counts."""
    backend = StimBackend()

    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.measure(0)
    counts = backend.run(qc, shots=100)
    assert counts == {"100": 100}


class TestStimBackendBatched:
    """Verify the four BatchedBackend properties for StimBackend."""

    def test_batched_returns_batched_backend(self):
        """StimBackend.batched() must return a BatchedBackend instance."""
        assert isinstance(StimBackend().batched(), BatchedBackend)

    def test_result_is_lazy_before_dispatch(self):
        """get_measurement must raise RuntimeError before dispatch() is called."""
        bb = StimBackend().batched()
        qv = QuantumVariable(2)
        qv[:] = "10"
        res = qv.get_measurement(backend=bb)
        assert isinstance(res, LazyDict)
        with pytest.raises(RuntimeError, match="dispatch"):
            len(res)
        bb.dispatch()
        assert res == {"10": 1.0}

    def test_dispatch_populates_two_results(self):
        """dispatch() must populate both results correctly for two Clifford circuits."""
        bb = StimBackend().batched()

        qf1 = QuantumFloat(3)
        qf1[:] = 3
        qf2 = QuantumFloat(3)
        qf2[:] = 5

        res1 = qf1.get_measurement(backend=bb)
        res2 = qf2.get_measurement(backend=bb)
        bb.dispatch()

        assert res1 == {3: 1.0}
        assert res2 == {5: 1.0}

    def test_n_circuits_one_run_async_call(self):
        """Two Clifford circuits queued with StimBackend must produce exactly one run_async call."""
        counting = CountingWrapper(StimBackend())
        bb = counting.batched()

        qf1 = QuantumFloat(3)
        qf1[:] = 1
        qf2 = QuantumFloat(3)
        qf2[:] = 2

        qf1.get_measurement(backend=bb)
        qf2.get_measurement(backend=bb)

        assert counting.run_async_call_count == 0
        bb.dispatch()
        assert counting.run_async_call_count == 1


def test_stim_failure_raises_job_failure_error():
    """result() must raise JobFailureError (not a raw exception) when Stim cannot run the circuit."""
    backend = StimBackend()
    qc = QuantumCircuit(1, 1)
    qc.t(0)  # T gate is non-Clifford; to_stim() will fail
    qc.measure(0, 0)
    job = backend.run_async(qc, shots=100)
    with pytest.raises(JobFailureError):
        job.result()


def test_stim_failure_error_message_contains_cause():
    """The JobFailureError message must include the underlying exception's description."""
    backend = StimBackend()
    qc = QuantumCircuit(1, 1)
    qc.t(0)
    qc.measure(0, 0)
    job = backend.run_async(qc, shots=100)
    with pytest.raises(JobFailureError) as exc_info:
        job.result()
    assert exc_info.value.__cause__ is not None


def test_stim_run_async_with_shots_list_uses_per_circuit_counts():
    """run_async with a list of shots must run each circuit with its own shot count."""
    qc = _build_deterministic_circuit()
    backend = StimBackend()

    job = backend.run_async([qc, qc], shots=[100, 300])
    result = job.result()

    assert sum(result.all_counts[0].values()) == 100
    assert sum(result.all_counts[1].values()) == 300
