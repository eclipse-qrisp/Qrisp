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

"""Tests for QrispSimulatorBackend and QrispSimulatorJob."""

import pytest
from conftest import CountingWrapper

from qrisp import QuantumCircuit, QuantumFloat, h
from qrisp.circuit import Operation
from qrisp.circuit.pass_management import CircuitPass, PassManager
from qrisp.default_backend import QrispSimulatorBackend, QrispSimulatorJob, def_backend
from qrisp.interface import BatchedBackend
from qrisp.interface.job import JobFailureError, JobResult, JobStatus
from qrisp.interface.measurement_result import LazyDict


def _simple_computation():
    """Return a deterministic QuantumFloat computation (3 * 3 = 9)."""
    qf = QuantumFloat(4)
    qf[:] = 3
    return qf * qf


class TestQrispSimulatorJobInterface:
    """Tests that run_async() returns a proper QrispSimulatorJob and the Job interface is correct.

    Note: run_async() is used here (rather than run()) because these tests need the
    Job handle. run() returns a plain dict for backward compatibility and does not
    expose the Job object.
    """

    def test_run_async_returns_qrisp_simulator_job(self):
        """Ensure run_async() returns a QrispSimulatorJob instance, not a raw dict."""
        backend = QrispSimulatorBackend()
        res = _simple_computation()
        job = backend.run_async(res.qs.compile())
        assert isinstance(job, QrispSimulatorJob)

    def test_job_is_done_before_run_async_returns(self):
        """Ensure the job is already DONE when run_async() returns to the caller."""
        backend = QrispSimulatorBackend()
        res = _simple_computation()
        job = backend.run_async(res.qs.compile())
        assert job.status() == JobStatus.DONE

    def test_job_done_returns_true(self):
        """Ensure done() is True after synchronous execution."""
        backend = QrispSimulatorBackend()
        res = _simple_computation()
        job = backend.run_async(res.qs.compile())
        assert job.done() is True

    def test_job_in_final_state_returns_true(self):
        """Ensure in_final_state() is True after synchronous execution."""
        backend = QrispSimulatorBackend()
        res = _simple_computation()
        job = backend.run_async(res.qs.compile())
        assert job.in_final_state() is True

    def test_result_returns_job_result_instance(self):
        """Ensure job.result() returns a JobResult object."""
        backend = QrispSimulatorBackend()
        res = _simple_computation()
        job = backend.run_async(res.qs.compile())
        assert isinstance(job.result(), JobResult)

    def test_cancel_returns_false(self):
        """Ensure cancel() always returns False for a synchronous job."""
        backend = QrispSimulatorBackend()
        res = _simple_computation()
        job = backend.run_async(res.qs.compile())
        assert job.cancel() is False

    def test_status_unchanged_after_cancel(self):
        """Ensure cancel() does not change the job status."""
        backend = QrispSimulatorBackend()
        res = _simple_computation()
        job = backend.run_async(res.qs.compile())
        job.cancel()
        assert job.status() == JobStatus.DONE

    def test_failure_raises_job_failure_error(self):
        """result() must raise JobFailureError (not a raw simulator exception) on failure."""
        backend = QrispSimulatorBackend()
        qc = QuantumCircuit(1, 1)
        qc.append(Operation(name="unknown_gate", num_qubits=1), [0])
        qc.measure(0, 0)
        job = backend.run_async(qc)
        with pytest.raises(JobFailureError):
            job.result()

    def test_failure_error_message_contains_cause(self):
        """The JobFailureError message must include the underlying exception's description."""
        backend = QrispSimulatorBackend()
        qc = QuantumCircuit(1, 1)
        qc.append(Operation(name="unknown_gate", num_qubits=1), [0])
        qc.measure(0, 0)
        job = backend.run_async(qc)
        with pytest.raises(JobFailureError, match="unknown_gate"):
            job.result()

    def test_failure_error_chains_original_exception(self):
        """The original simulator exception must be chained as __cause__ of JobFailureError."""
        backend = QrispSimulatorBackend()
        qc = QuantumCircuit(1, 1)
        qc.append(Operation(name="unknown_gate", num_qubits=1), [0])
        qc.measure(0, 0)
        job = backend.run_async(qc)
        with pytest.raises(JobFailureError) as exc_info:
            job.result()
        assert exc_info.value.__cause__ is not None


class TestQrispSimulatorBackendAnalytic:
    """Tests for analytic (exact probability) execution with shots=None."""

    def test_analytic_result_is_correct(self):
        """Verify that a deterministic computation yields probability 1.0."""
        backend = QrispSimulatorBackend()  # shots=None by default
        res = _simple_computation()
        assert res.get_measurement(backend=backend) == {9: 1.0}

    def test_analytic_probabilities_sum_to_one(self):
        """Verify that analytic result probabilities sum to 1.0."""
        backend = QrispSimulatorBackend()
        res = _simple_computation()
        result = backend.run_async(res.qs.compile()).result()
        total = sum(result.get_counts().values())
        assert abs(total - 1.0) < 1e-9

    def test_analytic_result_contains_floats(self):
        """Verify that analytic results are floating-point probabilities, not integer counts."""
        backend = QrispSimulatorBackend()
        res = _simple_computation()
        result = backend.run_async(res.qs.compile()).result()
        for value in result.get_counts().values():
            assert isinstance(value, float)


class TestQrispSimulatorBackendShots:
    """Tests for shot-based (sampled) execution."""

    def test_shot_based_differs_from_analytic(self):
        """Verify that shot-based execution gives a different result than analytic execution."""
        qv = QuantumFloat(1)
        h(qv[0])  # equal superposition of 0 and 1

        backend = QrispSimulatorBackend()  # shots=None
        analytic_result = qv.get_measurement(backend=backend)
        assert analytic_result == {0: 0.5, 1: 0.5}

        backend.update_options(shots=1)
        sampled_result = qv.get_measurement(backend=backend)

        # With 1 shot the outcome is definite: one key with probability 1.0.
        assert len(sampled_result) == 1
        assert list(sampled_result.values())[0] == 1.0
        assert sampled_result != analytic_result

    def test_backend_shots_option_used_when_not_overridden(self):
        """Verify that the backend's shots option is used and produces a sampled result."""
        qv = QuantumFloat(1)
        h(qv[0])

        backend = QrispSimulatorBackend()
        backend.update_options(shots=1)
        result = qv.get_measurement(backend=backend)

        # With 1 shot the outcome must be a single definite value.
        assert len(result) == 1
        assert list(result.values())[0] == 1.0


class TestQrispSimulatorBackendOptions:
    """Tests for QrispSimulatorBackend options handling."""

    def test_default_shots_is_none(self):
        """Ensure the default shots option is None (analytic execution)."""
        assert QrispSimulatorBackend().options["shots"] is None

    def test_default_token_is_empty_string(self):
        """Ensure the default token option is an empty string."""
        assert QrispSimulatorBackend().options["token"] == ""

    def test_update_options_shots(self):
        """Ensure shots can be updated via update_options."""
        backend = QrispSimulatorBackend()
        backend.update_options(shots=512)
        assert backend.options["shots"] == 512

    def test_update_options_invalid_key_raises(self):
        """Ensure update_options raises AttributeError for an unknown key."""
        backend = QrispSimulatorBackend()
        with pytest.raises(AttributeError):
            backend.update_options(nonexistent=42)

    def test_options_independent_between_instances(self):
        """Updating one instance must not affect another independent instance."""
        b1 = QrispSimulatorBackend()
        b2 = QrispSimulatorBackend()
        b1.update_options(shots=512)
        assert b2.options["shots"] is None

    def test_module_level_def_backend_is_qrisp_simulator(self):
        """Ensure the module-level def_backend is a QrispSimulatorBackend instance."""
        assert isinstance(def_backend, QrispSimulatorBackend)


class TestQrispSimulatorBackendBatched:
    """Verify that QrispSimulatorBackend.batched() produces correct lazy results."""

    def test_batched_returns_batched_backend(self):
        """batched() must return a BatchedBackend wrapping the original backend."""
        backend = QrispSimulatorBackend()
        bb = backend.batched()
        assert isinstance(bb, BatchedBackend)
        assert bb._backend is backend

    def test_result_is_lazy_before_dispatch(self):
        """get_measurement via batched QrispSimulatorBackend must raise before dispatch."""
        bb = QrispSimulatorBackend().batched()
        res = _simple_computation().get_measurement(backend=bb)
        assert isinstance(res, LazyDict)
        with pytest.raises(RuntimeError, match="dispatch"):
            len(res)
        bb.dispatch()
        assert res == {9: 1.0}

    def test_dispatch_populates_two_results(self):
        """dispatch() must populate both results correctly for two independent circuits."""
        qf1 = QuantumFloat(3)
        qf1[:] = 3
        qf2 = QuantumFloat(3)
        qf2[:] = 5

        bb = QrispSimulatorBackend().batched()
        res1 = qf1.get_measurement(backend=bb)
        res2 = qf2.get_measurement(backend=bb)
        bb.dispatch()

        assert res1 == {3: 1.0}
        assert res2 == {5: 1.0}

    def test_n_circuits_one_run_async_call(self):
        """Two circuits queued with QrispSimulatorBackend must produce exactly one run_async call."""
        counting = CountingWrapper(QrispSimulatorBackend())
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


# ---------------------------------------------------------------------------
# Tests for PassManager (pm) integration
# ---------------------------------------------------------------------------


def _prepend_x_on_first_qubit(qc):
    """Insert an X gate at the beginning of the circuit, before any measurements."""
    from qrisp.circuit import Instruction, XGate

    qc.data.insert(0, Instruction(XGate(), [qc.qubits[0]]))
    return qc


class TestQrispSimulatorBackendPassManager:
    """Tests for the pm (PassManager) keyword argument on QrispSimulatorBackend."""

    def test_construct_with_pm_none(self):
        """Backend constructed with pm=None must behave normally."""
        backend = QrispSimulatorBackend(pm=None)
        qf = QuantumFloat(2)
        qf[:] = 3
        assert qf.get_measurement(backend=backend) == {3: 1.0}

    def test_construct_with_pm(self):
        """Backend constructed with a PassManager must accept it without error."""
        pm = PassManager()
        backend = QrispSimulatorBackend(pm=pm)
        qf = QuantumFloat(2)
        qf[:] = 3
        assert qf.get_measurement(backend=backend) == {3: 1.0}

    def test_pm_pass_is_called(self):
        """Passes in the pm must be invoked for each circuit submitted."""
        call_counter = [0]

        def _counting_pass(qc):
            call_counter[0] += 1
            return qc

        pm = PassManager()
        pm += CircuitPass(_counting_pass)
        backend = QrispSimulatorBackend(pm=pm)

        qf = QuantumFloat(2)
        qf[:] = 3
        qf.get_measurement(backend=backend)
        assert call_counter[0] == 1

        qc = QuantumCircuit(2, 2)
        qc.measure(qc.qubits, qc.clbits)
        backend.run([qc, qc.copy(), qc.copy()])
        assert call_counter[0] == 4

    def test_pm_modifies_circuit_before_simulation(self):
        """A pass that prepends an X gate must change the measurement outcome."""
        pm = PassManager()
        pm += CircuitPass(_prepend_x_on_first_qubit)
        backend = QrispSimulatorBackend(pm=pm)

        # |00⟩ → X prepended on qubit 0 → |01⟩  (qubit 0 is LSB, rightmost bit)
        qc = QuantumCircuit(2, 2)
        qc.measure(qc.qubits, qc.clbits)
        assert backend.run(qc) == {"01": 1.0}

    def test_pm_is_applied_to_every_circuit(self):
        """Simulating multiple circuits must apply pm to each one."""
        pm = PassManager()
        pm += CircuitPass(_prepend_x_on_first_qubit)
        backend = QrispSimulatorBackend(pm=pm)

        qc_zero = QuantumCircuit(2, 2)
        qc_zero.measure(qc_zero.qubits, qc_zero.clbits)

        qc_one = QuantumCircuit(2, 2)
        qc_one.x(qc_one.qubits[0])  # built-in X on qubit 0 → |01⟩
        qc_one.measure(qc_one.qubits, qc_one.clbits)

        results = backend.run([qc_zero, qc_one])
        # qc_zero: |00⟩ + prepended X on qubit 0 → |01⟩
        assert results[0] == {"01": 1.0}
        # qc_one: |01⟩ + prepended X on qubit 0 → |00⟩ (X·X = I)
        assert results[1] == {"00": 1.0}

    def test_empty_pm_is_identity(self):
        """An empty PassManager must not alter circuit behaviour."""
        backend = QrispSimulatorBackend(pm=PassManager())
        qf = QuantumFloat(2)
        qf[:] = 3
        assert qf.get_measurement(backend=backend) == {3: 1.0}

    def test_pm_chain_executes_in_order(self):
        """Multiple passes in a PassManager must execute in the given order."""
        pm = PassManager()
        # Two prepended X gates on qubit 0 cancel
        pm += CircuitPass(_prepend_x_on_first_qubit)
        pm += CircuitPass(_prepend_x_on_first_qubit)
        backend = QrispSimulatorBackend(pm=pm)

        qc = QuantumCircuit(2, 2)
        qc.measure(qc.qubits, qc.clbits)
        assert backend.run(qc) == {"00": 1.0}

    def test_pm_works_with_shot_based_execution(self):
        """PassManager must work with shot-based (non-analytic) execution."""
        pm = PassManager()
        pm += CircuitPass(_prepend_x_on_first_qubit)
        backend = QrispSimulatorBackend(pm=pm)

        qc = QuantumCircuit(2, 2)
        qc.measure(qc.qubits, qc.clbits)
        assert backend.run(qc, shots=100) == {"01": 100}

    def test_pm_works_with_batched_backend(self):
        """PassManager must be applied when circuits are dispatched via BatchedBackend."""
        pm = PassManager()
        pm += CircuitPass(_prepend_x_on_first_qubit)
        backend = QrispSimulatorBackend(pm=pm)
        bb = backend.batched()

        qc = QuantumCircuit(2, 2)
        qc.measure(qc.qubits, qc.clbits)
        result = bb.run(qc)
        bb.dispatch()
        assert result == {"01": 1.0}

    def test_pm_works_with_analytic_execution(self):
        """PassManager must work with analytic execution (shots=None)."""
        pm = PassManager()
        pm += CircuitPass(_prepend_x_on_first_qubit)
        backend = QrispSimulatorBackend(pm=pm)

        qc = QuantumCircuit(3, 3)
        qc.measure(qc.qubits, qc.clbits)
        assert backend.run(qc) == {"001": 1.0}

    def test_pm_rejects_non_passmanager(self):
        """Constructing with a non-PassManager pm must raise TypeError."""
        with pytest.raises(TypeError):
            QrispSimulatorBackend(pm="not_a_pass_manager")
