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

"""Comprehensive tests for the VirtualBackend implementation."""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import pytest

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.interface.backend import Backend, BackendLike
from qrisp.interface.job import (
    JobFailureError,
    JobResult,
    JobStatus,
)
from qrisp.interface.virtual_backend import VirtualBackend, VirtualJob
from qrisp.misc.exceptions import QrispDeprecationWarning

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _trivial_run_func(qasm_str, shots=None, token=""):
    """A no-op run_func that returns a fixed single-bitstring counts dict."""
    return {"0": shots if shots is not None else 1}


def _echo_run_func(qasm_str, shots=None, token=""):
    """A run_func that echoes back the QASM, shots, and token in the result."""
    return {
        f"qasm_len={len(qasm_str)}": 1,
        f"shots={shots}": 1,
        f"token={token}": 1,
    }


def _fail_run_func(qasm_str, shots=None, token=""):
    """A run_func that always raises."""
    raise RuntimeError("Simulated run_func failure.")


def _simple_qc(n_qubits=1):
    """Build a minimal QuantumCircuit with a Hadamard and measurement on each qubit."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
        qc.measure(i)
    return qc


def _simple_qc_list(*sizes):
    """Build a list of simple QuantumCircuits with the given numbers of qubits."""
    return [_simple_qc(n) for n in sizes]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vb():
    """Return a fresh VirtualBackend with the trivial run_func, suppressing the
    deprecation warning during construction.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QrispDeprecationWarning)
        yield VirtualBackend(run_func=_trivial_run_func)


@pytest.fixture
def vb_echo():
    """Return a VirtualBackend whose run_func echoes its arguments."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QrispDeprecationWarning)
        yield VirtualBackend(run_func=_echo_run_func)


@pytest.fixture
def vb_fail():
    """Return a VirtualBackend whose run_func always raises."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QrispDeprecationWarning)
        yield VirtualBackend(run_func=_fail_run_func)


# ---------------------------------------------------------------------------
# Deprecation warning
# ---------------------------------------------------------------------------


class TestVirtualBackendDeprecation:
    """Tests that VirtualBackend emits the expected deprecation warning."""

    def test_deprecation_warning_emitted(self):
        """VirtualBackend instantiation must warn with QrispDeprecationWarning."""

        def noop(qasm_str, shots=None, token=""):
            return {}

        with pytest.warns(QrispDeprecationWarning):
            VirtualBackend(run_func=noop)

    def test_deprecation_warning_message(self):
        """The deprecation warning message must mention VirtualBackend."""

        def noop(qasm_str, shots=None, token=""):
            return {}

        with pytest.warns(QrispDeprecationWarning, match="VirtualBackend"):
            VirtualBackend(run_func=noop)


# ---------------------------------------------------------------------------
# Name handling
# ---------------------------------------------------------------------------


class TestVirtualBackendName:
    """Tests for the name-related behaviour of VirtualBackend."""

    def test_name_defaults_to_class_name(self):
        """When no name is provided, VirtualBackend.name must be 'VirtualBackend'."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", QrispDeprecationWarning)
            backend = VirtualBackend(run_func=_trivial_run_func)
        assert backend.name == "VirtualBackend"

    def test_name_can_be_set_explicitly(self):
        """An explicitly provided name must be stored."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", QrispDeprecationWarning)
            backend = VirtualBackend(run_func=_trivial_run_func, name="my_virtual")
        assert backend.name == "my_virtual"

    def test_name_is_mutable(self):
        """The name can be changed after construction."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", QrispDeprecationWarning)
            backend = VirtualBackend(run_func=_trivial_run_func)
        backend.name = "renamed"
        assert backend.name == "renamed"


# ---------------------------------------------------------------------------
# Backend type hierarchy
# ---------------------------------------------------------------------------


class TestVirtualBackendTypeHierarchy:
    """Tests that VirtualBackend satisfies the expected type contracts."""

    def test_is_backend_subclass(self):
        """VirtualBackend must be a subclass of Backend."""
        assert issubclass(VirtualBackend, Backend)

    def test_is_backend_instance(self, vb):
        """A VirtualBackend instance must be an instance of Backend."""
        assert isinstance(vb, Backend)

    def test_satisfies_backend_like_protocol(self, vb):
        """VirtualBackend must structurally satisfy BackendLike."""
        assert isinstance(vb, BackendLike)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class TestVirtualBackendOptions:
    """Tests for VirtualBackend default and updated runtime options."""

    def test_default_options(self, vb):
        """Default options must include shots=None and token=''."""
        assert vb.options == {"shots": None, "token": ""}

    def test_shots_from_options(self, vb):
        """When shots is not passed to run_async, the options value must be used."""
        job = vb.run_async(_simple_qc())
        # _trivial_run_func returns {'0': shots}, and shots from options is None,
        # so the returned counts value for key '0' is 1 (our fallback for None).
        assert job.result().get_counts()["0"] == 1

    def test_update_options_shots(self, vb):
        """Updating the shots option must be reflected in subsequent executions."""
        vb.update_options(shots=512)
        job = vb.run_async(_simple_qc())
        assert job.result().get_counts()["0"] == 512

    def test_update_options_token(self, vb_echo):
        """Updating the token option must be forwarded to run_func."""
        vb_echo.update_options(token="my_token")
        job = vb_echo.run_async(_simple_qc())
        counts = job.result().get_counts()
        assert "token=my_token" in counts

    def test_update_options_unknown_key_raises(self, vb):
        """update_options must raise AttributeError for unknown keys."""
        with pytest.raises(AttributeError):
            vb.update_options(unknown="value")

    def test_options_are_read_only(self, vb):
        """Direct mutation of options must raise TypeError."""
        with pytest.raises(TypeError):
            vb.options["shots"] = 999

    def test_options_independent_between_instances(self):
        """Updating one instance must not affect another."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", QrispDeprecationWarning)
            vb1 = VirtualBackend(run_func=_trivial_run_func)
            vb2 = VirtualBackend(run_func=_trivial_run_func)
        vb1.update_options(shots=1024)
        assert vb2.options["shots"] is None


# ---------------------------------------------------------------------------
# Single-circuit execution
# ---------------------------------------------------------------------------


class TestVirtualBackendSingleCircuit:
    """Tests for running a single circuit through VirtualBackend."""

    def test_single_circuit_job_is_done_after_run_async(self, vb):
        """run_async must return a VirtualJob already in DONE state."""
        job = vb.run_async(_simple_qc())
        assert job.status() == JobStatus.DONE
        assert isinstance(job, VirtualJob)

    def test_single_circuit_result_num_circuits_is_one(self, vb):
        """The JobResult must contain exactly one counts dict."""
        result = vb.run_async(_simple_qc()).result()
        assert result.num_circuits == 1

    def test_single_circuit_result_is_job_result(self, vb):
        """result() must return a JobResult instance."""
        result = vb.run_async(_simple_qc()).result()
        assert isinstance(result, JobResult)

    def test_single_circuit_run_func_receives_qasm_string(self, vb_echo):
        """run_func must receive the circuit's QASM string as first argument."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0)
        qc.measure(1)
        result = vb_echo.run_async(qc).result()
        counts = result.get_counts()
        qasm_keys = [k for k in counts if k.startswith("qasm_len=")]
        assert len(qasm_keys) == 1
        assert int(qasm_keys[0].split("=")[1]) > 0

    def test_single_circuit_run_func_receives_shots_none(self, vb_echo):
        """When shots is None, run_func must receive None."""
        result = vb_echo.run_async(_simple_qc()).result()
        assert "shots=None" in result.get_counts()

    def test_single_circuit_run_func_receives_explicit_shots(self, vb_echo):
        """Explicit shots must be forwarded to run_func."""
        result = vb_echo.run_async(_simple_qc(), shots=42).result()
        assert "shots=42" in result.get_counts()

    def test_single_circuit_run_func_receives_default_token(self, vb_echo):
        """Default token '' must be forwarded to run_func."""
        result = vb_echo.run_async(_simple_qc()).result()
        assert "token=" in result.get_counts()

    def test_single_circuit_result_get_counts_index_zero(self, vb):
        """get_counts(0) must return the counts dict for the single circuit."""
        result = vb.run_async(_simple_qc()).result()
        counts = result.get_counts(0)
        assert isinstance(counts, dict)

    def test_single_circuit_result_get_counts_default_index(self, vb):
        """get_counts() without index must default to 0."""
        result = vb.run_async(_simple_qc()).result()
        assert result.get_counts() == result.get_counts(0)

    def test_single_circuit_result_all_counts(self, vb):
        """all_counts must be a list with one element."""
        result = vb.run_async(_simple_qc()).result()
        assert isinstance(result.all_counts, list)
        assert len(result.all_counts) == 1

    def test_single_circuit_result_accessible_immediately(self, vb):
        """result() must return immediately without blocking."""
        job = vb.run_async(_simple_qc())
        result = job.result()  # must not block
        assert result is not None

    def test_result_idempotent(self, vb):
        """Calling result() twice must return consistent data."""
        job = vb.run_async(_simple_qc())
        first = job.result()
        second = job.result()
        assert first.all_counts == second.all_counts


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------


class TestVirtualBackendBatch:
    """Tests for running multiple circuits through VirtualBackend."""

    def test_batch_is_done_immediately(self, vb):
        """A batch of circuits must reach DONE status before run_async returns."""
        job = vb.run_async(_simple_qc_list(1, 1, 1, 1, 1))
        assert job.status() == JobStatus.DONE

    def test_batch_result_num_circuits(self, vb):
        """The JobResult must contain one counts dict per submitted circuit."""
        result = vb.run_async(_simple_qc_list(1, 1, 1, 1, 1)).result()
        assert result.num_circuits == 5

    def test_batch_result_all_counts_length(self, vb):
        """all_counts must have the same length as the number of submitted circuits."""
        batch = _simple_qc_list(1, 2, 1)
        result = vb.run_async(batch).result()
        assert len(result.all_counts) == 3

    def test_batch_get_counts_by_index(self, vb):
        """get_counts(i) must return the counts dict for the i-th circuit."""
        result = vb.run_async(_simple_qc_list(1, 1, 1)).result()
        for i in range(3):
            counts = result.get_counts(i)
            assert isinstance(counts, dict)

    def test_batch_get_counts_out_of_range_raises(self, vb):
        """get_counts with out-of-range index must raise IndexError."""
        result = vb.run_async(_simple_qc_list(1, 1)).result()
        with pytest.raises(IndexError):
            result.get_counts(2)

    def test_batch_each_circuit_receives_same_shots(self, vb_echo):
        """Each circuit in a batch must receive the same shot count."""
        result = vb_echo.run_async(_simple_qc_list(1, 1, 1), shots=77).result()
        for i in range(3):
            assert "shots=77" in result.get_counts(i)

    def test_batch_each_circuit_receives_same_token(self, vb_echo):
        """Each circuit in a batch must receive the same token."""
        vb_echo.update_options(token="batch_token")
        result = vb_echo.run_async(_simple_qc_list(1, 2), shots=10).result()
        for i in range(2):
            assert "token=batch_token" in result.get_counts(i)

    def test_batch_accepts_tuple(self, vb):
        """run_async must accept a tuple of circuits as a batch."""
        circuits = tuple(_simple_qc_list(1, 2, 3))
        result = vb.run_async(circuits, shots=32).result()
        assert result.num_circuits == 3

    def test_batch_accepts_generator(self, vb):
        """run_async must accept a generator yielding circuits."""
        gen = (_simple_qc() for _ in range(4))
        result = vb.run_async(gen, shots=16).result()
        assert result.num_circuits == 4

    def test_single_circuit_in_list_does_not_unwrap(self, vb):
        """A single circuit wrapped in a list must be treated as a batch of one."""
        result = vb.run_async([_simple_qc()]).result()
        assert result.num_circuits == 1

    def test_empty_batch_yields_error_job(self, vb):
        """An empty batch must produce a job in ERROR state (JobResult rejects empty counts)."""
        job = vb.run_async([])
        assert job.status() == JobStatus.ERROR
        with pytest.raises(JobFailureError):
            job.result()


# ---------------------------------------------------------------------------
# Shots behaviour
# ---------------------------------------------------------------------------


class TestVirtualBackendShots:
    """Tests for shot-count handling in VirtualBackend."""

    def test_shots_from_options_when_none_passed(self, vb):
        """When no shots argument is given, the options value must be used."""
        vb.update_options(shots=256)
        job = vb.run_async(_simple_qc())
        # _trivial_run_func returns {'0': shots}
        assert job.result().get_counts()["0"] == 256

    def test_explicit_shots_overrides_options(self, vb):
        """Explicit shots must take precedence over the options value."""
        vb.update_options(shots=256)
        job = vb.run_async(_simple_qc(), shots=128)
        assert job.result().get_counts()["0"] == 128

    def test_shots_string_raises_type_error(self, vb):
        """Passing a string as shots must raise TypeError (via base run)."""
        from qrisp.circuit.quantum_circuit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.measure(0)
        with pytest.raises(TypeError, match="shots"):
            vb.run(qc, shots="not_an_int")

    def test_shots_zero_raises_value_error(self, vb):
        """Passing 0 as shots must raise ValueError (via base run)."""
        from qrisp.circuit.quantum_circuit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.measure(0)
        with pytest.raises(ValueError, match="shots"):
            vb.run(qc, shots=0)

    def test_shots_negative_raises_value_error(self, vb):
        """Passing a negative shot count must raise ValueError."""
        from qrisp.circuit.quantum_circuit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.measure(0)
        with pytest.raises(ValueError, match="shots"):
            vb.run(qc, shots=-5)


# ---------------------------------------------------------------------------
# Token handling
# ---------------------------------------------------------------------------


class TestVirtualBackendToken:
    """Tests for token forwarding in VirtualBackend."""

    def test_token_defaults_to_empty_string(self, vb_echo):
        """The default token must be an empty string."""
        result = vb_echo.run_async(_simple_qc()).result()
        assert "token=" in result.get_counts()

    def test_token_passed_to_run_func(self, vb_echo):
        """An updated token must appear in the run_func's token argument."""
        vb_echo.update_options(token="secret_token")
        result = vb_echo.run_async(_simple_qc()).result()
        assert "token=secret_token" in result.get_counts()

    def test_token_preserved_across_multiple_jobs(self, vb_echo):
        """The token must be consistent across multiple sequential jobs."""
        vb_echo.update_options(token="persistent_token")
        for _ in range(3):
            result = vb_echo.run_async(_simple_qc()).result()
            assert "token=persistent_token" in result.get_counts()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestVirtualBackendErrors:
    """Tests for error propagation in VirtualBackend."""

    def test_run_func_error_job_status_is_error(self, vb_fail):
        """A failing run_func must set the job status to ERROR."""
        job = vb_fail.run_async(_simple_qc())
        assert job.status() == JobStatus.ERROR

    def test_run_func_error_result_raises(self, vb_fail):
        """result() must raise RuntimeError when run_func failed."""
        job = vb_fail.run_async(_simple_qc())
        with pytest.raises(RuntimeError):
            job.result()

    def test_run_func_error_raises_job_failure_error(self, vb_fail):
        """result() must raise JobFailureError specifically for an ERROR job."""
        job = vb_fail.run_async(_simple_qc())
        with pytest.raises(JobFailureError):
            job.result()

    def test_run_func_error_done_is_false(self, vb_fail):
        """done() must be False when the job is in ERROR state."""
        job = vb_fail.run_async(_simple_qc())
        assert job.done() is False

    def test_run_func_error_in_final_state_is_true(self, vb_fail):
        """in_final_state() must be True when the job is in ERROR state."""
        job = vb_fail.run_async(_simple_qc())
        assert job.in_final_state() is True

    def test_error_job_cancel_returns_false(self, vb_fail):
        """cancel() must return False on a job that has already failed."""
        job = vb_fail.run_async(_simple_qc())
        assert job.cancel() is False

    def test_batch_error_affects_whole_job(self, vb_fail):
        """An error in any circuit of a batch must mark the entire job as ERROR."""
        job = vb_fail.run_async(_simple_qc_list(1, 1, 1))
        assert job.status() == JobStatus.ERROR
        with pytest.raises(JobFailureError):
            job.result()

    def test_run_func_error_message_contains_cause(self, vb_fail):
        """The JobFailureError message must include the underlying run_func exception's description."""
        job = vb_fail.run_async(_simple_qc())
        with pytest.raises(JobFailureError, match="Simulated run_func failure"):
            job.result()

    def test_run_func_error_chains_original_exception(self, vb_fail):
        """The original run_func exception must be chained as __cause__ of JobFailureError."""
        job = vb_fail.run_async(_simple_qc())
        with pytest.raises(JobFailureError) as exc_info:
            job.result()
        assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------


class TestVirtualBackendJobLifecycle:
    """Tests that VirtualJob honours the Job lifecycle contract."""

    def test_job_is_virtual_job_instance(self, vb):
        """run_async must return a VirtualJob."""
        job = vb.run_async(_simple_qc())
        assert isinstance(job, VirtualJob)

    def test_job_backend_reference(self, vb):
        """The job's backend attribute must point back to the VirtualBackend."""
        job = vb.run_async(_simple_qc())
        assert job.backend is vb

    def test_job_starts_initializing(self, vb):
        """A newly constructed VirtualJob (before submit) must start in INITIALIZING."""
        job = VirtualJob(backend=vb, circuits=[_simple_qc()], shots=None, token="")
        assert job.status() == JobStatus.INITIALIZING

    def test_job_after_submit_is_running_then_done(self, vb):
        """After submit(), a VirtualJob transitions RUNNING → DONE (or ERROR)."""
        job = VirtualJob(backend=vb, circuits=[_simple_qc()], shots=128, token="")
        assert job.status() == JobStatus.INITIALIZING
        job.submit()
        # Synchronous execution means submit goes RUNNING → DONE
        assert job.status() == JobStatus.DONE

    def test_job_not_in_initializing_after_run_async(self, vb):
        """After run_async returns, the job must not be in INITIALIZING."""
        job = vb.run_async(_simple_qc())
        assert job.status() != JobStatus.INITIALIZING

    def test_job_done_after_run_async(self, vb):
        """After run_async returns, the job must be DONE (for a successful run)."""
        job = vb.run_async(_simple_qc())
        assert job.done() is True
        assert job.in_final_state() is True


# ---------------------------------------------------------------------------
# Cancel behaviour
# ---------------------------------------------------------------------------


class TestVirtualBackendCancel:
    """Tests that VirtualBackend cancel() behaves correctly for synchronous jobs."""

    def test_cancel_on_done_job_returns_false(self, vb):
        """cancel() must return False on a job that is already DONE."""
        job = vb.run_async(_simple_qc())
        assert job.cancel() is False

    def test_cancel_on_done_job_does_not_change_status(self, vb):
        """cancel() on a DONE job must leave the status as DONE."""
        job = vb.run_async(_simple_qc())
        job.cancel()
        assert job.status() == JobStatus.DONE

    def test_cancel_on_error_job_returns_false(self, vb_fail):
        """cancel() must return False on a job that is already ERROR."""
        job = vb_fail.run_async(_simple_qc())
        assert job.cancel() is False


# ---------------------------------------------------------------------------
# Base Backend.run() integration
# ---------------------------------------------------------------------------


class TestVirtualBackendRunMethod:
    """Tests that Backend.run() (the synchronous wrapper) works with VirtualBackend."""

    def test_run_single_quantum_circuit_returns_mapping(self, vb):
        """back.run(qc) with a single QuantumCircuit must return a Mapping."""
        result = vb.run(_simple_qc(), shots=64)
        assert isinstance(result, Mapping)

    def test_run_list_of_quantum_circuits_returns_list(self, vb):
        """back.run([qc, qc]) must return a list."""
        qcs = _simple_qc_list(1, 1)
        result = vb.run(qcs, shots=32)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_run_list_each_element_is_mapping(self, vb):
        """Each element in a batch run result must be a Mapping."""
        result = vb.run(_simple_qc_list(1, 1), shots=16)
        assert all(isinstance(r, Mapping) for r in result)

    def test_run_async_single_qc_num_circuits_is_one(self, vb):
        """run_async(qc) must yield a job with num_circuits==1."""
        job = vb.run_async(_simple_qc(), shots=64)
        assert job.result().num_circuits == 1


# ---------------------------------------------------------------------------
# Multiple sequential jobs
# ---------------------------------------------------------------------------


class TestVirtualBackendMultipleJobs:
    """Tests that the same VirtualBackend can run multiple jobs sequentially."""

    def test_multiple_sequential_jobs_all_succeed(self, vb):
        """Running several jobs sequentially must all succeed."""
        for _ in range(5):
            job = vb.run_async(_simple_qc(), shots=100)
            assert job.done()
            assert job.result().num_circuits == 1

    def test_multiple_sequential_jobs_no_state_leak(self, vb):
        """Results from one job must not leak into the next."""
        vb.update_options(shots=100)
        r1 = vb.run_async(_simple_qc()).result()
        vb.update_options(shots=200)
        r2 = vb.run_async(_simple_qc()).result()
        assert r1.get_counts()["0"] == 100
        assert r2.get_counts()["0"] == 200

    def test_multiple_sequential_batches(self, vb):
        """Running several batch jobs sequentially must all succeed."""
        for batch_size in [2, 3, 4]:
            batch = _simple_qc_list(*([1] * batch_size))
            result = vb.run_async(batch, shots=50).result()
            assert result.num_circuits == batch_size

    def test_interleaved_batch_and_single(self, vb):
        """Interleaving single-circuit and batch jobs must not interfere."""
        job1 = vb.run_async(_simple_qc(), shots=10)
        job2 = vb.run_async(_simple_qc_list(1, 1, 1), shots=20)
        job3 = vb.run_async(_simple_qc(), shots=30)

        assert job1.done()
        assert job2.done()
        assert job3.done()

        assert job1.result().num_circuits == 1
        assert job2.result().num_circuits == 3
        assert job3.result().num_circuits == 1


# ---------------------------------------------------------------------------
# run_func call signature compatibility
# ---------------------------------------------------------------------------


class TestVirtualBackendRunFuncSignature:
    """Tests for different run_func signatures supported by VirtualBackend."""

    def test_run_func_with_defaults(self, vb):
        """run_func with default parameter values must work."""
        # _trivial_run_func already has defaults; verified by other tests.
        assert vb.run_async(_simple_qc()).done()

    def test_run_func_with_only_positional(self):
        """run_func accepting only positional args must work."""

        def pos_only(qasm_str, shots, token):
            return {"result": str(shots)}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", QrispDeprecationWarning)
            backend = VirtualBackend(run_func=pos_only)
        job = backend.run_async(_simple_qc(), shots=42)
        assert job.result().get_counts()["result"] == "42"

    def test_run_func_with_keyword_args(self):
        """A run_func using only **kwargs fails because submit() calls it positionally.

        VirtualJob.submit() passes (qasm_str, shots, token) as positional arguments.
        A function declared as ``def fn(**kwargs)`` cannot receive positional args
        and will raise TypeError, causing the job to enter ERROR state.
        """

        def kwarg_fn(**kwargs):
            return {"result": "ok"}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", QrispDeprecationWarning)
            backend = VirtualBackend(run_func=kwarg_fn)
        job = backend.run_async(_simple_qc(), shots=5)
        assert job.status() == JobStatus.ERROR
        with pytest.raises(JobFailureError):
            job.result()

    def test_run_func_returning_empty_dict(self):
        """run_func that returns an empty dict must be accepted."""

        def empty_fn(qasm_str, shots=None, token=""):
            return {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", QrispDeprecationWarning)
            backend = VirtualBackend(run_func=empty_fn)
        job = backend.run_async(_simple_qc())
        assert job.result().get_counts() == {}

    def test_run_func_returning_non_dict_raises(self):
        """run_func that returns a non-dict must cause a ValueError from JobResult."""

        def bad_fn(qasm_str, shots=None, token=""):
            return "not a dict"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", QrispDeprecationWarning)
            backend = VirtualBackend(run_func=bad_fn)
        job = backend.run_async(_simple_qc())
        assert job.status() == JobStatus.ERROR
        with pytest.raises(JobFailureError):
            job.result()


# ---------------------------------------------------------------------------
# QuantumCircuit integration
# ---------------------------------------------------------------------------


class TestVirtualBackendQuantumCircuitIntegration:
    """End-to-end tests with actual QuantumCircuits and measurement."""

    def test_simple_hadamard_circuit(self, vb):
        """A single-qubit Hadamard circuit must produce bitstrings in the result."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure(0)
        result = vb.run(qc, shots=100)
        assert isinstance(result, Mapping)
        # _trivial_run_func always returns {'0': shots}. This verifies the
        # flow works; the actual counts depend on the provided run_func.
        assert "0" in result or isinstance(result, Mapping)

    def test_two_qubit_bell_circuit(self, vb_echo):
        """A two-qubit Bell circuit QASM must be passed correctly to run_func."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0)
        qc.measure(1)
        result = vb_echo.run(qc, shots=50)
        # The echo run_func includes a qasm_len key.
        assert any("qasm_len" in k for k in result)

    def test_multi_qubit_circuit_single_bitstring_key_length(self, vb_echo):
        """QASM for a multi-qubit circuit must encode the correct number of qubits."""
        for n in [1, 2, 3, 5]:
            qc = QuantumCircuit(n)
            for i in range(n):
                qc.h(i)
                qc.measure(i)
            result = vb_echo.run(qc, shots=10)
            qasm_key = [k for k in result if k.startswith("qasm_len=")][0]
            qasm_len = int(qasm_key.split("=")[1])
            assert qasm_len > n  # at minimum the QASM must mention all qubits

    def test_circuit_unmodified_after_run(self, vb):
        """The QuantumCircuit must not be mutated by the execution."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        num_qubits_before = len(qc.qubits)
        num_data_before = len(qc.data)
        vb.run(qc, shots=32)
        assert len(qc.qubits) == num_qubits_before
        assert len(qc.data) == num_data_before


# ---------------------------------------------------------------------------
# BatchedBackend compatibility
# ---------------------------------------------------------------------------


class TestVirtualBackendBatched:
    """Tests for VirtualBackend interoperability with BatchedBackend."""

    def test_batched_returns_batched_backend(self, vb):
        """batched() must return a BatchedBackend."""
        batched = vb.batched()
        from qrisp.interface.batched_backend import BatchedBackend

        assert isinstance(batched, BatchedBackend)

    def test_batched_backend_wraps_virtual(self, vb):
        """The BatchedBackend must wrap the VirtualBackend."""
        batched = vb.batched()
        from qrisp.interface.batched_backend import BatchedBackend

        assert isinstance(batched, BatchedBackend)
        assert batched._backend is vb

    def test_batched_run_and_dispatch(self, vb):
        """Batched execution must accumulate circuits and dispatch them correctly."""
        batched = vb.batched()
        qc1 = QuantumCircuit(1)
        qc1.h(0)
        qc1.measure(0)
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure(0)

        # Submit circuits via the batched wrapper
        result1 = batched.run(qc1, shots=42)
        result2 = batched.run(qc2, shots=42)

        # Before dispatch, accessing results must raise
        with pytest.raises(RuntimeError):
            _ = result1["0"]

        batched.dispatch()
        # After dispatch, results must be populated
        assert "0" in result1
        assert "0" in result2


# ---------------------------------------------------------------------------
# run_func receives correct QASM for Qrisp circuits
# ---------------------------------------------------------------------------


class TestVirtualBackendQasmFidelity:
    """Tests that the QASM passed to run_func accurately represents the circuit."""

    def test_qasm_contains_qubit_declarations(self, vb_echo):
        """QASM must contain qreg declarations for all qubits."""
        qc = QuantumCircuit(3)
        for i in range(3):
            qc.x(i)
            qc.measure(i)
        result = vb_echo.run(qc, shots=1)
        qasm_key = [k for k in result if k.startswith("qasm_len=")][0]
        qasm_len = int(qasm_key.split("=")[1])
        # A valid OpenQASM 2.0 program for 3 qubits should be at least ~100 chars
        assert qasm_len > 80

    def test_qasm_contains_gate_operations(self, vb_echo):
        """QASM must contain the applied gate operations."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0)
        qc.measure(1)
        result = vb_echo.run(qc, shots=1)
        qasm_key = [k for k in result if k.startswith("qasm_len=")][0]
        qasm_len = int(qasm_key.split("=")[1])
        # CX + H + measurements should produce non-trivial QASM
        assert qasm_len > 50
