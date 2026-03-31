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

"""Tests for QiskitBackend and QiskitJob."""

import numpy as np
import pytest
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2

from qrisp import QuantumCircuit, QuantumFloat
from qrisp.interface import QiskitBackend, VirtualBackend
from qrisp.interface.job import JobResult, JobStatus
from qrisp.interface.provider_backends.qiskit_backend import QiskitJob

aer_backend = AerSimulator()
fake_backend = FakeWashingtonV2()


class TestQiskitBackendConstruction:
    """Tests for QiskitBackend construction and metadata."""

    def test_name_taken_from_qiskit_backend(self):
        """Ensure the backend name matches the wrapped Qiskit backend's name."""
        backend = QiskitBackend(backend=aer_backend)
        assert backend.name == aer_backend.name

    def test_options_taken_from_qiskit_backend(self):
        """Ensure options are initialised from the wrapped Qiskit backend."""
        backend = QiskitBackend(backend=aer_backend)
        for key in aer_backend.options:
            assert backend.options[key] == aer_backend.options[key]

    def test_default_backend_is_aer_simulator(self):
        """Ensure that omitting the backend argument falls back to AerSimulator."""
        backend = QiskitBackend()
        assert backend is not None
        assert "aer" in backend.name.lower()

    def test_update_options_shots(self):
        """Ensure update_options correctly updates the shots value."""
        backend = QiskitBackend(backend=aer_backend)
        backend.update_options(shots=2048)
        assert backend.options["shots"] == 2048

    def test_update_options_invalid_key_raises(self):
        """Ensure update_options raises AttributeError for unknown keys."""
        backend = QiskitBackend(backend=aer_backend)
        with pytest.raises(AttributeError):
            backend.update_options(nonexistent_option=42)


class TestQiskitJobInterface:
    """Tests that run_async() returns a proper QiskitJob and the Job interface works.

    Note: run_async() is used here (rather than run()) because these tests need the
    Job handle. run() returns a plain dict for backward compatibility and does not
    expose the Job object.
    """

    def test_run_async_returns_qiskit_job(self):
        """Ensure run_async() returns a QiskitJob instance."""
        backend = QiskitBackend(backend=aer_backend)
        qf = QuantumFloat(4)
        qf[:] = 3
        res = qf * qf
        job = backend.run_async(res.qs.compile())
        assert isinstance(job, QiskitJob)

    def test_job_result_returns_job_result_instance(self):
        """Ensure job.result() returns a JobResult object."""
        backend = QiskitBackend(backend=aer_backend)
        qf = QuantumFloat(4)
        qf[:] = 3
        res = qf * qf
        job = backend.run_async(res.qs.compile())
        result = job.result()
        assert isinstance(result, JobResult)

    def test_job_status_is_done_after_result(self):
        """Ensure the job status is DONE after result() has been called."""
        backend = QiskitBackend(backend=aer_backend)
        qf = QuantumFloat(4)
        qf[:] = 3
        res = qf * qf
        job = backend.run_async(res.qs.compile())
        job.result()
        assert job.status() == JobStatus.DONE

    def test_job_done_and_in_final_state_after_result(self):
        """Ensure done() and in_final_state() are both True after result() returns."""
        backend = QiskitBackend(backend=aer_backend)
        qf = QuantumFloat(4)
        qf[:] = 3
        res = qf * qf
        job = backend.run_async(res.qs.compile())
        job.result()
        assert job.done() is True
        assert job.in_final_state() is True

    def test_cancel_on_finished_job_returns_false(self):
        """Ensure cancel() returns False on a job that has already completed."""
        backend = QiskitBackend(backend=aer_backend)
        qf = QuantumFloat(4)
        qf[:] = 3
        res = qf * qf
        job = backend.run_async(res.qs.compile())
        job.result()
        assert job.cancel() is False


class TestQiskitBackendExecution:
    """Integration tests for circuit execution via QiskitBackend."""

    def test_aer_simulator_exact_result(self):
        """Verify that a deterministic computation yields the exact expected result."""
        backend = QiskitBackend(backend=aer_backend)
        qf = QuantumFloat(4)
        qf[:] = 3
        res = qf * qf
        assert res.get_measurement(backend=backend) == {9: 1.0}

    def test_shots_override_via_update_options(self):
        """Verify that a custom shots value is respected and produces a correct result."""
        backend = QiskitBackend(backend=aer_backend)
        backend.update_options(shots=512)
        qf = QuantumFloat(4)
        qf[:] = 3
        res = qf * qf
        # With a deterministic computation the result is correct regardless of shot count.
        assert res.get_measurement(backend=backend) == {9: 1.0}

    def test_fake_backend_dominant_result_is_correct(self):
        """Verify that the most probable outcome is correct even under device noise."""
        backend = QiskitBackend(backend=fake_backend)
        qf = QuantumFloat(2)
        qf[:] = 2
        res = qf * qf
        meas_res = res.get_measurement(backend=backend)
        # The correct answer (4) must be the dominant outcome despite noise.
        assert meas_res[4] > 0.5

    def test_fake_backend_result_has_noise(self):
        """Verify that a fake hardware backend produces a noisy (non-sharp) distribution."""
        backend = QiskitBackend(backend=fake_backend)
        qf = QuantumFloat(2)
        qf[:] = 2
        res = qf * qf
        meas_res = res.get_measurement(backend=backend)
        # A noiseless result would have exactly one key; noise spreads mass across others.
        assert len(meas_res) > 1


# We keep this test even though VirtualBackend is deprecated.
def test_qiskit_virtual_backend():
    """Test that VirtualBackend still works via its legacy run() interface."""
    # TO-DO: prevent this test from crashing regardless of functionality.

    qc = QuantumCircuit()
    qc.add_qubit()
    qc.add_qubit()
    qc.add_clbit()

    qc.h(0)
    qc.rz(np.pi / 2, 0)
    qc.x(0)
    qc.cx(0, 1)
    qc.measure(1, 0)

    def sample_run_func(qc, shots=None, token=""):
        """Return a trivial all-zero result for any circuit."""
        if shots is None:
            shots = 10000
        return {"0": shots}

    test_virtual_backend = VirtualBackend(sample_run_func)

    assert str(test_virtual_backend.run(qc, 100)) == "{'0': 100}"
    assert test_virtual_backend.run(qc, 100)["0"] == 100
