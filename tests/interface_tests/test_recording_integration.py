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

"""Tests for recording and replaying real backends through Qrisp's high-level entry points."""

import numpy as np
import pytest

from qrisp import QuantumFloat, h, measure
from qrisp.default_backend import QrispSimulatorBackend
from qrisp.interface import BackendRecorder, QiskitBackend, ReplayBackend, StimBackend
from qrisp.jasp import backend_sampler, sample


@pytest.fixture(params=["qrisp_simulator", "stim", "qiskit_aer"])
def backend(request):
    """A fresh instance of each offline backend; the Qiskit one runs on AerSimulator, if installed."""
    if request.param == "qrisp_simulator":
        return QrispSimulatorBackend()
    if request.param == "stim":
        return StimBackend()
    pytest.importorskip("qiskit_aer")
    return QiskitBackend()


def _plus_state():
    """Return a 3-qubit QuantumFloat in the state |+++⟩."""
    qf = QuantumFloat(3)
    h(qf)
    return qf


def _kernel():
    """Measure a 3-qubit QuantumFloat in the state |+++⟩."""
    return measure(_plus_state())


class TestBackendIntegration:
    """Test recording and replaying each offline backend through Qrisp's high-level entry points."""

    def test_get_measurement_replay(self, backend):
        """get_measurement on the recording returns the recorded result."""
        with BackendRecorder(backend) as recorder:
            live = _plus_state().get_measurement(backend=backend, shots=500)
        rec = recorder.recording

        # A fresh QuantumFloat compiles to a circuit with different names, which
        # strict mode must still accept.
        assert _plus_state().get_measurement(backend=rec, shots=500) == live

    def test_batched_backend_replay(self, backend):
        """A BatchedBackend dispatch replays through the recording with the same results."""
        with BackendRecorder(backend) as recorder:
            bb = backend.batched()
            live = [_plus_state().get_measurement(backend=bb, shots=s) for s in (100, 200)]
            bb.dispatch()
            live = [dict(r) for r in live]

        bb = recorder.recording.batched()
        replayed = [_plus_state().get_measurement(backend=bb, shots=s) for s in (100, 200)]
        bb.dispatch()
        assert [dict(r) for r in replayed] == live

    def test_backend_sampler_replay(self, backend):
        """Per-shot order is shuffled by backend_sampler (spec caveat), so compare sorted."""

        def run(backend):
            @backend_sampler(backend=backend)
            def main():
                return sample(_kernel, shots=300)()

            return np.sort(np.asarray(main()))

        with BackendRecorder(backend) as recorder:
            live = run(backend)
        np.testing.assert_array_equal(run(recorder.recording), live)

    def test_seeded_backend_sampler_replays_shot_by_shot(self, backend):
        """With the same seed, the live run and every replay agree shot by shot, without sorting."""

        def run(backend):
            @backend_sampler(backend=backend, seed=2026)
            def main():
                return sample(_kernel, shots=300)()

            return np.asarray(main())

        with BackendRecorder(backend) as recorder:
            live = run(backend)
        rec = recorder.recording

        np.testing.assert_array_equal(run(rec), live)
        rec.reset()
        np.testing.assert_array_equal(run(rec), live)

    @pytest.mark.parametrize("fname", ["rec.json", "rec.json.gz"])
    def test_file_round_trip(self, backend, tmp_path, fname):
        """A recording saved to a file and loaded again replays the backend's real results."""
        with BackendRecorder(backend) as recorder:
            live = _plus_state().get_measurement(backend=backend, shots=500)
        path = tmp_path / fname
        recorder.recording.to_file(str(path))

        loaded = ReplayBackend.from_file(str(path))
        assert loaded.name == recorder.recording.name
        assert _plus_state().get_measurement(backend=loaded, shots=500) == live
