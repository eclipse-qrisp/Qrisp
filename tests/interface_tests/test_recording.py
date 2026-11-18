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

"""Tests for BackendRecorder and ReplayBackend, run against the ScriptedBackend defined below.

- ``TestBackendRecorder``: patching and restoring ``run_async``, reuse and nesting of recorders.
- ``TestRecorderOnError``: cleanup when the experiment, a submission or the final wait fails.
- ``TestReplay``: recorded results come back in order, for single circuits, batches and shot lists.
- ``TestCursor``: ``len``, ``remaining``, exhaustion and ``reset``.
- ``TestMismatch``: a call that differs from its record raises ``RecordingMismatchError``.
- ``TestFailedJobs``: ERROR and CANCELLED jobs replay as the same failure, in position.
- ``TestHardwareSnapshot``: hardware data is copied when recording starts.
- ``TestFileRoundTrip``: ``to_file``/``from_file``, the file layout and what is left out.

Real backends are tested in ``test_recording_integration.py``, the fingerprint in
``test_circuit_fingerprint.py``.
"""

import gzip
import json
import threading

import pytest
from conftest import MinimalJob

from qrisp import QuantumCircuit
from qrisp.interface import BackendRecorder, ReplayBackend
from qrisp.interface.backend import Backend
from qrisp.interface.job import JobCancelledError, JobFailureError, JobResult, JobStatus
from qrisp.interface.recording import RecordingExhaustedError, RecordingMismatchError


class ScriptedBackend(Backend):
    """A third-party-style backend with no knowledge of recording.

    Every call returns counts that are unique to that call, so a replay that
    re-simulated instead of returning the recording would be caught. Keys are
    ``int``, which exercises the non-string-key path of the file format.

    ``script`` lists the outcome of each successive ``run_async`` call:
    ``"ok"`` (default once exhausted), ``"error"``, ``"cancel"`` or
    ``"pending"`` (left QUEUED; resolve it later via ``pending_jobs``).
    """

    def __init__(self, script=(), **hardware):
        """Initialise with an outcome script and optional hardware overrides."""
        super().__init__(name="Scripted")
        self.script = list(script)
        self.calls = 0
        self.pending_jobs = []
        self.hardware = {
            "num_qubits": 5,
            "max_circuits": 10,
            "connectivity": [[0, 1], [1, 2]],
            "gate_set": ["cz", "prx", "measure"],
            "error_rates": {"cz": 0.01},
            "info": {"firmware": "1.0"},
        }
        self.hardware.update(hardware)

    @classmethod
    def _default_options(cls):
        return {"shots": 100}

    @property
    def num_qubits(self):
        """Return the scripted ``num_qubits``."""
        return self.hardware["num_qubits"]

    @property
    def max_circuits(self):
        """Return the scripted ``max_circuits``."""
        return self.hardware["max_circuits"]

    @property
    def connectivity(self):
        """Return the scripted ``connectivity``."""
        return self.hardware["connectivity"]

    @property
    def gate_set(self):
        """Return the scripted ``gate_set``."""
        return self.hardware["gate_set"]

    @property
    def error_rates(self):
        """Return the scripted ``error_rates``."""
        return self.hardware["error_rates"]

    @property
    def info(self):
        """Return the scripted ``info``."""
        return self.hardware["info"]

    @property
    def health(self):
        """Return a live-looking health status, which a recording must not copy."""
        return "healthy"

    @property
    def queue(self):
        """Return a live-looking queue length, which a recording must not copy."""
        return 3

    def run_async(self, circuits, shots=None):
        """Run the next scripted outcome; see the class docstring."""
        circuits = [circuits] if isinstance(circuits, QuantumCircuit) else list(circuits)
        if shots is None:
            shots = self.options["shots"]
        per_circuit = shots if isinstance(shots, list) else [shots] * len(circuits)

        call = self.calls
        self.calls += 1
        outcome = self.script[call] if call < len(self.script) else "ok"

        job = MinimalJob(backend=self, job_id=f"job-{call}")
        job.submit()
        result = JobResult(
            [
                {str(2 * (10 * call + i) + 1): 1, str(2 * (10 * call + i)): circuit_shots}
                for i, circuit_shots in enumerate(per_circuit)
            ],
            job_id=f"job-{call}",
        )
        if outcome == "ok":
            job._resolve(result)
        elif outcome == "error":
            job._fail(RuntimeError("device fault"))
        elif outcome == "cancel":
            job.cancel()
        elif outcome == "pending":
            self.pending_jobs.append((job, result))
        return job


def _circuit(num_clbits=1, gate="x", angle=0.5):
    """Small measured circuit with ``num_clbits`` clbits."""
    qc = QuantumCircuit(1, num_clbits)
    if gate == "x":
        qc.x(0)
    elif gate == "rx":
        qc.rx(angle, 0)
    qc.measure(0, 0)
    return qc


def _counts(job):
    """Return the counts of ``job``, raising if it failed."""
    return job.result().all_counts


def _record(backend, calls):
    """Record ``calls`` (a list of (circuits, shots)) and return (recording, live counts)."""
    with BackendRecorder(backend) as recorder:
        live_counts = [_counts(backend.run_async(circuits, shots)) for circuits, shots in calls]
    return recorder.recording, live_counts


class TestBackendRecorder:
    """Test :class:`BackendRecorder`."""

    def test_backend_has_no_recording_members(self):
        for name in ("start_recording", "stop_recording", "is_recording"):
            assert not hasattr(Backend, name), name

    def test_recording_before_exit_raises(self):
        backend = ScriptedBackend()
        recorder = BackendRecorder(backend)
        with pytest.raises(RuntimeError):
            recorder.recording
        with recorder:
            with pytest.raises(RuntimeError):
                recorder.recording

    def test_recording_is_replay_backend(self):
        backend = ScriptedBackend()
        with BackendRecorder(backend) as recorder:
            backend.run_async(_circuit())
        recording = recorder.recording
        assert isinstance(recording, ReplayBackend)
        assert isinstance(recording, Backend)
        assert len(recording) == 1

    def test_exit_restores_class_run_async(self):
        backend = ScriptedBackend()
        with BackendRecorder(backend):
            assert "run_async" in vars(backend)
        assert "run_async" not in vars(backend)

    def test_exit_restores_existing_instance_run_async(self):
        backend = ScriptedBackend()
        mock_calls = []

        def mock_run_async(circuits, shots=None):
            mock_calls.append(shots)
            return ScriptedBackend.run_async(backend, circuits, shots)

        backend.run_async = mock_run_async
        with BackendRecorder(backend) as recorder:
            backend.run_async(_circuit(), 10)

        assert backend.run_async is mock_run_async
        assert mock_calls == [10]
        assert len(recorder.recording) == 1

    def test_calls_outside_recording_are_not_recorded(self):
        backend = ScriptedBackend()
        backend.run_async(_circuit())
        with BackendRecorder(backend) as recorder:
            backend.run_async(_circuit())
        backend.run_async(_circuit())
        assert len(recorder.recording) == 1

    def test_recorder_can_be_reused(self):
        backend = ScriptedBackend()
        recorder = BackendRecorder(backend)
        with recorder:
            backend.run_async(_circuit())
        first = recorder.recording
        with recorder:
            backend.run_async(_circuit())
            backend.run_async(_circuit())
        assert (len(first), len(recorder.recording)) == (1, 2)

    def test_nested_recorders(self):
        backend = ScriptedBackend()
        with BackendRecorder(backend) as outer:
            backend.run_async(_circuit())
            with BackendRecorder(backend) as inner:
                backend.run_async(_circuit())
            backend.run_async(_circuit())
        assert (len(outer.recording), len(inner.recording)) == (3, 1)
        assert "run_async" not in vars(backend)

    def test_recording_returns_original_job(self):
        backend = ScriptedBackend()
        with BackendRecorder(backend):
            job = backend.run_async(_circuit())
        assert isinstance(job, MinimalJob)
        assert job.job_id == "job-0"

    def test_run_is_recorded(self):
        backend = ScriptedBackend()
        with BackendRecorder(backend) as recorder:
            backend.run(_circuit())
            backend.run([_circuit(), _circuit()])
        assert len(recorder.recording) == 2

    def test_vendor_keyword_arguments_are_forwarded_and_replayable(self):
        class TimeslotBackend(ScriptedBackend):
            def run_async(self, circuits, shots=None, *, use_timeslot=False):
                """Record the use_timeslot value, like IQMBackend's keyword-only argument."""
                self.last_use_timeslot = use_timeslot
                return super().run_async(circuits, shots)

        backend = TimeslotBackend()
        with BackendRecorder(backend) as recorder:
            live = _counts(backend.run_async(_circuit(), 10, use_timeslot=True))
        recording = recorder.recording

        assert backend.last_use_timeslot is True
        assert _counts(recording.run_async(_circuit(), 10, use_timeslot=True)) == live

    def test_recording_does_not_block_run_async(self):
        backend = ScriptedBackend(script=["pending"])
        with BackendRecorder(backend) as recorder:
            job = backend.run_async(_circuit())
            assert not job.done()

            pending_job, result = backend.pending_jobs[0]
            timer = threading.Timer(0.2, pending_job._resolve, args=(result,))
            timer.start()
        timer.join()

        assert _counts(recorder.recording.run_async(_circuit())) == result.all_counts


def _interrupted_result(timeout=None):
    """Stand-in for Job.result that behaves like Ctrl-C while waiting."""
    raise KeyboardInterrupt


class TestRecorderOnError:
    """Test that :class:`BackendRecorder` cleans up when the experiment, a submission or the final wait fails."""

    def test_experiment_raises_unpatches_and_keeps_the_recordings(self):
        backend = ScriptedBackend()
        recorder = BackendRecorder(backend)
        with pytest.raises(ValueError, match="experiment crashed"):
            with recorder:
                live = _counts(backend.run_async(_circuit(), 10))
                raise ValueError("experiment crashed")

        assert "run_async" not in vars(backend)
        recording = recorder.recording
        assert len(recording) == 1
        assert _counts(recording.run_async(_circuit(), 10)) == live

    def test_backend_can_be_recorded_again_after_a_crash(self):
        backend = ScriptedBackend()
        with pytest.raises(ValueError):
            with BackendRecorder(backend):
                backend.run_async(_circuit())
                raise ValueError

        with BackendRecorder(backend) as recorder:
            backend.run_async(_circuit())
            backend.run_async(_circuit())
        assert len(recorder.recording) == 2

    def test_experiment_raises_restores_existing_instance_run_async(self):
        backend = ScriptedBackend()

        def mock_run_async(circuits, shots=None):
            return ScriptedBackend.run_async(backend, circuits, shots)

        backend.run_async = mock_run_async
        with pytest.raises(ValueError):
            with BackendRecorder(backend):
                backend.run_async(_circuit())
                raise ValueError

        assert backend.run_async is mock_run_async

    def test_experiment_raises_still_collects_pending_jobs(self):
        backend = ScriptedBackend(script=["pending"])
        recorder = BackendRecorder(backend)
        with pytest.raises(ValueError):
            with recorder:
                backend.run_async(_circuit())
                pending_job, result = backend.pending_jobs[0]
                timer = threading.Timer(0.2, pending_job._resolve, args=(result,))
                timer.start()
                raise ValueError
        timer.join()

        assert _counts(recorder.recording.run_async(_circuit())) == result.all_counts

    def test_inner_recorder_raises_outer_keeps_recording(self):
        backend = ScriptedBackend()
        with BackendRecorder(backend) as outer:
            backend.run_async(_circuit())
            with pytest.raises(ValueError):
                with BackendRecorder(backend) as inner:
                    backend.run_async(_circuit())
                    raise ValueError
            backend.run_async(_circuit())

        assert (len(outer.recording), len(inner.recording)) == (3, 1)
        assert "run_async" not in vars(backend)

    def test_failed_submission_is_not_recorded(self):
        class RejectingBackend(ScriptedBackend):
            def run_async(self, circuits, shots=None):
                """Reject submissions with shots above 1000, like a server-side limit."""
                if shots is not None and shots > 1000:
                    raise ValueError("too many shots")
                return super().run_async(circuits, shots)

        backend = RejectingBackend()
        with BackendRecorder(backend) as recorder:
            with pytest.raises(ValueError, match="too many shots"):
                backend.run_async(_circuit(), 5000)
            backend.run_async(_circuit(), 10)

        assert len(recorder.recording) == 1
        assert "run_async" not in vars(backend)

    def test_interrupted_wait_on_exit_still_unpatches(self):
        backend = ScriptedBackend(script=["pending"])
        recorder = BackendRecorder(backend)
        with pytest.raises(KeyboardInterrupt):
            with recorder:
                job = backend.run_async(_circuit())
                job.result = _interrupted_result

        assert "run_async" not in vars(backend)
        with pytest.raises(RuntimeError):
            recorder.recording


class TestReplay:
    """Test that :class:`ReplayBackend` replays the recorded results."""

    def test_replay_single_circuit(self):
        recording, live = _record(ScriptedBackend(), [(_circuit(), 50)])
        assert _counts(recording.run_async(_circuit(), 50)) == live[0]

    def test_replay_preserves_order_across_calls(self):
        calls = [(_circuit(), 10), (_circuit(), 20), (_circuit(), 30)]
        recording, live = _record(ScriptedBackend(), calls)
        assert [_counts(recording.run_async(circuit, shots)) for circuit, shots in calls] == live

    def test_replay_batch(self):
        batch = [_circuit(), _circuit(2), _circuit(3)]
        recording, live = _record(ScriptedBackend(), [(batch, 40)])
        assert _counts(recording.run_async(batch, 40)) == live[0]

    def test_replay_per_circuit_shot_list(self):
        batch = [_circuit(), _circuit()]
        recording, live = _record(ScriptedBackend(), [(batch, [7, 9])])
        assert _counts(recording.run_async(batch, [7, 9])) == live[0]

    def test_replay_job_is_already_done(self):
        recording, _ = _record(ScriptedBackend(), [(_circuit(), 10)])
        job = recording.run_async(_circuit(), 10)
        assert job.status() == JobStatus.DONE

    def test_replay_returns_metadata(self):
        recording, _ = _record(ScriptedBackend(), [(_circuit(), 10)])
        assert recording.run_async(_circuit(), 10).result().metadata["job_id"] == "job-0"

    def test_none_shots_resolved_from_options(self):
        backend = ScriptedBackend()
        recording, live = _record(backend, [(_circuit(), None)])
        assert _counts(recording.run_async(_circuit(), 100)) == live[0]
        recording.reset()
        assert _counts(recording.run_async(_circuit(), None)) == live[0]

    def test_replay_via_run(self):
        backend = ScriptedBackend()
        with BackendRecorder(backend) as recorder:
            expected = backend.run(_circuit(), shots=10)
        assert recorder.recording.run(_circuit(), shots=10) == expected


class TestCursor:
    """Test the replay cursor: :meth:`ReplayBackend.reset`, ``remaining``, ``len`` and exhaustion."""

    def test_len_and_remaining(self):
        calls = [(_circuit(), 10), (_circuit(), 10)]
        recording, _ = _record(ScriptedBackend(), calls)
        assert (len(recording), recording.remaining) == (2, 2)
        recording.run_async(_circuit(), 10)
        assert (len(recording), recording.remaining) == (2, 1)

    def test_exhausted_raises(self):
        recording, _ = _record(ScriptedBackend(), [(_circuit(), 10)])
        recording.run_async(_circuit(), 10)
        with pytest.raises(RecordingExhaustedError):
            recording.run_async(_circuit(), 10)

    def test_reset_replays_from_start(self):
        calls = [(_circuit(), 10), (_circuit(), 20)]
        recording, live = _record(ScriptedBackend(), calls)
        for circuit, shots in calls:
            recording.run_async(circuit, shots)
        recording.reset()
        assert recording.remaining == 2
        assert [_counts(recording.run_async(circuit, shots)) for circuit, shots in calls] == live

    def test_reset_restores_options(self):
        def experiment(backend):
            first = _counts(backend.run_async(_circuit()))
            backend.update_options(shots=7)
            return first, _counts(backend.run_async(_circuit()))

        backend = ScriptedBackend()
        with BackendRecorder(backend) as recorder:
            live = experiment(backend)
        recording = recorder.recording

        assert experiment(recording) == live
        recording.reset()
        assert recording.options["shots"] == ScriptedBackend._default_options()["shots"]
        assert experiment(recording) == live

    def test_empty_recording_is_exhausted(self):
        with BackendRecorder(ScriptedBackend()) as recorder:
            pass
        recording = recorder.recording
        assert len(recording) == 0
        with pytest.raises(RecordingExhaustedError):
            recording.run_async(_circuit())


class TestMismatch:
    """Test that replay raises :class:`RecordingMismatchError` on a differing call."""

    @pytest.mark.parametrize("strict", [True, False])
    def test_mismatch_circuit_count(self, strict):
        recording, _ = _record(ScriptedBackend(), [([_circuit(), _circuit()], 10)])
        recording.strict = strict
        with pytest.raises(RecordingMismatchError):
            recording.run_async([_circuit()], 10)

    @pytest.mark.parametrize("strict", [True, False])
    def test_mismatch_num_clbits(self, strict):
        recording, _ = _record(ScriptedBackend(), [(_circuit(1), 10)])
        recording.strict = strict
        with pytest.raises(RecordingMismatchError):
            recording.run_async(_circuit(2), 10)

    @pytest.mark.parametrize("strict", [True, False])
    def test_mismatch_shots(self, strict):
        recording, _ = _record(ScriptedBackend(), [(_circuit(), 10)])
        recording.strict = strict
        with pytest.raises(RecordingMismatchError):
            recording.run_async(_circuit(), 11)

    @pytest.mark.parametrize("strict", [True, False])
    def test_mismatch_shot_list(self, strict):
        recording, _ = _record(ScriptedBackend(), [([_circuit(), _circuit()], [5, 6])])
        recording.strict = strict
        with pytest.raises(RecordingMismatchError):
            recording.run_async([_circuit(), _circuit()], [6, 5])

    def test_fingerprint_mismatch_only_in_strict_mode(self):
        recording, live = _record(ScriptedBackend(), [(_circuit(gate="x"), 10)])
        assert recording.strict is True
        with pytest.raises(RecordingMismatchError):
            recording.run_async(_circuit(gate="rx"), 10)

        recording.reset()
        recording.strict = False
        assert _counts(recording.run_async(_circuit(gate="rx"), 10)) == live[0]

    def test_mismatch_does_not_advance_cursor(self):
        recording, live = _record(ScriptedBackend(), [(_circuit(), 10)])
        with pytest.raises(RecordingMismatchError):
            recording.run_async(_circuit(), 11)
        assert recording.remaining == 1
        assert _counts(recording.run_async(_circuit(), 10)) == live[0]


class TestFailedJobs:
    """Test recording and replaying jobs that ended in ERROR or CANCELLED."""

    def test_failed_and_cancelled_jobs_replay_in_position(self):
        backend = ScriptedBackend(script=["ok", "error", "cancel", "ok"])
        live = []
        with BackendRecorder(backend) as recorder:
            for _ in range(4):
                try:
                    live.append(_counts(backend.run_async(_circuit(), 10)))
                except (JobFailureError, JobCancelledError) as error:
                    live.append(type(error))
        recording = recorder.recording
        assert live[1:3] == [JobFailureError, JobCancelledError]

        assert _counts(recording.run_async(_circuit(), 10)) == live[0]
        with pytest.raises(JobFailureError):
            recording.run_async(_circuit(), 10).result()
        with pytest.raises(JobCancelledError):
            recording.run_async(_circuit(), 10).result()
        assert _counts(recording.run_async(_circuit(), 10)) == live[3]

    def test_failed_job_never_awaited_is_still_recorded(self):
        backend = ScriptedBackend(script=["error", "ok"])
        with BackendRecorder(backend) as recorder:
            backend.run_async(_circuit(), 10)
            backend.run_async(_circuit(), 10)
        recording = recorder.recording
        assert len(recording) == 2
        assert recording.run_async(_circuit(), 10).status() == JobStatus.ERROR


class TestHardwareSnapshot:
    """Test the hardware properties that :class:`ReplayBackend` takes from the recorded backend."""

    def test_hardware_snapshot(self):
        backend = ScriptedBackend()
        with BackendRecorder(backend) as recorder:
            pass
        recording = recorder.recording
        assert recording.options["shots"] == 100
        for key in ("num_qubits", "max_circuits", "connectivity", "gate_set", "error_rates", "info"):
            assert getattr(recording, key) == backend.hardware[key], key
        assert recording.health is None
        assert recording.queue is None

    def test_hardware_snapshot_taken_at_start(self):
        backend = ScriptedBackend()
        with BackendRecorder(backend) as recorder:
            backend.hardware["error_rates"] = {"cz": 0.5}
        assert recorder.recording.error_rates == {"cz": 0.01}

    def test_hardware_snapshot_is_a_copy(self):
        backend = ScriptedBackend()
        with BackendRecorder(backend) as recorder:
            pass
        backend.hardware["error_rates"]["cz"] = 0.5
        assert recorder.recording.error_rates == {"cz": 0.01}


class TestFileRoundTrip:
    """Test :meth:`ReplayBackend.to_file` and :meth:`ReplayBackend.from_file`."""

    @pytest.mark.parametrize("fname", ["rec.json", "rec.json.gz"])
    def test_file_round_trip(self, tmp_path, fname):
        calls = [(_circuit(), 10), ([_circuit(), _circuit(2)], [3, 4])]
        backend = ScriptedBackend()
        recording, live = _record(backend, calls)

        path = tmp_path / fname
        recording.to_file(str(path))
        loaded = ReplayBackend.from_file(str(path))

        assert len(loaded) == 2
        replayed = [_counts(loaded.run_async(circuit, shots)) for circuit, shots in calls]
        assert replayed == live
        # Non-string keys and insertion order survive.
        assert [list(counts) for counts in replayed[0]] == [list(counts) for counts in live[0]]
        assert loaded.error_rates == backend.hardware["error_rates"]
        assert loaded.options["shots"] == 100

    def test_gz_file_is_gzipped(self, tmp_path):
        recording, _ = _record(ScriptedBackend(), [(_circuit(), 10)])
        path = tmp_path / "rec.json.gz"
        recording.to_file(str(path))
        with gzip.open(path, "rt") as file:
            assert json.load(file)["format_version"] == 1

    def test_file_layout(self, tmp_path):
        recording, _ = _record(ScriptedBackend(), [(_circuit(2), 10)])
        path = tmp_path / "rec.json"
        recording.to_file(str(path))
        data = json.loads(path.read_text())

        assert data["format_version"] == 1
        assert {"qrisp_version", "created", "backend", "records"} <= data.keys()
        assert data["backend"]["class"] == "ScriptedBackend"
        (record,) = data["records"]
        assert record["status"] == "DONE"
        assert record["shots"] == 10
        (circ,) = record["circuits"]
        assert circ["fingerprint"].startswith("sha256:")
        assert circ["num_clbits"] == 2
        assert record["counts"] == [{"1": 1, "0": 10}]

    def test_failed_job_survives_round_trip(self, tmp_path):
        backend = ScriptedBackend(script=["error"])
        with BackendRecorder(backend) as recorder:
            backend.run_async(_circuit(), 10)
        recording = recorder.recording

        path = tmp_path / "rec.json"
        recording.to_file(str(path))
        loaded = ReplayBackend.from_file(str(path))
        with pytest.raises(JobFailureError):
            loaded.run_async(_circuit(), 10).result()

    def test_unknown_format_version_rejected(self, tmp_path):
        recording, _ = _record(ScriptedBackend(), [(_circuit(), 10)])
        path = tmp_path / "rec.json"
        recording.to_file(str(path))
        data = json.loads(path.read_text())
        data["format_version"] = 999
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            ReplayBackend.from_file(str(path))

    @pytest.mark.parametrize("strict", [True, False])
    def test_from_file_strict_flag(self, tmp_path, strict):
        recording, _ = _record(ScriptedBackend(), [(_circuit(gate="x"), 10)])
        path = tmp_path / "rec.json"
        recording.to_file(str(path))
        loaded = ReplayBackend.from_file(str(path), strict=strict)
        assert loaded.strict is strict

    def test_non_serializable_metadata_dropped_with_warning(self, tmp_path):
        class Opaque:
            pass

        class MetaBackend(ScriptedBackend):
            def run_async(self, circuits, shots=None):
                """Return a job whose metadata holds a non-JSON-serializable value."""
                job = MinimalJob(backend=self, job_id="j")
                job.submit()
                job._resolve(JobResult([{"0": 1}], job_id="j", raw=Opaque()))
                return job

        backend = MetaBackend()
        with BackendRecorder(backend) as recorder:
            backend.run_async(_circuit(), 1)
        recording = recorder.recording

        # In memory, everything is kept.
        assert isinstance(recording.run_async(_circuit(), 1).result().metadata["raw"], Opaque)

        path = tmp_path / "rec.json"
        with pytest.warns(UserWarning):
            recording.to_file(str(path))
        meta = ReplayBackend.from_file(str(path)).run_async(_circuit(), 1).result().metadata
        assert meta == {"job_id": "j"}

    def test_to_file_saves_initial_options(self, tmp_path):
        recording, _ = _record(ScriptedBackend(), [(_circuit(), None)])
        recording.update_options(shots=7)

        path = tmp_path / "rec.json"
        recording.to_file(str(path))
        loaded = ReplayBackend.from_file(str(path))
        assert loaded.options["shots"] == ScriptedBackend._default_options()["shots"]
        loaded.run_async(_circuit())

    def test_non_serializable_hardware_saved_as_none_with_warning(self, tmp_path):
        class Graph:
            pass

        recording, _ = _record(ScriptedBackend(connectivity=Graph()), [(_circuit(), 10)])

        # The snapshot deep-copies, so check the type, not the identity.
        assert isinstance(recording.connectivity, Graph)

        path = tmp_path / "rec.json"
        with pytest.warns(UserWarning, match="connectivity"):
            recording.to_file(str(path))
        loaded = ReplayBackend.from_file(str(path))
        assert loaded.connectivity is None
        assert loaded.error_rates == {"cz": 0.01}
