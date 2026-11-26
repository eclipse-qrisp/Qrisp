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

"""The context manager that records the jobs a backend runs."""

from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Any

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.interface.backend import Backend
from qrisp.interface.job import Job
from qrisp.interface.recording.records import CircuitInfo, HardwareSnapshot, PendingRecord
from qrisp.interface.recording.replay import ReplayBackend


class BackendRecorder:
    """Context manager that records every ``run_async`` call a backend answers.

    Hardware runs are expensive, and each one draws new shot noise. Recording a
    run once lets variants of the experiment's classical side, such as the
    decoder, post-selection or the estimator, be compared on the exact same
    device data, without running on the device again.

    On enter, a wrapper is put over ``backend.run_async`` on the instance; on
    exit it is removed, also when the ``with`` block raises. The experiment
    keeps using the backend object itself, so calls through :meth:`Backend.run`,
    ``BatchedBackend``, vendor methods and code that captured the backend
    earlier are all recorded, and the backend's class needs no changes.

    There is one record per ``run_async`` call, in submission order. It keeps
    the fingerprint and clbit count of each circuit, the resolved shots, the
    final job status, and the counts and metadata of the result. Jobs that
    ended in ``ERROR`` or ``CANCELLED`` are recorded too. The backend's options
    and hardware data are copied when the ``with`` block starts.

    After the ``with`` block, an :attr:`recording` is created.
    :attr:`recording` is a :class:`ReplayBackend` that replays the recorded calls.

    Notes
    -----
    - Replay is only meaningful while the submitted circuits do not depend on
      the setting being changed, as with a VQE run using another optimizer.
    - Backends return counts, not individual shots. Results summed over all
      shots replay identically; results compared shot by shot only if both
      runs use ``backend_sampler(backend, seed=...)``.
    - If ``run_async`` itself raises, no job exists and nothing is recorded. An
      experiment that then retries a different call cannot be replayed.

    Examples
    --------
    >>> with BackendRecorder(backend) as recorder:
    ...     run_memory_experiment(backend)
    >>> rec = recorder.recording
    >>> rec.to_file("memory_d3.json.gz")

    Later, possibly on another machine:

    >>> rec = ReplayBackend.from_file("memory_d3.json.gz")
    >>> run_memory_experiment(rec, decoder="bposd")  # same shots as above

    """

    _slots: list[PendingRecord]
    _options_snapshot: dict[str, Any]
    _hardware_snapshot: HardwareSnapshot
    _previous_run_async: Callable[..., Job]
    _recording: ReplayBackend
    _backend: Backend

    def __init__(self, backend: Backend):
        """Prepare to record ``backend``; nothing is patched until the ``with`` block is entered."""
        self._backend = backend

    @property
    def recording(self) -> ReplayBackend:
        """The recording made by the last ``with`` block.

        Raises
        ------
        RuntimeError
            If no ``with`` block on this recorder has exited yet.

        """
        if "_recording" not in vars(self):
            raise RuntimeError("The recording is available only after the `with` block has exited.")
        return self._recording

    def __enter__(self) -> "BackendRecorder":
        """Snapshot the backend's options and hardware, then patch ``backend.run_async`` to record calls."""
        backend = self._backend
        vars(self).pop("_recording", None)
        self._slots = []
        self._options_snapshot = dict(backend.options)
        self._hardware_snapshot = HardwareSnapshot.from_backend(backend)
        # Something may already sit on the instance (a test mock, an outer recorder); __exit__ puts it back.
        if "run_async" in vars(backend):
            self._previous_run_async = vars(backend)["run_async"]
        # Picks defined instance run_async, or fall back to class's run_async.
        original_run_async = backend.run_async

        def recording_run(
            circuits: QuantumCircuit | Sequence[QuantumCircuit],
            shots: int | list[int] | None = None,
            **kwargs: Any,
        ) -> Job:
            circuit_seq = [circuits] if isinstance(circuits, QuantumCircuit) else circuits
            circuit_infos = [CircuitInfo.from_circuit(qc) for qc in circuit_seq]

            # Vendor-specific keywords (e.g. IQM's use_timeslot) are passed through unchanged.
            job = original_run_async(circuits, shots, **kwargs)

            resolved_shots = backend.options.get("shots") if shots is None else shots
            self._slots.append(PendingRecord(resolved_shots, circuit_infos, job))
            return job

        backend.run_async = recording_run
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Remove the patch, then wait for every recorded job and build :attr:`recording`."""
        backend = self._backend
        # Unpatch before waiting, so the patch is already removed if the wait is interrupted.
        if "_previous_run_async" in vars(self):
            backend.run_async = self._previous_run_async
            del self._previous_run_async
        else:
            del backend.run_async

        records = [slot.collect() for slot in self._slots]
        self._recording = ReplayBackend(
            records, self._hardware_snapshot, name=backend.name, options=self._options_snapshot
        )
