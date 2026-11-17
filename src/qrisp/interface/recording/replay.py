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

"""The backend that replays a recording, and the job it returns."""

import gzip
import importlib.metadata
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from typing_extensions import override

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.interface.backend import Backend
from qrisp.interface.job import Job, JobCancelledError, JobFailureError, JobResult, JobStatus
from qrisp.interface.recording.fingerprint import circuit_fingerprint
from qrisp.interface.recording.records import (
    FORMAT_VERSION,
    DoneRecord,
    FailedRecord,
    HardwareSnapshot,
    Record,
    _hardware_field_to_json,
    _record_from_json,
    _record_to_json,
)


class ReplayJob(Job):
    """A job replaying one :class:`Record`; it is in its final state from the start."""

    def __init__(self, backend: Backend, record: Record):
        """Wrap ``record`` as a finished job of ``backend``."""
        super().__init__(backend=backend)
        self._record = record
        self._last_known_status = record.status

    @override
    def result(self, timeout: float | None = None) -> JobResult:
        match self._record:
            case DoneRecord(counts=counts, metadata=metadata):
                return JobResult(counts, **metadata)
            case FailedRecord(status=status):
                if status == JobStatus.ERROR:
                    error_type, message = JobFailureError, "The recorded job failed (ERROR); replaying that failure."
                else:
                    error_type, message = (
                        JobCancelledError,
                        "The recorded job was cancelled; replaying that cancellation.",
                    )
                raise error_type(message)

    @override
    def cancel(self) -> bool:
        return False

    @override
    def status(self) -> JobStatus:
        return self._record.status


class RecordingMismatchError(ValueError):
    """A replayed call does not match the recorded call at the same position."""


class RecordingExhaustedError(RuntimeError):
    """A call was made after the last record was replayed."""


class ReplayBackend(Backend):
    """Backend that replays the jobs recorded by :class:`BackendRecorder`.

    Its hardware properties are those of the recorded backend at the start of the recording.

    The *i*-th ``run_async`` call returns record *i* as a finished job; a
    recorded failure raises the same error from ``result()``. Before that, the
    call is checked against the record: the number of circuits, the clbits per
    circuit and the resolved shots, and with ``strict=True`` (the default) also
    the circuit fingerprints, which catches an experiment that submits other
    circuits than were recorded. A mismatch raises
    :class:`RecordingMismatchError`, and a call after the last record raises
    :class:`RecordingExhaustedError`; call :meth:`reset` to replay again.

    :meth:`to_file` saves the recording as JSON, gzipped when the file name ends
    in ``.gz``. Pickle is avoided because recordings are shared, and loading
    someone else's pickle can execute arbitrary code.
    """

    records: list[Record]

    def __init__(
        self,
        records: Sequence[Record],
        hardware: HardwareSnapshot,
        options: Mapping | None = None,
        strict: bool = True,
        **kwargs,
    ):
        """Replay ``records`` in order; with ``strict``, also check that each circuit matches the recorded one."""
        super().__init__(name=f"ReplayBackend for {hardware.name} of type {hardware.backend_class}", options=options)
        self.records = list(records)
        self.strict = strict
        self._hardware = hardware
        self._initial_options = dict(self.options)
        self._cursor = 0

    @property
    @override
    def num_qubits(self) -> int | None:
        """Number of qubits of the recorded backend."""
        return self._hardware.num_qubits

    @property
    @override
    def max_circuits(self) -> int | None:
        """Circuit limit per job of the recorded backend."""
        return self._hardware.max_circuits

    @property
    @override
    def connectivity(self) -> object:
        """Connectivity of the recorded backend."""
        return self._hardware.connectivity

    @property
    @override
    def gate_set(self) -> object:
        """Gate set of the recorded backend."""
        return self._hardware.gate_set

    @property
    @override
    def error_rates(self) -> object:
        """Error rates of the recorded backend."""
        return self._hardware.error_rates

    @property
    @override
    def info(self) -> object:
        """Info of the recorded backend."""
        return self._hardware.info

    def __len__(self) -> int:
        """Return the total number of records."""
        return len(self.records)

    @property
    def remaining(self) -> int:
        """Number of records not replayed yet."""
        return len(self.records) - self._cursor

    def reset(self) -> None:
        """Rewind to the first record and restore the initial options, so the recording can be replayed again."""
        self._cursor = 0
        self._options = dict(self._initial_options)

    def to_file(self, fname: str) -> None:
        """Write the recording to ``fname`` as JSON, gzipped if the name ends in ``.gz``.

        Metadata values that are not JSON-serializable are left out of the file,
        with a warning; this recording in memory keeps them. Hardware properties
        that are not JSON-serializable are saved as ``None``.

        The file has one entry per recorded ``run_async`` call, in order. A failed
        job has no ``counts`` or ``metadata``::

            {
              "format_version": 1,
              "qrisp_version": "0.10.0",
              "created": "2026-11-17T09:57:11Z",
              "backend": {
                "class": "IQMBackend", "name": "IQM-emerald", "options": {"shots": 1000},
                "num_qubits": 54, "max_circuits": 200,
                "connectivity": [[0, 1], ...], "gate_set": [...], "error_rates": {...}, "info": {...}
              },
              "records": [
                {"status": "DONE", "shots": 1000,
                 "circuits": [{"fingerprint": "sha256:bbb18600...", "num_clbits": 2}],
                 "counts": [{"00": 479, "11": 521}],
                 "metadata": {"job_id": "..."}},
                {"status": "ERROR", "shots": 1000,
                 "circuits": [{"fingerprint": "sha256:5f0c2e91...", "num_clbits": 2}]}
              ]
            }

        """
        payload = {
            "format_version": FORMAT_VERSION,
            "qrisp_version": importlib.metadata.version("qrisp"),
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "backend": {
                "class": self._hardware.backend_class,
                "name": self._hardware.name,
                "options": dict(self._initial_options),
                "num_qubits": self._hardware.num_qubits,
                "max_circuits": self._hardware.max_circuits,
                "connectivity": _hardware_field_to_json("connectivity", self._hardware.connectivity),
                "gate_set": _hardware_field_to_json("gate_set", self._hardware.gate_set),
                "error_rates": _hardware_field_to_json("error_rates", self._hardware.error_rates),
                "info": _hardware_field_to_json("info", self._hardware.info),
            },
            "records": [_record_to_json(record) for record in self.records],
        }
        if fname.endswith(".gz"):
            with gzip.open(fname, "wt", encoding="utf-8") as f:
                f.write(json.dumps(payload))
        else:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(json.dumps(payload))

    @classmethod
    def from_file(cls, fname: str, strict: bool = True) -> "ReplayBackend":
        """Load a recording written by :meth:`to_file`, gunzipping it if the name ends in ``.gz``.

        Raises
        ------
        ValueError
            If the file has an unknown ``format_version`` or a record with a non-final status.

        """
        if fname.endswith(".gz"):
            with gzip.open(fname, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(fname, encoding="utf-8") as f:
                data = json.load(f)
        if data["format_version"] != FORMAT_VERSION:
            raise ValueError(f"Unsupported recording: unexpected format version {data['format_version']!r}.")
        backend = data["backend"]
        hardware = HardwareSnapshot(
            backend_class=backend["class"],
            name=backend["name"],
            num_qubits=backend["num_qubits"],
            max_circuits=backend["max_circuits"],
            connectivity=backend["connectivity"],
            gate_set=backend["gate_set"],
            error_rates=backend["error_rates"],
            info=backend["info"],
        )
        records = [_record_from_json(record) for record in data["records"]]
        return cls(records, hardware, backend["options"], strict)

    @override
    def run_async(
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        shots: int | list[int] | None = None,
        **kwargs: Any,
    ) -> Job:
        """Return the next record as an already finished job.

        Vendor-specific keyword arguments of the recorded backend (e.g. IQM's
        ``use_timeslot``) are accepted and ignored, since the results come from
        the recording.

        Raises
        ------
        RecordingExhaustedError
            If every record has been replayed; call :meth:`reset` to replay again.
        RecordingMismatchError
            If the call does not match the recorded call at this position.

        """
        if self.remaining == 0:
            raise RecordingExhaustedError(f"All {len(self)} records have been replayed; call reset() to replay again.")
        record = self.records[self._cursor]
        circuit_seq = [circuits] if isinstance(circuits, QuantumCircuit) else circuits
        resolved_shots = self.options.get("shots") if shots is None else shots
        self._check_matches(record, circuit_seq, resolved_shots)
        self._cursor += 1
        return ReplayJob(self, record)

    def _check_matches(
        self, record: Record, circuit_seq: Sequence[QuantumCircuit], shots: int | list[int] | None
    ) -> None:
        """Raise :class:`RecordingMismatchError` if the call differs from ``record``."""
        index = self._cursor
        if len(circuit_seq) != len(record.circuits):
            raise RecordingMismatchError(
                f"Record {index}: {len(record.circuits)} circuit(s) were recorded, "
                f"but {len(circuit_seq)} were submitted."
            )
        if shots != record.shots:
            raise RecordingMismatchError(f"Record {index}: recorded with shots={record.shots}, replayed with {shots}.")
        for i, (qc, info) in enumerate(zip(circuit_seq, record.circuits)):
            if len(qc.clbits) != info.num_clbits:
                raise RecordingMismatchError(
                    f"Record {index}, circuit {i}: {info.num_clbits} clbit(s) recorded, {len(qc.clbits)} submitted."
                )
            if self.strict and circuit_fingerprint(qc) != info.fingerprint:
                raise RecordingMismatchError(f"Record {index}, circuit {i}: the circuit differs from the recorded one.")
