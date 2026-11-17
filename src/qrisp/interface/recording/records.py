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

"""What a recording stores per ``run_async`` call, and its JSON form."""

import copy
import json
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Literal

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.interface.backend import Backend
from qrisp.interface.job import Job, JobCancelledError, JobFailureError, JobStatus
from qrisp.interface.recording.fingerprint import circuit_fingerprint


@dataclass(frozen=True)
class CircuitInfo:
    """What a record keeps of a submitted circuit."""

    fingerprint: str
    num_clbits: int

    @classmethod
    def from_circuit(cls, qc: QuantumCircuit) -> "CircuitInfo":
        """Fingerprint ``qc`` and count its clbits."""
        return cls(circuit_fingerprint(qc), len(qc.clbits))


@dataclass(frozen=True)
class HardwareSnapshot:
    """The backend's hardware data at the start of a recording."""

    backend_class: str
    name: str
    num_qubits: int | None
    max_circuits: int | None
    connectivity: object
    gate_set: object
    error_rates: object
    info: object

    @classmethod
    def from_backend(cls, backend: Backend) -> "HardwareSnapshot":
        """Copy ``backend``'s hardware properties, so later changes on the device don't leak in."""
        return cls(
            backend_class=type(backend).__name__,
            name=backend.name,
            num_qubits=backend.num_qubits,
            max_circuits=backend.max_circuits,
            connectivity=copy.deepcopy(backend.connectivity),
            gate_set=copy.deepcopy(backend.gate_set),
            error_rates=copy.deepcopy(backend.error_rates),
            info=copy.deepcopy(backend.info),
        )


@dataclass
class DoneRecord:
    """A recorded ``run_async`` call whose job finished successfully."""

    shots: int | list[int] | None
    circuits: list[CircuitInfo]
    counts: list[dict]
    metadata: dict[str, object]

    @property
    def status(self) -> Literal[JobStatus.DONE]:
        """Always :attr:`~JobStatus.DONE`."""
        return JobStatus.DONE


@dataclass
class FailedRecord:
    """A recorded ``run_async`` call whose job ended in ERROR or CANCELLED."""

    shots: int | list[int] | None
    circuits: list[CircuitInfo]
    status: Literal[JobStatus.ERROR, JobStatus.CANCELLED]


Record = DoneRecord | FailedRecord

FORMAT_VERSION = 1


def _record_to_json(record: Record) -> dict[str, object]:
    """Return the file representation of one record."""
    data: dict[str, object] = {
        "status": record.status.name,
        "shots": record.shots,
        "circuits": [asdict(info) for info in record.circuits],
    }
    if isinstance(record, DoneRecord):
        data["counts"] = [dict(counts) for counts in record.counts]
        data["metadata"] = _json_safe_metadata(record.metadata)
    return data


def _record_from_json(data: dict[str, Any]) -> Record:
    """Rebuild one record from its file representation."""
    circuits = [CircuitInfo(**info) for info in data["circuits"]]
    if data["status"] not in JobStatus.__members__:
        raise ValueError(f"Unknown job status {data['status']!r} in the recording.")
    status = JobStatus[data["status"]]
    match status:
        case JobStatus.DONE:
            return DoneRecord(
                shots=data["shots"],
                circuits=circuits,
                counts=data["counts"],
                metadata=data["metadata"],
            )
        case JobStatus.ERROR | JobStatus.CANCELLED:
            return FailedRecord(shots=data["shots"], circuits=circuits, status=status)
    raise ValueError(f"A recorded job cannot have the non-final status {status.name!r}.")


def _json_safe_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Return the JSON-serializable entries of ``metadata``, warning about the others."""
    safe: dict[str, object] = {}
    for key, value in metadata.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            # QUESTION: How should stacklevel be chosen? 4 points at the caller of to_file only for the
            # current call chain, and only on Python >= 3.12: on 3.11 the list comprehension in to_file
            # adds its own frame, so 4 points at to_file instead.
            warnings.warn(f"Metadata entry {key!r} is not JSON-serializable and is left out of the file.", stacklevel=4)
            continue
        safe[key] = value
    return safe


def _hardware_field_to_json(field: str, value: object) -> object:
    """Return ``value`` for the file, or ``None`` with a warning if it is not JSON-serializable."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        warnings.warn(
            f"Hardware property {field!r} ({type(value).__name__}) is not JSON-serializable and is saved as None.",
            stacklevel=3,
        )
        return None
    return value


@dataclass
class PendingRecord:
    """A ``run_async`` call made while recording, whose job may not have finished yet."""

    shots: int | list[int] | None
    circuits: list[CircuitInfo]
    job: Job

    def collect(self) -> Record:
        """Wait for the job and turn this slot into a :class:`Record`."""
        try:
            result = self.job.result()
        except JobFailureError:
            return FailedRecord(self.shots, self.circuits, JobStatus.ERROR)
        except JobCancelledError:
            return FailedRecord(self.shots, self.circuits, JobStatus.CANCELLED)
        return DoneRecord(self.shots, self.circuits, result.all_counts, result.metadata)
