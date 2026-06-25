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

********************************************************************************
* Kipu Quantum Hub — Qiskit API-based backend for Qrisp
*
* Wraps https://docs.hub.kipu-quantum.com/sdk-api-quantum in a Qrisp-compatible Backend.
* Supports so far Azure IonQ Simulator
* 
*
* Auth (pick one):
*   1. qhubctl login -t <token>          (CLI — auto-injected)
*   2. KipuBackend(..., access_token="<token>")
*   3. Env var  KIPU_ACCESS_TOKEN
*
* Install:
*   pip install qhub-api
********************************************************************************
"""

import time
import warnings
import importlib
from typing import Sequence, Dict, Any, List

from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.interface.backend import Backend
from qrisp.interface.job import (
    Job,
    JobCancelledError,
    JobFailureError,
    JobResult,
    JobStatus,
)

__all__ = ["KipuBackend", "KipuJob"]


def _map_kipu_status(client, job_id: str) -> JobStatus:
    """Translate a Kipu Hub job status string to a Qrisp :class:`~qrisp.interface.JobStatus`."""
    try:
        raw_status = client.jobs.get_job_status(job_id).status.upper()
    except Exception:
        return JobStatus.RUNNING

    _MAP = {
        "PENDING": JobStatus.QUEUED,
        "RUNNING": JobStatus.RUNNING,
        "SUCCEEDED": JobStatus.DONE,
        "COMPLETED": JobStatus.DONE,
        "CANCELLED": JobStatus.CANCELLED,
        "FAILED": JobStatus.ERROR,
        "ERROR": JobStatus.ERROR,
    }
    return _MAP.get(raw_status, JobStatus.RUNNING)


class KipuJob(Job):
    """
    A :class:`~qrisp.interface.Job` that wraps asynchronous Kipu Hub jobs.

    Supports tracking multiple underlying jobs when a batch of circuits is dispatched.
    """

    def __init__(self, backend: "KipuBackend", client, job_ids: List[str], shots: int):
        super().__init__(backend=backend)
        self._client = client
        self._job_ids = job_ids
        self._shots = shots

    def submit(self) -> None:
        """Mark the job as QUEUED. The jobs are already submitted during run_async()."""
        self._last_known_status = JobStatus.QUEUED

    def result(self, timeout: float | None = None) -> JobResult:
        """Block until all associated Kipu jobs complete and return a :class:`~qrisp.interface.JobResult`."""
        start_time = time.time()
        poll_interval = 2

        for job_id in self._job_ids:
            while True:
                if timeout is not None and (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Polling timed out for Kipu job {job_id!r}")

                status = _map_kipu_status(self._client, job_id)
                self._last_known_status = status

                if status == JobStatus.DONE:
                    break
                elif status == JobStatus.CANCELLED:
                    raise JobCancelledError(f"Kipu job {job_id!r} was cancelled.")
                elif status == JobStatus.ERROR:
                    raise JobFailureError(f"Kipu job {job_id!r} failed on the execution layer.")

                time.sleep(poll_interval)

        self._last_known_status = JobStatus.DONE

        result_dicts = []
        for job_id in self._job_ids:
            try:
                raw_results = self._client.jobs.get_job_result(job_id)
            except Exception as exc:
                raise JobFailureError(f"Failed to fetch results for job {job_id!r}: {exc}") from exc
            
            result_dicts.append(self._post_process_to_qrisp(raw_results))

        return JobResult(result_dicts)

    def cancel(self) -> bool:
        """Attempt to cancel all active jobs. Returns True if all actions completed."""
        if not hasattr(self._client.jobs, "cancel_job"):
            return False
        try:
            for job_id in self._job_ids:
                self._client.jobs.cancel_job(job_id)
            self._last_known_status = JobStatus.CANCELLED
            return True
        except Exception:
            return False

    def status(self) -> JobStatus:
        """Query and aggregate the execution status from the Kipu Hub endpoints."""
        if not self._job_ids:
            return JobStatus.INITIALIZING
        
        statuses = [_map_kipu_status(self._client, j_id) for j_id in self._job_ids]
        if any(s == JobStatus.ERROR for s in statuses):
            self._last_known_status = JobStatus.ERROR
        elif any(s == JobStatus.CANCELLED for s in statuses):
            self._last_known_status = JobStatus.CANCELLED
        elif any(s == JobStatus.RUNNING for s in statuses):
            self._last_known_status = JobStatus.RUNNING
        elif any(s == JobStatus.QUEUED for s in statuses):
            self._last_known_status = JobStatus.QUEUED
        else:
            self._last_known_status = JobStatus.DONE
            
        return self._last_known_status

    def _post_process_to_qrisp(self, raw_results: Any) -> Dict[str, float]:
        """Normalize disparate result payload structures to standard Qrisp histograms."""
        if isinstance(raw_results, dict) and "histogram" in raw_results:
            counts_dict = raw_results["histogram"]
        elif isinstance(raw_results, list) and len(raw_results) > 0:
            counts_dict = raw_results[0]
        elif hasattr(raw_results, "results") and len(raw_results.results) > 0:
            counts_dict = raw_results.results[0]
        else:
            counts_dict = raw_results

        if not isinstance(counts_dict, dict):
            counts_dict = getattr(counts_dict, "counts", {}) or counts_dict.get("counts", {}) or {}

        processed_results = {}
        for bitstring, val in counts_dict.items():
            if isinstance(val, float):
                processed_results[str(bitstring)] = round(val, 4)
            else:
                processed_results[str(bitstring)] = round(val / self._shots, 4)

        return processed_results


class KipuBackend(Backend):
    """
    A :class:`~qrisp.interface.Backend` targeting dynamic Kipu Quantum hardware and simulators.
    
    Dynamically maps backend_ids to their respective Input and CircuitItem classes.
    """

    def __init__(self, api_key: str, backend_id: str = "azure.ionq.simulator"):
        if not isinstance(api_key, str):
            raise TypeError("api_key must be a string.")
        if not isinstance(backend_id, str):
            raise TypeError("backend_id must be a string.")

        self._backend_id = backend_id
        
        # Resolve class names based on standard string parsing patterns
        # E.g., "azure.ionq.simulator" -> "AzureIonqSimulator"
        # E.g., "aws.ionq.forte" -> "AwsIonqForte"
        # E.g., "kipu.sim.qsim" -> "KipuSimQsim"
        components = backend_id.replace(".", "_").split("_")
        camel_case_suffix = "".join(c.capitalize() for c in components)
        
        input_cls_name = f"CreateJobRequestInput_{camel_case_suffix}"
        
        # Circuit item maps matching standard naming variants (e.g. AzureIonqJobInputCircuitItem)
        # For non-azure items, we strip out specific segments to accommodate the type sdk convention
        provider_prefix = components[0].capitalize()
        engine_mid = "".join(c.capitalize() for c in components[1:-1])
        item_cls_name = f"{provider_prefix}{engine_mid}JobInputCircuitItem"
        
        # Override specific edge case naming rules if any deviate in the SDK layers
        if "Azure" in camel_case_suffix:
            item_cls_name = "AzureIonqJobInputCircuitItem"

        try:
            hub_mod = importlib.import_module("qhub.api.quantum")
            jobs_mod = importlib.import_module("qhub.api.quantum.jobs")
            types_mod = importlib.import_module("qhub.api.quantum.types")
            
            HubQuantumClient = getattr(hub_mod, "HubQuantumClient")
            self._input_cls = getattr(jobs_mod, input_cls_name)
            self._item_cls = getattr(types_mod, item_cls_name)
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f"Failed to load target API schemas for backend {backend_id!r}. "
                "Please make sure the required qhub API client libraries are installed."
            ) from exc

        self._client = HubQuantumClient(api_key=api_key)
        super().__init__(name=backend_id)

    @classmethod
    def _default_options(cls):
        return {"shots": 100}

    def run_async(self, circuits: QuantumCircuit | Sequence[QuantumCircuit], shots: int | List[int] | None = None) -> KipuJob:
        """Transpile and asynchronously execute one or multiple circuits."""
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        else:
            circuits = list(circuits)

        if isinstance(shots, list):
            self._validate_shots_length(shots, circuits)
            warnings.warn(
                "KipuBackend does not support per-circuit shot counts. "
                f"Running all circuits at max(shots)={max(shots)}.",
                UserWarning,
                stacklevel=2,
            )
            n_shots = max(shots)
        else:
            self._validate_shots(shots)
            n_shots = shots if shots is not None else self._options.get("shots", 100)

        self._check_circuit_limit(circuits)
        job_ids = []

        for qc in circuits:
            native_items = self._pre_process_circuit(qc)
            
            job_input = self._input_cls(
                circuit=native_items,
                gateset="qis",
                qubits=qc.num_qubits() if hasattr(qc, "num_qubits") else 2,
            )
            
            try:
                native_job = self._client.jobs.create_job(
                    backend_id=self._backend_id,
                    shots=n_shots,
                    input=job_input,
                )
                job_ids.append(native_job.id)
            except Exception as exc:
                raise RuntimeError(f"Failed to dispatch job to Kipu Hub: {exc}") from exc

        job = KipuJob(backend=self, client=self._client, job_ids=job_ids, shots=n_shots)
        job.submit()
        return job

    def _pre_process_circuit(self, qc: QuantumCircuit) -> List[Any]:
        """Convert a Qrisp circuit representation down to Kipu direct native target items."""
        qiskit_qc = qc.to_qiskit()
        native_items = []
        
        for instruction in qiskit_qc.data:
            gate_name = instruction.operation.name
            if gate_name in ["measure", "barrier"]:
                continue
            
            qubit_indices = [qiskit_qc.find_bit(q).index for q in instruction.qubits]
            if gate_name == "cx":
                gate_name = "cnot"
                
            if len(qubit_indices) == 1:
                native_items.append(
                    self._item_cls(gate=gate_name, targets=[qubit_indices[0]])
                )
            elif len(qubit_indices) == 2:
                native_items.append(
                    self._item_cls(
                        gate=gate_name, 
                        controls=[qubit_indices[0]], 
                        targets=[qubit_indices[1]]
                    )
                )
        return native_items