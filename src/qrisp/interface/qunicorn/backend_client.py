"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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
********************************************************************************/
"""
from typing import Optional, List
import time
import qiskit

from qrisp.interface.request_manager import RequestManager

"""
This file sets up a client adhering to the interface specified by the Qunicorn middleware
developed in the SeQuenC project: https://sequenc.de/

To learn more about Qunicorn check out the Qunicorn GitHub: https://github.com/qunicorn/qunicorn-core
And it's documentation:
https://qunicorn-core.readthedocs.io/en/latest/index.html


"""


class BackendClient:
    """
    This object allows connecting to Qunicorn backend servers.

    Parameters
    ----------
        requestManager : RequestManager
        An instance of the `RequestManager` class responsible for managing requests
        to the quantum service provider. It handles communication and sending requests.

    provider : str, optional, default="IBM"
        The optional name or identifier of the quantum service provider. This string should correspond
        to the specific provider being used, such as "IBM", "QMWare", etc.

    device : str, optional, default="aer_simulator"
        The optional identifier or name of the quantum device (e.g., "aer_simulator") to which
        the request will be sent for execution.

    token : str, optional, default=""
        An optional authentication token for secure access to the provider’s service.
        If no token is provided, the default is an empty string, implying unauthenticated access.

    batch_size : int, optional, default=1
        The number of circuits to be executed in a single batch. The default value is `1`,
        meaning a single circuit is executed per request. A higher value indicates multiple
        circuits will be submitted together in one batch.

    measurementValue : str, optional, default="COUNTS"
        The measurement type to use for the quantum circuit. Options include:
        - "COUNTS" for counting the occurrences of specific outcomes.
        - "PROBS" for returning the probabilities of measurement outcomes.
        The default is "COUNTS".

    Examples
    --------

    We assume that the example from BackendServer has been executed in the same console.

    >>> from qrisp.interface import BackendClient, RequestManager
    >>> request_manager = RequestManager(base_url="http://127.0.0.1:8080")
    >>> example_backend = BackendClient(requestManager=request_manager)
    >>> from qrisp import QuantumCircuit
    >>> qc = QuantumCircuit(2)
    >>> qc.h(0)
    >>> qc.cx(0,1)
    >>> qc.measure(qc.qubits)
    >>> example_backend.run(qc, shots = 1000)
    {'00': 510, '11': 490}

    """

    def __init__(
            self, requestManager: RequestManager, provider: str, device: str, token: str = "",
            batch_size: int = 1, measurementValue: str = "COUNTS"
    ):
        self.provider = provider
        self.device = device
        self.token = token
        self.request_manager = requestManager
        self.batch_size = batch_size
        self.circuit_batch = []
        self.measurement_value = measurementValue
        self.validate_measurement_value()

    @staticmethod
    def get_qasm(qiskit_qc: qiskit.QuantumCircuit):
        try:
            qasm_str = qiskit_qc.qasm()
        except AttributeError:
            from qiskit.qasm2 import dumps
            qasm_str = dumps(qiskit_qc)
        return qasm_str

    def validate_measurement_value(self):
        if self.measurement_value not in {"COUNTS", "PROBS"}:
            raise AttributeError("Invalid measurement value. Must be 'COUNTS' or 'PROBS'.")

    def collect_batch(self, qc, shots):
        qiskit_qc = monolithical_clreg(qc)
        qasm_str = self.get_qasm(qiskit_qc)
        self.circuit_batch.append((qasm_str, shots))

    def run(self, qc, shots):
        """
        Runs a quantum circuit with batching support.

        The method collects quantum circuits into batches of a specified size (`batch_size`). Once the batch
        is filled, it dispatches the batch for execution. If the batch size is not yet met, the method logs
        the current state and waits for more circuits to be added before dispatching.

        Parameters:
        -----------
        qc : QuantumCircuit
             The quantum circuit to be executed. This will be added to the current batch if the batch has not yet
             reached the defined `batch_size`.

        shots : int
             The number of shots (repetitions) for the quantum circuit execution. This parameter will be used in the
             dispatched batch, specifying how many times the quantum circuit will be executed for measurement.

        Returns:
        --------
        list or None
             If the batch has reached the specified `batch_size`, the method will return the current batch of circuits
             that are ready for dispatch. If the batch size has not yet been reached, `None` is returned and the process
             continues waiting for more circuits to fill the batch.

        Batching Process:
        -----------------
        1. **Circuit Collection**: The quantum circuit (`qc`) is added to an internal batch (`circuit_batch`).
        2. **Batch Size Check**: The method checks if the number of circuits in `circuit_batch` matches the defined
          `batch_size`.
        3. **Dispatch**: If the batch size is met, the batch is dispatched for execution, and the batch is cleared
           for the next set of circuits.
        4. **Logging and Waiting**: If the batch size is not met, a debug log is generated, and the method waits
           until the batch reaches the required size.
        5. **Return**: When a full batch is dispatched, the method returns the execution results
           (length is the same as `circuit_batch`); otherwise, `None` is returned while the batch is still waiting
           for more circuits.

        Example:
        --------
           If the `batch_size` is set to `5`, the method will collect quantum circuits until the batch contains 5
           circuits. Once 5 circuits are collected, they are dispatched for execution, and the `circuit_batch` is cleared.
        """
        self.collect_batch(qc, shots)
        current_batch_length = len(self.circuit_batch)
        if current_batch_length == self.batch_size:
            self.dispatch()
            res = self.circuit_batch
            self.circuit_batch = []
        else:
            #print(
            #    f"Waiting for the batch, current batch size: {current_batch_length}, expected: {self.batch_size}")
            res = None
        return res

    def dispatch(self):
        if not self.circuit_batch:
            raise ValueError("Circuit batch must not be empty")

        shots = self.circuit_batch[0][1]

        deployment_data = {
            "programs": [
                {
                    "quantumCircuit": batch_data[0],
                    "assemblerLanguage": "QASM2",
                    "pythonFilePath": "",
                    "pythonFileMetadata": "",
                }
                for batch_data in self.circuit_batch],
            "name": "",
        }

        deployment_response = self.request_manager.post(
            "/deployments/",
            json=deployment_data, verify=False
        )

        if deployment_response.status_code == 422:
            raise ValueError(
                f"Unprocessable quantum circuit {deployment_response.status_code}"
            )
        elif deployment_response.status_code != 201:
            raise RuntimeError(
                f"Failed to deploy quantum circuit. Error status code {deployment_response.status_code}"
            )

        deployment_id = deployment_response.json()["id"]

        job_data = {
            "name": "",
            "providerName": self.provider,
            "deviceName": self.device,
            "shots": shots,
            "token": self.token,
            "type": "RUNNER",
            "deploymentId": deployment_id,
        }

        job_post_response = self.request_manager.post(
            f"/jobs/", json=job_data, verify=False
        )

        if job_post_response.status_code == 422:
            raise Exception(
                f'Unprocessable job (status code: {job_post_response.status_code}, message: '
                f'{job_post_response.json()["message"]})'
            )
        elif job_post_response.status_code != 201:
            raise Exception(
                f'Failed to post job (status code: {job_post_response.status_code}, message: '
                f'{job_post_response.json()["message"]})'
            )

        job_id = job_post_response.json()["id"]

        while True:
            job_get_response = self.request_manager.get(
                f"/jobs/{job_id}/",
                json=job_data, verify=False
            )

            if job_get_response.status_code != 200:
                raise RuntimeError(
                    f'Quantum circuit execution failed: {job_get_response.json()["message"]}'
                )

            job_state = job_get_response.json()["state"]
            if job_state == "ERROR":
                raise RuntimeError("Job has failed!")

            if job_state == "FINISHED":
                break

            time.sleep(0.1)

        batch_idx = 0
        for i in range(len(job_get_response.json()["results"])):
            job_result = job_get_response.json()["results"][i]

            if job_result["resultType"] == self.measurement_value:
                measurement = job_result["data"]
                metadata = job_result["metadata"]
                counts_format = metadata["format"]
                registers: List[int] | None = None

                if counts_format == "hex":
                    registers = [r["size"] for r in metadata["registers"]]

                counts_binary = {
                    self._ensure_binary(
                        k, job_result["metadata"]["format"], registers
                    ): v
                    for k, v in measurement.items()
                }

                self.circuit_batch[
                    batch_idx] = counts_binary
                batch_idx += 1

    @staticmethod
    def _ensure_binary(
            result: str, counts_format: str, registers: Optional[list[int]]
    ) -> str:
        if counts_format == "bin":
            return result  # result is already binary
        elif counts_format == "hex":
            if registers is None:
                raise ValueError("Parameter registers is required for hex values!")

            register_counts = result.split()

            return "".join(
                format(int(val, 16), f'0{size}b')
                for val, size in zip(register_counts, registers)
            )

        else:
            raise ValueError(f"Unknown format {counts_format}")


def monolithical_clreg(qc):
    # Filter (de)allocations
    new_qc = qc.clearcopy()
    for instr in qc.data:
        if instr.op.name not in ["qb_alloc", "qb_dealloc"]:
            new_qc.append(instr)

    qc = new_qc

    qiskit_qc = qc.to_qiskit()

    new_qiskit_qc = qiskit.QuantumCircuit(len(qc.qubits), len(qc.clbits))

    for i in range(len(qiskit_qc.data)):
        qrisp_instr = qc.data[i]
        qubit_indices = [qc.qubits.index(qubit) for qubit in qrisp_instr.qubits]
        clbit_indices = [qc.clbits.index(clbit) for clbit in qrisp_instr.clbits]

        qiskit_instr = qiskit_qc.data[i]
        new_qiskit_qc.append(qiskit_instr.operation, qubit_indices, clbit_indices)

    return new_qiskit_qc
