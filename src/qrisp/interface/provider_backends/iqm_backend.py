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
"""

from uuid import UUID

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.interface.backend import Backend
from qrisp.interface.job import (
    JOB_FINAL_STATES,
    Job,
    JobCancelledError,
    JobFailureError,
    JobResult,
    JobStatus,
)

# ---------------------------------------------------------------------------
# TEMPORARY IMPLEMENTATION — TO BE REMOVED
#
# This module is a stop-gap.  Once the QCCSW release from IQM is published,
# the classes below will be deleted and replaced by a direct import.
#
# Do NOT extend or refactor this file; any effort spent here is wasted.
# ---------------------------------------------------------------------------

__all__ = ["IQMBackend", "IQMJob"]


def _map_iqm_status(iqm_job) -> JobStatus:
    """Translate an IQM ``CircuitJob`` status to :class:`~qrisp.interface.JobStatus`.

    ``CircuitJob.status`` is a **property** (not a method); this helper accesses it
    accordingly.  IQM ``JobStatus`` values (as of iqm-client ≥ 34) are:
    ``waiting``, ``processing``, ``completed``, ``failed``, ``cancelled``.
    Any unrecognised value falls back to ``RUNNING``.
    """
    try:
        raw = iqm_job.status  # property access — NOT a method call
        name = raw.value if hasattr(raw, "value") else str(raw).lower()
    except Exception:
        return JobStatus.RUNNING

    _status_map = {
        "waiting": JobStatus.QUEUED,
        "processing": JobStatus.RUNNING,
        "completed": JobStatus.DONE,
        "failed": JobStatus.ERROR,
        "cancelled": JobStatus.CANCELLED,
    }
    return _status_map.get(name, JobStatus.RUNNING)


class IQMJob(Job):
    """
    A :class:`~qrisp.interface.Job` that wraps a single IQM ``CircuitJob``.

    .. warning::

        **Temporary implementation.**  This class exists only until the IQM
        QCCSW release ships the new Qrisp-compatible ``IQMBackend``.
        At that point, this file will be deleted and ``IQMBackend`` will be imported
        from that package instead. Do not depend on implementation details here.

    One ``IQMJob`` is created per :meth:`IQMBackend.run_async` call.
    The underlying IQM job is already submitted before this object is returned;
    :meth:`submit` only updates :attr:`~qrisp.interface.Job.last_known_status`
    to ``QUEUED``.

    :meth:`result` blocks until IQM hardware finishes by delegating to
    ``CircuitJob.wait_for_completion()``.  Counts are fetched from the server
    in histogram form via ``IQMClient.get_job_measurement_counts``.
    """

    def __init__(
        self,
        backend: "IQMBackend",
        iqm_job,
        shots_per_circuit: list[int],
        client,
    ):
        """Initialise the wrapper.

        Parameters
        ----------
        backend : IQMBackend
            The backend that created this job.
        iqm_job : CircuitJob
            The IQM client job returned by ``IQMClient.submit_circuits``.
        shots_per_circuit : list[int]
            Per-circuit shot counts (used to record the max sent to the IQM API).
        client : IQMClient
            The IQM client used to fetch measurement counts after completion.
            Stored directly to avoid accessing a protected member of the backend.
        """
        super().__init__(backend=backend, job_id=str(iqm_job.job_id))
        self._iqm_job = iqm_job
        self._shots_per_circuit = shots_per_circuit
        self._iqm_client = client
        self._cached_result: JobResult | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def submit(self) -> None:
        """Mark the job as QUEUED. The IQM job is already submitted by the client in run_async()."""
        self._last_known_status = JobStatus.QUEUED

    def result(self, timeout: float | None = None) -> JobResult:
        """
        Block until the IQM job finishes and return the :class:`~qrisp.interface.JobResult`.

        Waiting is delegated to ``CircuitJob.wait_for_completion()``.
        Counts are fetched via ``IQMClient.get_job_measurement_counts``,
        which returns per-circuit histograms (bitstring → count) directly.

        Parameters
        ----------
        timeout : float or None, optional
            Not currently forwarded to the IQM client. Included for interface
            compatibility. Pass ``None`` (the default) to wait indefinitely.

        Returns
        -------
        JobResult

        Raises
        ------
        JobFailureError
            If the IQM job failed.
        JobCancelledError
            If the IQM job was cancelled.
        """
        if self._cached_result is not None:
            return self._cached_result

        try:
            from iqm.iqm_client.iqm_client import JobStatus as IQMJobStatus

            final_iqm_status = self._iqm_job.wait_for_completion()

            if final_iqm_status == IQMJobStatus.CANCELLED:
                self._last_known_status = JobStatus.CANCELLED
                raise JobCancelledError(f"IQM job {self._job_id!r} was cancelled.")
            if final_iqm_status != IQMJobStatus.COMPLETED:
                self._last_known_status = JobStatus.ERROR
                raise JobFailureError(
                    f"IQM job {self._job_id!r} failed with status {final_iqm_status.value!r}."
                )

        except (JobCancelledError, JobFailureError):
            raise
        except Exception as exc:
            self._last_known_status = JobStatus.ERROR
            raise JobFailureError(f"IQM job {self._job_id!r} failed: {exc}") from exc

        self._last_known_status = JobStatus.DONE

        counts_batch_raw = self._iqm_client.get_job_measurement_counts(
            self._iqm_job.job_id
        )
        counts_batch = [c.counts for c in counts_batch_raw]

        self._cached_result = JobResult(counts_batch)
        return self._cached_result

    def cancel(self) -> bool:
        """
        Attempt to cancel the underlying IQM job.

        Returns
        -------
        bool
            ``True`` if the cancellation request was dispatched successfully;
            ``False`` if the job is already in a terminal state or cancellation
            failed.
        """
        if self._last_known_status in JOB_FINAL_STATES:
            return False
        try:
            self._iqm_job.cancel()
            self._last_known_status = JobStatus.CANCELLED
            return True
        except Exception:
            return False

    def status(self) -> JobStatus:
        """Return the current :class:`~qrisp.interface.JobStatus` by querying the IQM server."""
        try:
            self._iqm_job.update()
        except Exception:
            pass
        self._last_known_status = _map_iqm_status(self._iqm_job)
        return self._last_known_status


class IQMBackend(Backend):
    """
    A :class:`~qrisp.interface.Backend` for executing circuits on IQM hardware.

    .. warning::

        **Temporary implementation.**  This class exists only until the IQM
        QCCSW release ships the new Qrisp-compatible ``IQMBackend``.
        At that point, this file will be deleted and ``IQMBackend`` will be imported
        from that package instead. Do not depend on implementation details here.

    .. deprecated:: 0.8

        ``IQMBackend`` will be removed from qrisp in a future release once the
        IQM QCCSW package is publicly available.

    Circuits are transpiled from Qrisp's internal representation to Qiskit
    ``QuantumCircuit`` objects, serialized via the IQM provider backend, and
    submitted through the IQM client. An ``IQMJob`` handle is returned
    immediately. Call :meth:`Job.result` to block and retrieve the
    :class:`~qrisp.interface.JobResult`.

    Parameters
    ----------
    api_token : str
        An API token retrieved from the IQM Resonance website or IQM backend.
    device_instance : str, optional
        The device instance of the IQM backend such as ``garnet``.
        For an up-to-date list, see the IQM Resonance website.
        Required if ``server_url`` is not provided.
    server_url : str, optional
        The server URL of the IQM backend. If not provided, it defaults to IQM
        Resonance using the ``device_instance``. If a server URL is provided,
        a device instance should not be provided.
    compilation_options : CircuitCompilationOptions, optional
        An object to specify several options regarding pulse-level compilation.
    transpiler : callable, optional
        A function receiving and returning a QuantumCircuit, mapping the given
        circuit to a hardware friendly circuit. By default the
        ``transpile_to_IQM`` function will be used.
    calibration_set_id : str or UUID or None, optional
        ID of the calibration set the backend will use. ``None`` means the IQM
        Server will be queried for the current default calibration set.
    use_metrics : bool, optional
        If ``True``, the backend will query the server for calibration data and
        related quality metrics. Defaults to ``False``.
    use_timeslot : bool, optional
        Passed to ``IQMClient.submit_circuits``. Defaults to ``False``.

    Examples
    --------

    We evaluate a :ref:`QuantumFloat` multiplication on the 20-qubit IQM Garnet.

    >>> from qrisp.interface import IQMBackend
    >>> qrisp_garnet = IQMBackend(
    ...     api_token="YOUR_IQM_RESONANCE_TOKEN", device_instance="garnet"
    ... )
    >>> from qrisp import QuantumFloat
    >>> a = QuantumFloat(2)
    >>> a[:] = 2
    >>> b = a*a
    >>> b.get_measurement(backend = qrisp_garnet, shots = 1000)
    {4: 0.548, ...}

    **Manual qubit selection and routing**

    ::

        from qrisp import QuantumCircuit

        def custom_transpiler(qc: QuantumCircuit) -> QuantumCircuit:
            return qc.transpile(basis_gates = ["cz", "r", "measure", "reset"])

        custom_transpiled_garnet = IQMBackend("YOUR_IQM_RESONANCE_TOKEN",
                                   device_instance = "garnet",
                                   transpiler = custom_transpiler)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0)

        meas_res = qc.run(shots = 10000, backend = custom_transpiled_garnet)
    """

    def __init__(
        self,
        api_token: str,
        device_instance: str | None = None,
        server_url: str | None = None,
        compilation_options=None,
        transpiler=None,
        calibration_set_id: str | UUID | None = None,
        use_metrics: bool = False,
        use_timeslot: bool = False,
    ):

        if not isinstance(api_token, str):
            raise TypeError(
                "api_token must be a string. "
                "You can create an API token on the IQM Resonance website."
            )

        if server_url is not None and device_instance is not None:
            raise ValueError(
                "Please provide either a server_url or a device_instance, but not both."
            )

        if server_url is None and device_instance is None:
            raise ValueError("Please provide either a server_url or a device_instance.")

        if device_instance is not None and not isinstance(device_instance, str):
            raise TypeError(
                "device_instance must be a string. "
                "You can retrieve a list of available devices on the IQM Resonance website."
            )

        if server_url is not None and not isinstance(server_url, str):
            raise TypeError("server_url must be a string.")

        try:
            from iqm.iqm_client import CircuitCompilationOptions
            from iqm.iqm_client.iqm_client import IQMClient
            from iqm.qiskit_iqm import transpile_to_IQM
            from iqm.qiskit_iqm.iqm_provider import IQMBackend as _IQMProviderBackend
        except ImportError as exc:
            raise ImportError(
                "Please install qiskit-iqm to use the IQMBackend. "
                "You can do this by running `pip install qrisp[iqm]`."
            ) from exc

        if server_url is None:
            server_url = "https://resonance.meetiqm.com/"

        self._client = IQMClient(
            iqm_server_url=server_url, token=api_token, quantum_computer=device_instance
        )
        self._iqm_provider_backend = _IQMProviderBackend(
            self._client,
            calibration_set_id=calibration_set_id,
            use_metrics=use_metrics,
        )

        if compilation_options is None:
            compilation_options = CircuitCompilationOptions()
        self._compilation_options = compilation_options
        self._use_timeslot = use_timeslot
        self._device_instance = device_instance
        self._transpile_to_iqm = transpile_to_IQM

        if transpiler is None:

            def transpiler(qc):
                qiskit_qc = qc.to_qiskit()
                transpiled_qiskit_qc = transpile_to_IQM(
                    qiskit_qc, self._iqm_provider_backend
                )
                return QuantumCircuit.from_qiskit(transpiled_qiskit_qc)

        self._transpiler = transpiler

        name = device_instance or server_url
        super().__init__(name=name)

    @classmethod
    def _default_options(cls):
        """Return the default runtime options (shots=1000)."""
        return {"shots": 1000}

    def run_async(
        self,
        circuits,
        shots: int | list[int] | None = None,
    ) -> IQMJob:
        """
        Transpile and submit one or more circuits to the IQM backend.

        This method returns an ``IQMJob`` immediately.  Call
        :meth:`Job.result` on the returned object to block and retrieve the
        :class:`~qrisp.interface.JobResult`.

        Parameters
        ----------
        circuits : QuantumCircuit or Sequence[QuantumCircuit]
            One Qrisp circuit or a sequence of Qrisp circuits to execute.

        shots : int or list[int] or None, optional
            Number of shots.  If ``None``, the backend's ``shots`` option is
            used.  If a ``list[int]`` is provided, each entry specifies the shot
            count for the circuit at the corresponding index.  The IQM client
            always executes ``max(shots)`` repetitions for the entire batch.

        Returns
        -------
        IQMJob
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        else:
            circuits = list(circuits)

        if isinstance(shots, list):
            self._validate_shots_length(shots, circuits)
            shots_per_circuit = shots
        else:
            self._validate_shots(shots)
            n_shots = shots if shots is not None else self._options.get("shots", 1000)
            shots_per_circuit = [n_shots] * len(circuits)

        self._check_circuit_limit(circuits)

        circuit_batch = []
        for qc in circuits:
            if self._device_instance == "sirius":
                qiskit_qc = self._transpile_to_iqm(
                    qc.to_qiskit(), self._iqm_provider_backend
                )
            else:
                transpiled_qc = self._transpiler(qc)
                qiskit_qc = transpiled_qc.to_qiskit()
            circuit_batch.append(
                self._iqm_provider_backend.serialize_circuit(qiskit_qc)
            )

        iqm_job = self._client.submit_circuits(
            circuit_batch,
            options=self._compilation_options,
            shots=max(shots_per_circuit),
            use_timeslot=self._use_timeslot,
        )

        job = IQMJob(
            backend=self,
            iqm_job=iqm_job,
            shots_per_circuit=shots_per_circuit,
            client=self._client,
        )
        job.submit()
        return job
