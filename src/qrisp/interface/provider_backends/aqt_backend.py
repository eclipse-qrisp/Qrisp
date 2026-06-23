"""********************************************************************************
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

import warnings

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

__all__ = ["AQTBackend", "AQTJob"]


def _map_aqt_status(aqt_job) -> JobStatus:
    """Translate an AQT (Qiskit-compatible) job's status to :class:`~qrisp.interface.JobStatus`.

    ``VALIDATING`` has no direct Qrisp equivalent and is mapped to ``QUEUED``.
    Any unrecognised status string falls back to ``RUNNING``.
    """
    try:
        raw = aqt_job.status()
    except Exception:
        return JobStatus.RUNNING

    name = raw.name if hasattr(raw, "name") else str(raw).upper()

    _MAP = {
        "INITIALIZING": JobStatus.INITIALIZING,
        "QUEUED": JobStatus.QUEUED,
        "VALIDATING": JobStatus.QUEUED,
        "RUNNING": JobStatus.RUNNING,
        "DONE": JobStatus.DONE,
        "CANCELLED": JobStatus.CANCELLED,
        "ERROR": JobStatus.ERROR,
    }
    return _MAP.get(name, JobStatus.RUNNING)


class AQTJob(Job):
    """A :class:`~qrisp.interface.Job` that wraps a single AQT primitive job.

    One ``AQTJob`` is created per :meth:`AQTBackend.run_async` call.
    The underlying AQT job is already submitted before this object is returned;
    :meth:`submit` only updates :attr:`~qrisp.interface.Job.last_known_status` to ``QUEUED``.

    :meth:`result` blocks until the AQT hardware finishes, delegating the wait
    to the AQT job's own ``result()`` call.

    .. note::

        AQT does not expose a job cancellation API via its Qiskit-compatible
        sampler. :meth:`cancel` therefore always returns ``False``.
    """

    def __init__(self, backend: "AQTBackend", aqt_job, cl_bits_per_circuit: list[int]):
        """Initialise the wrapper with the Qrisp backend, the AQT job, and the
        per-circuit classical-bit counts needed to format result bitstrings.
        """
        super().__init__(backend=backend)
        self._aqt_job = aqt_job
        self._cl_bits_per_circuit = cl_bits_per_circuit

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def submit(self) -> None:
        """Mark the job as QUEUED. The AQT job is already submitted by the sampler in run_async()."""
        self._last_known_status = JobStatus.QUEUED

    def result(self, timeout: float | None = None) -> JobResult:
        """Block until the AQT job finishes and return the :class:`~qrisp.interface.JobResult`.

        Waiting is delegated to the AQT job's own blocking ``result()`` call.

        Parameters
        ----------
        timeout : float or None, optional
            Not currently forwarded to the AQT job. Included for interface
            compatibility. Pass ``None`` (the default) to wait indefinitely.

        Returns
        -------
        JobResult

        Raises
        ------
        JobFailureError
            If the AQT job failed or raised an exception.
        JobCancelledError
            If the AQT job was cancelled.

        """
        try:
            aqt_result = self._aqt_job.result()
        except TimeoutError:
            self._last_known_status = _map_aqt_status(self._aqt_job)
            raise
        except Exception as exc:
            terminal_status = _map_aqt_status(self._aqt_job)
            self._last_known_status = terminal_status
            if terminal_status == JobStatus.CANCELLED:
                raise JobCancelledError(f"AQT job {self._job_id!r} was cancelled.") from exc
            raise JobFailureError(f"AQT job {self._job_id!r} failed: {exc}") from exc

        self._last_known_status = JobStatus.DONE
        result_dicts = [
            {bin(outcome)[2:].zfill(self._cl_bits_per_circuit[i]): prob for outcome, prob in quasi_dist.items()}
            for i, quasi_dist in enumerate(aqt_result.quasi_dists)
        ]
        return JobResult(result_dicts)

    def cancel(self) -> bool:
        """AQT does not expose a job cancellation API.

        Returns
        -------
        bool
            Always ``False``.

        """
        return False

    def status(self) -> JobStatus:
        """Return the current :class:`~qrisp.interface.JobStatus` by querying the AQT job."""
        self._last_known_status = _map_aqt_status(self._aqt_job)
        return self._last_known_status


class AQTBackend(Backend):
    """A :class:`~qrisp.interface.Backend` that executes circuits on AQT
    quantum hardware via `AQT ARNICA <https://www.aqt.eu/products/arnica/>`_.

    Circuits are transpiled from Qrisp's internal representation to Qiskit
    ``QuantumCircuit`` objects and submitted through AQT's ``AQTSampler``
    primitive. An ``AQTJob`` handle is returned immediately. Call
    :meth:`Job.result` to block and retrieve the
    :class:`~qrisp.interface.JobResult`.

    For further details, see the :ref:`Backend` documentation
    and the `AQT ARNICA <https://www.aqt.eu/products/arnica/>`_ website.

    .. note::

        Requires the optional dependency ``qiskit-aqt-provider``::

            pip install qiskit-aqt-provider

    .. note::

        ``AQTSampler`` returns quasi-probability distributions
        (float values that sum to 1.0) rather than integer counts.
        These are returned as-is and are handled by Qrisp's measurement
        normalisation logic.

    Parameters
    ----------
    api_token : str
        API token for AQT ARNICA.
    device_instance : str
        The device instance to target, e.g. ``"ibex"`` or
        ``"simulator_noise"``. The offline simulators
        ``"offline_simulator_no_noise"`` and ``"offline_simulator_noise"``
        are available with any API token.
        For an up-to-date list see the AQT ARNICA website.
    workspace : str | None = None
        The workspace identifier for your company or project.

    Examples
    --------
    We evaluate a :ref:`QuantumFloat` multiplication on the 12-qubit AQT IBEX:

    >>> from qrisp import QuantumFloat
    >>> from qrisp.interface import AQTBackend
    >>> qrisp_ibex = AQTBackend(
    ...     api_token="YOUR_AQT_ARNICA_TOKEN",
    ...     device_instance="ibex",
    ...     workspace="YOUR_COMPANY_OR_PROJECT_NAME",
    ... )
    >>> a = QuantumFloat(2)
    >>> a[:] = 2
    >>> b = a * a
    >>> b.get_measurement(backend=qrisp_ibex, shots=100)
    {4: 0.49, 8: 0.11, 2: 0.08, 0: 0.06, ...}

    """

    def __init__(self, api_token: str, device_instance: str, workspace: str | None = None):
        if not isinstance(api_token, str):
            raise TypeError("api_token must be a string. You can create an API token on the AQT ARNICA website.")
        if workspace is not None and not isinstance(workspace, str):
            raise TypeError("workspace must be a string.")
        if not isinstance(device_instance, str):
            raise TypeError(
                "Please provide a device_instance as a string. "
                "You can retrieve a list of available devices on the AQT ARNICA website."
            )

        try:
            from qiskit_aqt_provider import AQTProvider
            from qiskit_aqt_provider.primitives import AQTSampler
        except ImportError as exc:
            raise ImportError(
                "Please install qiskit-aqt-provider to use AQTBackend: pip install qiskit-aqt-provider"
            ) from exc

        provider = AQTProvider(api_token)
        self._aqt_device = provider.get_backend(name=device_instance, workspace=workspace)
        self._aqt_sampler = AQTSampler

        name = self._aqt_device.name if isinstance(self._aqt_device.name, str) else self._aqt_device.name()
        super().__init__(name=name)

    @classmethod
    def _default_options(cls):
        """Return the default runtime options (shots=100, matching the original AQT default)."""
        return {"shots": 100}

    @property
    def max_circuits(self) -> int | None:
        """Maximum circuits per job, as reported by the underlying AQT device.

        Delegates to the AQT device's own ``max_circuits`` attribute, which is
        part of the Qiskit-compatible ``BackendV2`` interface that AQT exposes.
        Returns ``None`` if the device does not expose a limit.
        """
        value = getattr(self._aqt_device, "max_circuits", None)
        return value if isinstance(value, int) else None

    def run_async(self, circuits, shots: int | list[int] | None = None) -> AQTJob:
        """Transpile and submit one or more circuits to the AQT backend.

        This method returns an ``AQTJob`` immediately.  Call
        ``Job.result`` on the returned object to block and retrieve the
        :class:`~qrisp.interface.JobResult`.

        Parameters
        ----------
        circuits : QuantumCircuit or Sequence[QuantumCircuit]
            One Qrisp circuit or a sequence of Qrisp circuits to execute.

        shots : int or list[int] or None, optional
            Number of shots.  If ``None``, the backend's ``shots`` option is
            used. If a ``list[int]`` is provided, the AQT sampler does not
            support per-circuit shot counts, so all circuits are run at
            ``max(shots)`` and a ``UserWarning`` is emitted.

        Returns
        -------
        AQTJob

        """
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        else:
            circuits = list(circuits)

        if isinstance(shots, list):
            self._validate_shots_length(shots, circuits)
            warnings.warn(
                "AQTBackend does not support per-circuit shot counts. "
                f"Running all {len(circuits)} circuits at max(shots)={max(shots)}.",
                UserWarning,
                stacklevel=2,
            )
            n_shots = max(shots)
        else:
            self._validate_shots(shots)
            n_shots = shots if shots is not None else self._options.get("shots", 100)

        self._check_circuit_limit(circuits)

        # Transpile each Qrisp circuit and convert to a Qiskit QuantumCircuit.
        # Re-index to a single monolithic register to avoid register-name issues.
        qiskit_circuits = []
        cl_bits_per_circuit = []

        for qc in circuits:
            qiskit_qc = qc.transpile().to_qiskit()
            new_qiskit_qc = QiskitQuantumCircuit(len(qiskit_qc.qubits), len(qiskit_qc.clbits))
            for instr in qiskit_qc:
                new_qiskit_qc.append(
                    instr.operation,
                    [qiskit_qc.qubits.index(qb) for qb in instr.qubits],
                    [qiskit_qc.clbits.index(cb) for cb in instr.clbits],
                )
            qiskit_circuits.append(new_qiskit_qc)
            cl_bits_per_circuit.append(len(qiskit_qc.clbits))

        sampler = self._aqt_sampler(self._aqt_device)
        sampler.set_transpile_options(optimization_level=3)
        aqt_job = sampler.run(qiskit_circuits, shots=n_shots)

        job = AQTJob(
            backend=self,
            aqt_job=aqt_job,
            cl_bits_per_circuit=cl_bits_per_circuit,
        )
        job.submit()
        return job
