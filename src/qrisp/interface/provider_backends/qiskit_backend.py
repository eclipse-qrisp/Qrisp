# """
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
# """

"""This module defines :class:`QiskitBackend` and its associated :class:`QiskitJob`."""

from __future__ import annotations

import warnings

import re
from collections.abc import Mapping
from typing import cast

from qiskit import QuantumCircuit, transpile
from qiskit.primitives import BackendSamplerV2
from qiskit.providers import Backend as QiskitBackendBase
from qiskit.providers import BackendV2

from qrisp.circuit.quantum_circuit import QuantumCircuit as QrispQuantumCircuit
from qrisp.interface.backend import Backend
from qrisp.interface.job import (
    JOB_FINAL_STATES,
    Job,
    JobCancelledError,
    JobFailureError,
    JobResult,
    JobStatus,
)


def _map_qiskit_status(qiskit_job) -> JobStatus:
    """Translate a Qiskit job's current status to the equivalent Qrisp JobStatus.

    Qiskit's ``VALIDATING`` state has no direct Qrisp equivalent.
    Therefore, it is mapped to ``QUEUED`` as the closest approximation.
    """
    try:
        # qiskit-ibm-runtime jobs expose .status() directly
        raw = qiskit_job.status()
    except Exception:
        return JobStatus.RUNNING

    # status() may return a string (newer runtimes) or a JobStatus enum
    name = raw.name if hasattr(raw, "name") else str(raw).upper()

    mapping = {
        "INITIALIZING": JobStatus.INITIALIZING,
        "QUEUED": JobStatus.QUEUED,
        "VALIDATING": JobStatus.QUEUED,  # no direct equivalent
        "RUNNING": JobStatus.RUNNING,
        "DONE": JobStatus.DONE,
        "CANCELLED": JobStatus.CANCELLED,
        "ERROR": JobStatus.ERROR,
    }
    return mapping.get(name, JobStatus.RUNNING)


class QiskitJob(Job):
    """
    A :class:`~qrisp.interface.Job` that wraps a Qiskit ``SamplerV2`` job.

    One ``QiskitJob`` is created per :meth:`QiskitBackend.run_async` call,
    regardless of how many circuits were submitted.  Internally it holds
    the underlying Qiskit job object and the number of circuits so it can
    extract the right slice of results when :meth:`result` is called.

    :meth:`result` blocks until the Qiskit job reaches a terminal state,
    delegating the actual waiting to Qiskit's own ``job.result()`` call.
    No Qrisp-level threading primitives are required here because Qiskit's
    ``SamplerV2.run()`` is already asynchronous.
    """

    def __init__(
        self,
        backend: QiskitBackend,
        qiskit_job,
        num_circuits: int,
    ):
        """Initialise the wrapper with the Qrisp backend, the Qiskit job, and the circuit count."""
        try:
            job_id = qiskit_job.job_id()
        except Exception:
            job_id = None
        super().__init__(backend=backend, job_id=job_id)
        self._qiskit_job = qiskit_job
        self._num_circuits = num_circuits
        self._cached_result: JobResult | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def submit(self) -> None:
        """Mark the job as QUEUED. The Qiskit job is already submitted by the sampler in run_async()."""
        self._last_known_status = JobStatus.QUEUED

    def result(self, timeout: float | None = None) -> JobResult:
        """
        Block until the Qiskit job finishes and return the :class:`~qrisp.interface.JobResult`.

        Waiting is delegated to Qiskit's own blocking ``job.result()`` call.
        If the job is already in a terminal state, this method returns (or
        raises) immediately without making a second call to Qiskit.

        Parameters
        ----------
        timeout : float or None, optional
            Maximum number of seconds to wait. Forwarded to the underlying
            Qiskit job's ``result()`` call. ``None`` (the default) waits
            indefinitely. Raises :exc:`TimeoutError` if the deadline expires.

        Returns
        -------
        JobResult

        Raises
        ------
        JobFailureError
            If the Qiskit job failed or raised an exception.

        JobCancelledError
            If the Qiskit job was cancelled.

        TimeoutError
            If *timeout* expires before the job completes.
        """
        if self._last_known_status in JOB_FINAL_STATES:
            self._raise_for_status(self._last_known_status)
            return cast(JobResult, self._cached_result)

        try:
            # Only pass timeout when explicitly requested: local backends such
            # as AerSimulator's PrimitiveJob do not accept a timeout argument.
            if timeout is not None:
                qiskit_result = self._qiskit_job.result(timeout=timeout)
            else:
                qiskit_result = self._qiskit_job.result()
        except TimeoutError:
            self._last_known_status = _map_qiskit_status(self._qiskit_job)
            raise
        except Exception as exc:
            terminal_status = _map_qiskit_status(self._qiskit_job)
            self._last_known_status = terminal_status
            if terminal_status == JobStatus.CANCELLED:
                raise JobCancelledError(
                    f"Qiskit job {self._job_id!r} was cancelled."
                ) from exc
            raise JobFailureError(f"Qiskit job {self._job_id!r} failed: {exc}") from exc

        self._last_known_status = JobStatus.DONE
        counts_list = []
        for i in range(self._num_circuits):
            counts_dict = {}
            for reg_name in qiskit_result[i].data:
                reg_data = getattr(qiskit_result[i].data, reg_name)
                for key, val in reg_data.get_counts().items():
                    clean_key = re.sub(r"\W", "", key)
                    counts_dict[clean_key] = counts_dict.get(clean_key, 0) + val
            counts_list.append(counts_dict)

        self._cached_result = JobResult(counts_list)
        return self._cached_result

    def cancel(self) -> bool:
        """
        Attempt to cancel the underlying Qiskit job.

        Returns
        -------
        bool
            ``True`` if the cancellation request was accepted by Qiskit;
            ``False`` if the job is already in a terminal state or the
            backend does not support cancellation.
        """
        if self.in_final_state():
            return False
        try:
            self._qiskit_job.cancel()
            return True
        except Exception:
            return False

    def status(self) -> JobStatus:
        """Return the current :class:`~qrisp.interface.JobStatus` by querying the Qiskit job."""
        self._last_known_status = _map_qiskit_status(self._qiskit_job)
        return self._last_known_status


class QiskitBackend(Backend):
    """
    A :class:`~qrisp.interface.Backend` that wraps a Qiskit backend.

    This allows easy access to any Qiskit-compatible simulator or real
    quantum hardware through the Qrisp backend interface.

    Circuits are converted from Qrisp's internal representation to Qiskit
    ``QuantumCircuit`` objects, transpiled for the target backend, and
    submitted through Qiskit's ``SamplerV2`` primitive.
    A ``QiskitJob`` handle is returned immediately; call
    :meth:`Job.result` to block and retrieve the :class:`~qrisp.interface.JobResult`.

    Parameters
    ----------
    backend : Qiskit backend object, optional
        A Qiskit backend object that runs ``QuantumCircuit`` objects.
        Defaults to ``AerSimulator()`` if not provided.

    name : str or None, optional
        A name for the backend.  Defaults to the Qiskit backend's own name.

    options : dict or None, optional
        Runtime options.  Defaults to ``{"shots": 1000}``.

    Examples
    --------

    **Simulation on the Aer simulator**

    We start by creating a ``QiskitBackend`` wrapping the Qiskit Aer simulator,
    and then measure a :ref:`QuantumFloat` multiplication:

    .. code-block:: python

        from qrisp import QuantumFloat
        from qrisp.interface import QiskitBackend
        from qiskit_aer import AerSimulator

        qiskit_backend = QiskitBackend(backend=AerSimulator())

        qf = QuantumFloat(4)
        qf[:] = 3
        res = qf * qf

    When ``get_measurement`` is called, Qrisp compiles the computation into a
    ``QuantumCircuit``, converts it directly to a Qiskit ``QuantumCircuit``,
    transpiles it for the target backend, and submits it through ``SamplerV2``. Internally, ``run_async`` returns a ``QiskitJob``
    immediately; :meth:`~qrisp.interface.Backend.run` then blocks on
    ``job.result()`` until execution completes and the counts are returned to
    ``get_measurement``:

    >>> res.get_measurement(backend=qiskit_backend)
    {9: 1.0}

    For local simulators the job is already ``DONE`` by the time ``run_async()``
    returns. For remote hardware backends the job will initially be
    ``QUEUED`` or ``RUNNING``, and ``result()`` will block until execution
    completes on the device.

    **Noisy simulation on a fake hardware backend**

    The same interface works with Qiskit's fake hardware backends, which
    model real device noise and connectivity constraints:

    .. code-block:: python

        from qrisp import QuantumFloat
        from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
        from qrisp.interface import QiskitBackend

        qiskit_backend = QiskitBackend(backend=FakeWashingtonV2())

        qf = QuantumFloat(2)
        qf[:] = 2
        res = qf * qf

    Internally, ``run_async()`` transpiles the circuit to the device's native gate
    set and qubit connectivity before submission, so the results reflect
    realistic noise characteristics:

    >>> res.get_measurement(backend=qiskit_backend)
    {4: 0.6962, ...}   # Note: actual counts may vary due to noise and randomness

    The result is no longer a sharp peak at ``{4: 1.0}`` because the noise
    model introduces gate errors and readout errors, spreading probability
    mass across neighbouring bitstrings.
    """

    def __init__(
        self,
        backend: QiskitBackendBase | None = None,
        name: str | None = None,
        options: Mapping | None = None,
    ):
        """Initialise the QiskitBackend, defaulting to AerSimulator if no backend is provided."""
        if backend is None:
            try:
                from qiskit_aer import AerSimulator

                backend = AerSimulator()
            except ImportError as exc:
                raise ImportError(
                    "Encountered ImportError when trying to import AerSimulator. "
                    "Install it with: pip install qiskit-aer"
                ) from exc

        self.backend = backend
        self.sampler = BackendSamplerV2(backend=cast(BackendV2, backend))

        # Fall back to the Qiskit backend's own name/options when not provided.
        # Only use the backend's options when they are a plain Mapping (e.g. dict).
        # Non-Mapping Qiskit Options objects are ignored so that _default_options()
        # applies, keeping the Backend contract satisfied.
        if name is None:
            name = getattr(backend, "name", None)
        if options is None:
            candidate = getattr(backend, "options", None)
            options = candidate if isinstance(candidate, Mapping) else None

        super().__init__(name=name, options=options)

    @classmethod
    def _default_options(cls):
        """Return the default runtime options (shots=1024)."""
        return {"shots": 1024}

    @property
    def max_circuits(self) -> int | None:
        """Maximum circuits per job, as reported by the underlying Qiskit backend.

        Delegates to the wrapped backend's own ``max_circuits`` attribute.
        Returns ``None`` if the backend does not expose a limit (e.g. local
        simulators such as ``AerSimulator``).
        """
        value = getattr(self.backend, "max_circuits", None)
        return value if isinstance(value, int) else None

    def run_async(self, circuits, shots: int | list[int] | None = None) -> QiskitJob:
        """
        Transpile and submit one or more circuits to the Qiskit backend.

        This method returns a :class:`QiskitJob` immediately.  Call
        :meth:`Job.result` on the returned object to block and retrieve
        the :class:`~qrisp.interface.JobResult`.

        Parameters
        ----------
        circuits : QuantumCircuit or Sequence[QuantumCircuit]
            One Qrisp circuit or a sequence of Qrisp circuits to execute.

        shots : int or list[int] or None, optional
            Number of shots.  If ``None``, the backend's ``shots`` option is
            used. If a ``list[int]`` is provided, the Qiskit sampler does not
            support per-circuit shot counts, so all circuits are run at
            ``max(shots)`` and a ``UserWarning`` is emitted.

        Returns
        -------
        QiskitJob
        """

        if isinstance(circuits, QrispQuantumCircuit):
            circuits = [circuits]
        else:
            circuits = list(circuits)

        if isinstance(shots, list):
            self._validate_shots_length(shots, circuits)
            warnings.warn(
                "QiskitBackend does not support per-circuit shot counts. "
                f"Running all {len(circuits)} circuits at max(shots)={max(shots)}.",
                UserWarning,
                stacklevel=2,
            )
            n_shots = max(shots)
        else:
            self._validate_shots(shots)
            n_shots = shots if shots is not None else self._options.get("shots", 1024)

        self._check_circuit_limit(circuits)

        # Convert each Qrisp circuit to a Qiskit QuantumCircuit, then rebuild
        # with integer-indexed qubits and transpile for the target backend.
        qiskit_circuits = []
        for circuit in circuits:
            qc = circuit.to_qiskit()

            # Rebuild with plain integer indices to avoid register-name mismatches.
            new_qc = QuantumCircuit(len(qc.qubits), len(qc.clbits))
            for instr in qc:
                new_qc.append(
                    instr.operation,
                    [qc.qubits.index(qb) for qb in instr.qubits],
                    [qc.clbits.index(cb) for cb in instr.clbits],
                )

            qiskit_circuits.append(transpile(new_qc, backend=self.backend))

        # Submit all circuits in a single SamplerV2 call and wrap the result.
        qiskit_job = self.sampler.run(qiskit_circuits, shots=n_shots)
        job = QiskitJob(backend=self, qiskit_job=qiskit_job, num_circuits=len(circuits))
        job.submit()
        return job


class QiskitRuntimeBackend(QiskitBackend):
    """
    A :class:`~qrisp.interface.Backend` that wraps an IBM Quantum Runtime backend.

    This allows easy access to IBM Quantum Runtime backends through the Qrisp
    backend interface. Circuits are transpiled and submitted through Qiskit's
    ``SamplerV2`` primitive, using either a direct job or a persistent session.

    It is important to close the session after execution when using
    ``mode="session"`` (see :meth:`close_session`).

    Parameters
    ----------
    api_token : str
        IBM Cloud API token used to authenticate with the Qiskit Runtime service.
    backend : str or None, optional
        Name of the IBM Quantum backend. If ``None``, the least-busy available
        backend is selected automatically.
    channel : str, optional
        Channel type for the Runtime service. Available: ``"ibm_cloud"`` or
        ``"ibm_quantum_platform"``. Defaults to ``"ibm_cloud"``.
    mode : str, optional
        Execution mode. ``"job"`` submits each circuit batch as an independent
        job; ``"session"`` opens a persistent session for lower latency across
        multiple submissions. Defaults to ``"job"``.
    instance : str or None, optional
        The Cloud Resource Name (CRN) for IBM Cloud. Passed to
        ``QiskitRuntimeService``. Defaults to ``None``.

    Attributes
    ----------
    session : Session
        The `Qiskit Runtime session <https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/session>`_.
        Only set when ``mode="session"`` is passed at construction time.

    Examples
    --------

    >>> from qrisp import QuantumFloat
    >>> from qrisp.interface import QiskitRuntimeBackend
    >>> example_backend = QiskitRuntimeBackend(api_token="YOUR_IBM_CLOUD_TOKEN", backend="ibm_brisbane", channel="ibm_cloud")
    >>> qf = QuantumFloat(2)
    >>> qf[:] = 2
    >>> res = qf * qf
    >>> result = res.get_measurement(backend=example_backend)
    >>> print(result)
    >>> # example_backend.close_session()  # Only needed when mode="session"
    {4: 0.6133,
    8: 0.1126,
    0: 0.0838,
    12: 0.0798,
    5: 0.0272,
    6: 0.016,
    9: 0.0125,
    1: 0.0117,
    13: 0.0081,
    14: 0.0073,
    3: 0.0071,
    2: 0.0062,
    10: 0.0051,
    7: 0.0044,
    11: 0.0035,
    15: 0.0014}

    """

    def __init__(
        self, api_token, backend=None, channel="ibm_cloud", mode="job", instance=None
    ):
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session
        except ImportError as exc:
            raise ImportError(
                "Please install qiskit-ibm-runtime to use QiskitRuntimeBackend: "
                "pip install qiskit-ibm-runtime"
            ) from exc

        service = QiskitRuntimeService(
            channel=channel, token=api_token, instance=instance
        )
        ibm_backend = (
            service.least_busy() if backend is None else service.backend(backend)
        )

        # Delegate common setup (self.backend, name, options) to QiskitBackend.
        # Pass _default_options() explicitly so the IBM hardware backend's internal
        # options (device config, noise model settings, etc.) do not bleed through
        # into the runtime options seen by Qrisp callers.
        super().__init__(backend=ibm_backend, options=self._default_options())

        # Replace the BackendSamplerV2 created by the parent with the IBM Runtime
        # SamplerV2, which is required for actual IBM hardware execution.
        if mode == "session":
            self.session = Session(ibm_backend)
            self.sampler = SamplerV2(self.session)
        elif mode == "job":
            self.sampler = SamplerV2(ibm_backend)
        else:
            raise ValueError(
                f"Execution mode {mode!r} not available. Choose 'job' or 'session'."
            )

    @classmethod
    def _default_options(cls):
        """Return the default runtime options (shots=1000)."""
        return {"shots": 1000}

    def close_session(self):
        """Close the IBM Runtime session opened during construction.

        Only call this when ``mode="session"`` was passed at construction time.
        """
        self.session.close()
