# ********************************************************************************
# * Copyright (c) 2026 the Qrisp Authors
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

"""This module defines :class:`QiskitBackend` and its associated :class:`QiskitJob`."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import cast

from qiskit import QuantumCircuit, transpile
from qiskit.providers import Backend as QiskitBackendBase
from qiskit.providers import BackendV2

from qrisp.interface.backend import Backend
from qrisp.interface.job import Job, JobResult, JobStatus
from qrisp.interface.virtual_backend import VirtualBackend


def _map_qiskit_status(qiskit_job) -> JobStatus:
    """Translate a Qiskit job's current status to the equivalent Qrisp JobStatus.

    Qiskit's ``VALIDATING`` state has no direct Qrisp equivalent.
    Therefore, it is mapped to ``QUEUED`` as the closest approximation.
    """
    try:
        # qiskit-ibm-runtime jobs expose .status() directly
        raw = qiskit_job.status()
    except Exception:
        return JobStatus.INITIALIZING

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
        super().__init__(backend=backend)
        self._qiskit_job = qiskit_job
        self._num_circuits = num_circuits

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def submit(self) -> None:
        """No-op: the Qiskit job is submitted by the sampler inside QiskitBackend.run_async()."""

    def result(self) -> JobResult:
        """
        Block until the Qiskit job finishes and return the :class:`~qrisp.interface.JobResult`.

        Waiting is delegated to Qiskit's own blocking ``job.result()`` call,
        so no additional synchronisation is needed on the Qrisp side.

        Returns
        -------
        JobResult

        Raises
        ------
        RuntimeError
            If the Qiskit job failed or was cancelled.
        """
        if self.status() == JobStatus.CANCELLED:
            raise RuntimeError("QiskitJob was cancelled.")

        try:
            qiskit_result = self._qiskit_job.result()
        except Exception as exc:
            raise RuntimeError(f"Qiskit job failed: {exc}") from exc

        counts_list = []
        for i in range(self._num_circuits):
            counts_dict = {}
            for reg_name in qiskit_result[i].data:
                reg_data = getattr(qiskit_result[i].data, reg_name)
                for key, val in reg_data.get_counts().items():
                    clean_key = re.sub(r"\W", "", key)
                    counts_dict[clean_key] = counts_dict.get(clean_key, 0) + val
            counts_list.append(counts_dict)

        return JobResult(counts_list)

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
        return _map_qiskit_status(self._qiskit_job)


class QiskitBackend(Backend):
    """
    A :class:`~qrisp.interface.Backend` that wraps a Qiskit backend.

    This allows easy access to any Qiskit-compatible simulator or real
    quantum hardware through the Qrisp backend interface.

    Circuits are converted from Qrisp's internal representation to Qiskit
    ``QuantumCircuit`` objects via OpenQASM 2, transpiled for the target
    backend, and submitted through Qiskit's ``SamplerV2`` primitive.
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
        Runtime options.  Defaults to ``{"shots": 1024}``.

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
    ``QuantumCircuit``, converts it to OpenQASM 2, rebuilds it as a Qiskit
    ``QuantumCircuit``, transpiles it for the target backend, and submits it
    through ``SamplerV2``. A ``QiskitJob`` is returned immediately, and
    ``get_measurement`` then blocks on ``job.result()`` to retrieve the outcome:

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

        try:
            from qiskit_ibm_runtime import SamplerV2
        except ImportError as exc:
            raise ImportError(
                "Please install qiskit-ibm-runtime to use QiskitBackend: "
                "pip install qiskit-ibm-runtime"
            ) from exc

        self.sampler = SamplerV2(cast(BackendV2, backend))

        # Fall back to the Qiskit backend's own name/options when not provided.
        if name is None:
            name = getattr(backend, "name", None)
        if options is None:
            options = getattr(backend, "options", None)

        super().__init__(name=name, options=options)

    @classmethod
    def _default_options(cls):
        """Return the default runtime options (shots=1024)."""
        return {"shots": 1024}

    def run_async(
        self,
        circuits,
        shots: int | None = None,
    ) -> QiskitJob:
        """
        Transpile and submit one or more circuits to the Qiskit backend.

        This method returns a :class:`QiskitJob` immediately.  Call
        :meth:`Job.result` on the returned object to block and retrieve
        the :class:`~qrisp.interface.JobResult`.

        Parameters
        ----------
        circuits : QuantumCircuit or Sequence[QuantumCircuit]
            One Qrisp circuit or a sequence of Qrisp circuits to execute.

        shots : int or None, optional
            Number of shots.  If ``None``, the backend's ``shots`` option is used.

        Returns
        -------
        QiskitJob
        """
        circuits = [circuits] if not isinstance(circuits, Sequence) else circuits
        n_shots = shots if shots is not None else self._options.get("shots", 1024)

        # Convert each Qrisp circuit to a Qiskit QuantumCircuit via OpenQASM 2,
        # then rebuild with integer-indexed qubits and transpile for the target backend.
        qiskit_circuits = []
        for circuit in circuits:
            qasm_str = circuit.qasm()
            qc = QuantumCircuit.from_qasm_str(qasm_str)

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


class QiskitRuntimeBackend(VirtualBackend):
    """
    This class instantiates a VirtualBackend using a Qiskit Runtime backend.
    This allows easy access to Qiskit Runtime backends through the qrisp interface.
    It is imporant to close the session after the execution of the algorithm
    (as reported in the example below).

    .. warning::

        This Backend has not been migrated to the new Backend API yet, and is not guaranteed to
        work with the latest versions of Qiskit Runtime and qrisp.
        Use with caution and refer to the documentation for the latest updates.

    Parameters
    ----------
    api_token : str
        The token is necessary to create correctly the Qiskit Runtime
        service and be able to run algorithms on their backends.
    backend : str, optional
        A string associated to the name of a Qiskit Runtime backend.
        By default, the least busy available backend is selected.
    channel : str, optional
        The channel type. Available are ``ibm_cloud`` or ``ibm_quantum_platform``.
        The default is ``ibm_cloud``.
    mode : str, optional
        The `execution mode <https://quantum.cloud.ibm.com/docs/en/guides/execution-modes>`_. Available are ``job`` and ``session``.
        The default is ``job``.

    Attributes
    ----------
    session : Session
        The `Qiskit Runtime session <https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/session>`_.

    Examples
    --------

    >>> from qrisp import QuantumFloat
    >>> from qrisp.interface import QiskitRuntimeBackend
    >>> example_backend = QiskitRuntimeBackend(api_token = "YOUR_IBM_CLOUD_TOKEN", backend = "ibm_brisbane", channel = "ibm_cloud")
    >>> qf = QuantumFloat(2)
    >>> qf[:] = 2
    >>> res = qf*qf
    >>> result = res.get_measurement(backend = example_backend)
    >>> print(result)
    >>> # example_backend.close_session() # Use only when mode = "session"
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

    def __init__(self, api_token, backend=None, channel="ibm_cloud", mode="job"):

        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session
        except ImportError as exc:
            raise ImportError(
                "Please install qiskit-ibm-runtime to use the QiskitBackend. You can do this by running `pip install qiskit-ibm-runtime`."
            ) from exc

        service = QiskitRuntimeService(channel=channel, token=api_token)
        if backend is None:
            backend = service.least_busy()
        else:
            backend = service.backend(backend)

        if mode == "session":
            self.session = Session(backend)
            sampler = SamplerV2(self.session)
        elif mode == "job":
            sampler = SamplerV2(backend)
        else:
            raise ValueError("Execution mode" + str(mode) + " not available.")

        # Create the run method
        def run(qasm_str, shots=None, token=""):
            if shots is None:
                shots = 1000

            # Convert to qiskit
            qiskit_qc = QuantumCircuit.from_qasm_str(qasm_str)

            # Make circuit with one monolithic register
            new_qiskit_qc = QuantumCircuit(len(qiskit_qc.qubits), len(qiskit_qc.clbits))
            for instr in qiskit_qc:
                new_qiskit_qc.append(
                    instr.operation,
                    [qiskit_qc.qubits.index(qb) for qb in instr.qubits],
                    [qiskit_qc.clbits.index(cb) for cb in instr.clbits],
                )

            qiskit_qc = transpile(new_qiskit_qc, backend=backend)

            job = sampler.run([qiskit_qc], shots=shots)

            qiskit_result = (
                job.result()[0].data.c.get_counts()
                # https://docs.quantum.ibm.com/migration-guides/v2-primitives
            )

            # Remove the spaces in the qiskit result keys
            result_dic = {}

            for key in qiskit_result.keys():
                counts_string = re.sub(r"\W", "", key)
                result_dic[counts_string] = qiskit_result[key]

            return result_dic

        super().__init__(run)

    def close_session(self):
        """
        Method to call in order to close the session started by the init method.
        """
        self.session.close()
