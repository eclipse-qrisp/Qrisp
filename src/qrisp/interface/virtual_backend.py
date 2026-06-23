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

"""This module defines :class:`VirtualBackend` and its associated :class:`VirtualJob`."""

import warnings
from typing import Sequence, cast

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.interface.backend import Backend
from qrisp.interface.job import Job, JobResult, JobStatus
from qrisp.misc.exceptions import QrispDeprecationWarning


class VirtualJob(Job):
    """A synchronous :class:`~qrisp.interface.Job` produced by :class:`VirtualBackend`.

    Execution is performed inline inside :meth:`submit` by calling the
    user-provided ``run_func`` once per circuit with the circuit's QASM
    representation, shot count, and token.
    """

    def __init__(
        self,
        backend: "VirtualBackend",
        circuits: Sequence[QuantumCircuit],
        shots: int | list[int] | None,
        token: str,
    ):
        """Initialise the job with the backend, normalised circuit list, shot count, and token."""
        super().__init__(backend=backend)
        self._circuits = circuits
        self._shots = shots
        self._token = token
        self._result_data = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def submit(self) -> None:
        """Execute all circuits synchronously via the user-provided ``run_func``.

        Each circuit is serialised to QASM and passed to ``run_func`` along
        with the shot count and token.  The collected counts dictionaries
        are wrapped in a :class:`~qrisp.interface.JobResult`.
        """
        self._last_known_status = JobStatus.RUNNING
        try:
            run_func = self._backend.run_func
            if isinstance(self._shots, list):
                counts_list = [
                    run_func(circuit.qasm(), s, self._token) for circuit, s in zip(self._circuits, self._shots)
                ]
            else:
                counts_list = [run_func(circuit.qasm(), self._shots, self._token) for circuit in self._circuits]
            self._result_data = JobResult(counts_list)
            self._last_known_status = JobStatus.DONE
        except Exception as exc:
            self._failure_cause = exc
            self._last_known_status = JobStatus.ERROR

    def result(self, timeout: float | None = None) -> JobResult:
        """Return the :class:`~qrisp.interface.JobResult`.

        Because execution is synchronous the result is already available as
        soon as :meth:`submit` has been called.  The *timeout* parameter is
        accepted for interface compatibility but has no effect.

        Returns
        -------
        JobResult

        Raises
        ------
        JobFailureError
            If the user-provided ``run_func`` raised an exception.

        """
        self._raise_for_status(self._last_known_status)
        return cast(JobResult, self._result_data)

    def cancel(self) -> bool:
        """Return ``False``: synchronous jobs cannot be cancelled after submission."""
        return False

    def status(self) -> JobStatus:
        """Return the current :class:`~qrisp.interface.JobStatus` of the job."""
        return self._last_known_status


class VirtualBackend(Backend):
    """A :class:`~qrisp.interface.Backend` that wraps a user-provided circuit
    execution function.

    This class replaces the legacy ``VirtualBackend`` (which depended on the
    now-removed ``BackendClient`` / ``BackendServer`` infrastructure).  It
    follows the standard :class:`~qrisp.interface.Backend` interface and
    executes circuits synchronously by calling the user-supplied
    ``run_func`` for each circuit.

    .. deprecated:: 0.8

        ``VirtualBackend`` is deprecated. New code should subclass
        :class:`~qrisp.interface.Backend` directly instead. See
        :class:`~qrisp.interface.simulators.qrisp_simulator_backend.QrispSimulatorBackend`
        for a reference implementation.

    Parameters
    ----------
    run_func : callable
        A function with signature ``run_func(qasm_str, shots, token)`` that
        receives the circuit's QASM string, an integer shot count (or
        ``None``), and a token string.  It must return a ``dict`` mapping
        bitstring outcomes to counts (or probabilities when ``shots`` is
        ``None``).
    name : str, optional
        A human-readable name for this backend.  Defaults to
        ``"VirtualBackend"``.

    Examples
    --------
    We set up a ``VirtualBackend`` that prints the received circuit and
    returns the results of the QASM simulator.  It is recommended that
    ``run_func`` provides a default value of ``None`` for the ``shots``
    parameter, substituting its own default::

        def run_func(qasm_str, shots=None, token=""):
            if shots is None:
                shots = 1000

            from qiskit import QuantumCircuit

            qiskit_qc = QuantumCircuit.from_qasm_str(qasm_str)
            print(qiskit_qc)

            from qiskit_aer import AerSimulator
            qiskit_backend = AerSimulator()

            return qiskit_backend.run(qiskit_qc, shots=shots).result().get_counts()

    >>> from qrisp.interface import VirtualBackend
    >>> example_backend = VirtualBackend(run_func)
    >>> from qrisp import QuantumFloat
    >>> qf = QuantumFloat(3)
    >>> qf[:] = 4
    >>> qf.get_measurement(backend=example_backend)
                 ┌─┐
    qf.0:   ─────┤M├──────
                 └╥┘┌─┐
    qf.1:   ──────╫─┤M├───
            ┌───┐ ║ └╥┘┌─┐
    qf.2:   ┤ X ├─╫──╫─┤M├
            └───┘ ║  ║ └╥┘
    cb_0: 1/══════╩══╬══╬═
                  0  ║  ║
    cb_1: 1/═════════╩══╬═
                     0  ║
    cb_2: 1/════════════╩═
    {4: 1.0}

    """

    def __init__(self, run_func, name=None):
        warnings.warn(
            "DeprecationWarning: VirtualBackend is deprecated and will be removed in "
            "a later release of Qrisp. Please subclass the ``Backend`` abstract class "
            "to implement custom backends.",
            QrispDeprecationWarning,
        )
        super().__init__(name=name or "VirtualBackend")
        self.run_func = run_func

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    @classmethod
    def _default_options(cls):
        """Return the default runtime options.

        ``shots=None`` enables analytic (exact probability) execution.
        ``token`` is forwarded to the user-provided ``run_func``.
        """
        return {"shots": None, "token": ""}

    def run_async(
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        shots: int | list[int] | None = None,
    ) -> VirtualJob:
        """Submit one or more circuits for execution via the user-provided ``run_func``.

        This method returns a :class:`VirtualJob` that is already
        :attr:`~qrisp.interface.JobStatus.DONE` before :meth:`run_async`
        returns, because execution is synchronous.

        Parameters
        ----------
        circuits : QuantumCircuit or list[QuantumCircuit]
            One circuit or a sequence of circuits to execute.

        shots : int or None, optional
            Number of shots.  ``None`` selects analytic execution.
            If not provided, the backend's ``shots`` option is used.

        Returns
        -------
        VirtualJob

        """
        self._check_circuit_limit(circuits)
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        else:
            circuits = list(circuits)
        if isinstance(shots, list):
            self._validate_shots_length(shots, circuits)
        n_shots = shots if shots is not None else self.options.get("shots")
        token = self.options.get("token", "")
        job = VirtualJob(backend=self, circuits=circuits, shots=n_shots, token=token)
        job.submit()
        return job
