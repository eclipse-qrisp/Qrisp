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

"""This module defines :class:`QrispSimulatorBackend` and its associated :class:`QrispSimulatorJob`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from qrisp.circuit.pass_management.pass_manager import PassManager
from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.interface.backend import Backend
from qrisp.interface.job import Job, JobResult, JobStatus
from qrisp.simulator.simulator import run as default_run


class QrispSimulatorJob(Job):
    """
    A synchronous :class:`~qrisp.interface.Job` produced by :class:`QrispSimulatorBackend`.
    """

    def __init__(
        self,
        backend: "QrispSimulatorBackend",
        circuits: Sequence,
        shots: int | list[int] | None,
    ):
        """Initialise the job with the backend, normalised circuit list, and shot count."""
        super().__init__(backend=backend)
        self._circuits = circuits
        self._shots = shots
        self._result_data = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def submit(self) -> None:
        """Execute all circuits synchronously and store the result.

        Because the built-in Qrisp simulator executes circuits inline,
        the result is fully available before :meth:`result` is ever called.
        :meth:`submit` performs the actual execution and transitions the job
        to :attr:`~qrisp.interface.JobStatus.DONE` (or
        :attr:`~qrisp.interface.JobStatus.ERROR`) synchronously.
        """

        token = self._backend.options.get("token", "")
        self._last_known_status = JobStatus.RUNNING
        try:
            if isinstance(self._shots, list):
                counts_list = [
                    default_run(circuit, shot, token)
                    for circuit, shot in zip(self._circuits, self._shots)
                ]
            else:
                counts_list = [
                    default_run(circuit, self._shots, token)
                    for circuit in self._circuits
                ]
            self._result_data = JobResult(counts_list)
            self._last_known_status = JobStatus.DONE
        except Exception as exc:
            self._failure_cause = exc
            self._last_known_status = JobStatus.ERROR

    def result(self, timeout: float | None = None) -> JobResult:
        """Return the :class:`~qrisp.interface.JobResult`.

        Because the simulator is synchronous the result is already available
        as soon as :meth:`submit` has been called by ``QrispSimulatorBackend.run_async``.
        The *timeout* parameter is accepted for interface compatibility but has
        no effect: the job is always already in a terminal state before this
        method can be called.

        Parameters
        ----------
        timeout : float or None, optional
            Maximum time to wait for the result.
            Ignored by this implementation because the result is always available immediately.  Defaults to ``None``.

        Returns
        -------
        JobResult

        Raises
        ------
        RuntimeError
            If the simulation raised an exception.
        """
        self._raise_for_status(self._last_known_status)
        return cast(JobResult, self._result_data)

    def cancel(self) -> bool:
        """Return ``False``: synchronous jobs cannot be cancelled after submission."""
        return False

    def status(self) -> JobStatus:
        """Return the current :class:`~qrisp.interface.JobStatus` of the job."""
        return self._last_known_status


class QrispSimulatorBackend(Backend):
    """
    The built-in Qrisp statevector simulator backend.

    This is the simplest concrete :class:`~qrisp.interface.Backend`
    implementation. It executes circuits synchronously. That is, the
    :class:`QrispSimulatorJob` returned by :meth:`run_async` is already
    :attr:`~qrisp.interface.JobStatus.DONE` before :meth:`run_async` returns
    to the caller.

    .. note::
        The module-level singleton :data:`~qrisp.default_backend.def_backend`
        (an instance of this class) is used as the default backend throughout
        Qrisp when no explicit backend is specified. To change the global
        default, edit :mod:`qrisp.default_backend`.

    Parameters
    ----------
    pm : PassManager or None, optional
        An optional :class:`~qrisp.circuit.pass_management.PassManager` that
        is applied to every circuit before it is submitted to the simulator.
        This allows users to inject custom transpilation or optimisation
        passes into the simulation pipeline. Defaults to ``None``.

    Examples
    --------

    **Analytic execution (default)**

    We first create a :class:`QrispSimulatorBackend`:

    >>> from qrisp import QuantumFloat
    >>> from qrisp.interface import QrispSimulatorBackend
    >>> backend = QrispSimulatorBackend()

    When ``get_measurement`` is called, Qrisp compiles the computation into a
    circuit and passes it to the built-in simulator via :meth:`run_async`. A
    :class:`QrispSimulatorJob` is returned immediately. And, because the simulator
    is synchronous the job is already :attr:`~qrisp.interface.JobStatus.DONE`
    before ``get_measurement`` even calls :meth:`~qrisp.interface.Job.result`:

    >>> qf = QuantumFloat(3)
    >>> qf[:] = 4
    >>> res = qf * qf
    >>> res.get_measurement(backend=backend)
    Simulating ... {16: 1.0}

    With ``shots=None`` the result is an exact probability distribution, so
    ``{16: 1.0}`` means the outcome ``16`` has probability ``1.0``.

    **Using a PassManager to pre-process circuits**

    >>> from qrisp import PassManager
    >>> from qrisp import convert_to_cz, fuse_adjacents
    >>> pm = PassManager()
    >>> pm += convert_to_cz()
    >>> pm += fuse_adjacents
    >>> backend = QrispSimulatorBackend(pm=pm)
    >>> # Circuits are now passed through pm before simulation

    **Inspecting circuits with visualize when evaluating expectation values**

    ``pm`` is especially useful for understanding what circuits the
    simulator actually receives.  By inserting
    :func:`~qrisp.visualize` as the last pass you can inspect every
    circuit just before it is executed.  This can for instance be used
    when evaluating expectation values via
    :meth:`QubitOperator.expectation_value <qrisp.operators.qubit.QubitOperator.expectation_value>`:
    under the hood
    the operator groups terms by commutativity, appends change-of-basis
    gates, and submits one circuit per group — details that are invisible
    from the operator expression alone.

    .. code-block:: python

        from qrisp import QuantumFloat, ry, PassManager, visualize, decompose
        from qrisp.operators import X, Z
        from qrisp.interface import QrispSimulatorBackend
        import numpy as np

        def state_prep(theta):
            qv = QuantumFloat(2)
            ry(theta, qv)
            return qv

        H = X(0)*Z(1) + Z(0)*X(1) + X(0)

        # Attach visualize at the end of the pipeline
        pm = PassManager()
        pm += decompose()
        pm += visualize
        backend = QrispSimulatorBackend(pm=pm)

        ev_function = H.expectation_value(state_prep, backend=backend)
        result = ev_function(np.pi/2)

    .. code-block:: none

               ┌─────────┐┌───┐┌─┐
         qv.0: ┤ Ry(π/2) ├┤ H ├┤M├
               ├─────────┤└┬─┬┘└╥┘
         qv.1: ┤ Ry(π/2) ├─┤M├──╫─
               └─────────┘ └╥┘  ║
        cb_15: ═════════════╬═══╩═
                            ║
        cb_16: ═════════════╩═════

               ┌─────────┐     ┌─┐
         qv.0: ┤ Ry(π/2) ├─────┤M├───
               ├─────────┤┌───┐└╥┘┌─┐
         qv.1: ┤ Ry(π/2) ├┤ H ├─╫─┤M├
               └─────────┘└───┘ ║ └╥┘
        cb_21: ═════════════════╩══╬═
                                   ║
        cb_22: ════════════════════╩═

    The measured operator contains three terms where two of them
    commute (``X(0)*Z(1)`` and ``X(0)``) and a third term that doesn't
    commute (``Z(0)*X(1)``). Non-commuting terms can not be measured
    simultaneously so we need to distinct simulator calls.

    Each circuit sent to the simulator is printed to stdout before
    execution — revealing the state preparation, the change-of-basis
    gates (e.g. Hadamards to rotate X to Z), and the qubit measurements.

    **Updating options after construction**

    Runtime options can be updated via :meth:`~qrisp.interface.Backend.update_options`.
    Only keys that were present at construction time may be modified:

    >>> backend.update_options(shots=512)
    >>> print(backend.options["shots"])
    512

    **Using the Job interface directly**

    :meth:`run_async` returns a :class:`QrispSimulatorJob` that supports the full
    :class:`~qrisp.interface.Job` interface, even though the result is
    already available synchronously:

    >>> from qrisp import QuantumFloat
    >>> from qrisp.interface import QrispSimulatorBackend
    >>> backend = QrispSimulatorBackend()
    >>> qf3 = QuantumFloat(2)
    >>> qf3[:] = 3
    >>> res3 = qf3 * qf3
    >>> qc = res3.qs.compile()
    >>> qc.measure(qc.qubits)
    >>> job = backend.run_async(qc)
    Simulating ...
    >>> print(job.status())
    done
    >>> result = job.result()
    >>> print(result.get_counts())
    {'0100111': 1.0}
    """

    def __init__(
        self,
        pm: PassManager | None = None,
    ) -> None:
        """
        Initialize the QrispSimulatorBackend.

        Parameters
        ----------
        pm : PassManager or None, optional
            An optional :class:`~qrisp.circuit.pass_management.PassManager`
            that is applied to every circuit before it is submitted to the
            simulator. Defaults to ``None``.
        """
        super().__init__(name="QrispSimulator", options=None)
        if pm is not None and not isinstance(pm, PassManager):
            raise TypeError(
                f"Expected a PassManager instance for 'pm', got {type(pm).__name__}."
            )
        self._pm = pm

    @classmethod
    def _default_options(cls):
        """Return the default runtime options.

        ``shots=None`` enables analytic (exact probability) execution.
        ``token`` is passed through to the simulator for authenticated backends.
        """
        return {"shots": None, "token": ""}

    def run_async(
        self, circuits, shots: int | list[int] | None = None
    ) -> QrispSimulatorJob:
        """Submit one or more circuits to the built-in simulator.

        This method returns a :class:`QrispSimulatorJob` that is already
        :attr:`~qrisp.interface.JobStatus.DONE` before :meth:`run_async`
        returns, because the simulator executes synchronously inside
        :meth:`~QrispSimulatorJob.submit`.

        If a :class:`~qrisp.circuit.pass_management.PassManager` was
        provided at construction time, it is applied to each circuit
        before simulation.

        Parameters
        ----------
        circuits : QuantumCircuit or list[QuantumCircuit]
            One circuit or a sequence of circuits to simulate.

        shots : int or list[int] or None, optional
            Number of shots.  ``None`` selects analytic execution.
            If a ``list[int]`` is provided, each circuit is executed with its
            own shot count.  If not provided, the backend's ``shots`` option
            is used.

        Returns
        -------
        QrispSimulatorJob
        """
        self._check_circuit_limit(circuits)
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        else:
            circuits = list(circuits)

        # Apply the pass manager (if any) to each circuit before simulation
        if self._pm is not None:
            circuits = [self._pm.run(c) for c in circuits]

        if isinstance(shots, list):
            self._validate_shots_length(shots, circuits)
        n_shots = shots if shots is not None else self.options.get("shots")
        job = QrispSimulatorJob(backend=self, circuits=circuits, shots=n_shots)
        job.submit()
        return job
