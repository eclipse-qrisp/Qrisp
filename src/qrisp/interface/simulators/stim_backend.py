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

from typing import Sequence, cast

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.interface.backend import Backend
from qrisp.interface.job import Job, JobResult, JobStatus


def _run_on_stim(qc: QuantumCircuit, shots: int | None):
    if shots is None:
        shots = 10000

    stim_circuit, measurement_map = qc.to_stim(return_measurement_map=True)

    sampler = stim_circuit.compile_sampler()
    shot_array = sampler.sample(shots=shots)

    permutation = [measurement_map[cb] for cb in qc.clbits]
    permuted_shot_array = shot_array[:, permutation]

    counts = {}
    for row in permuted_shot_array:
        bitstring = "".join(str(int(b)) for b in reversed(row))
        counts[bitstring] = counts.get(bitstring, 0) + 1

    return counts


class _StimJob(Job):
    """Synchronous :class:`~qrisp.interface.Job` for :class:`StimBackend`."""

    def __init__(self, backend: "StimBackend", circuits: Sequence, shots):
        super().__init__(backend=backend)
        self._circuits = circuits
        self._shots = shots
        self._result_data = None
        self._error = None

    def submit(self) -> None:
        self._last_known_status = JobStatus.RUNNING
        try:
            counts_list = [_run_on_stim(qc, self._shots) for qc in self._circuits]
            self._result_data = JobResult(counts_list)
            self._last_known_status = JobStatus.DONE
        except Exception as exc:
            self._error = exc
            self._last_known_status = JobStatus.ERROR

    def result(self, timeout=None) -> JobResult:
        if self._last_known_status == JobStatus.ERROR:
            raise self._error
        self._raise_for_status(self._last_known_status)
        return cast(JobResult, self._result_data)

    def cancel(self) -> bool:
        return False

    def status(self) -> JobStatus:
        return self._last_known_status


class StimBackend(Backend):
    """
    A :class:`~qrisp.interface.Backend` that simulates Clifford circuits via
    `Stim <https://github.com/quantumlib/Stim>`_.

    :meth:`run` returns a :class:`~qrisp.interface.MeasurementResult` immediately.
    For lazy, buffered execution call :meth:`~qrisp.interface.Backend.batched`
    first::

        bb = StimBackend().batched()
        res = qv.get_measurement(backend=bb)
        bb.dispatch()

    Parameters
    ----------
    options : Mapping or None, optional
        Runtime options.  Defaults to ``{"shots": 10000}``.

    Examples
    --------

    ::

        from qrisp import QuantumVariable
        from qrisp.interface import StimBackend

        qv = QuantumVariable(2)
        qv[:] = "10"
        res = qv.get_measurement(backend=StimBackend())
        print(res)
        # Yields: {'10': 1.0}
    """

    @classmethod
    def _default_options(cls):
        return {"shots": 10000}

    def run_async(self, circuits, shots=None) -> _StimJob:
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        else:
            circuits = list(circuits)
        n_shots = shots if shots is not None else self.options.get("shots")
        job = _StimJob(backend=self, circuits=circuits, shots=n_shots)
        job.submit()
        return job
