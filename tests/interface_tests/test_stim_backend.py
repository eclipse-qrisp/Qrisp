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

from qrisp import QuantumCircuit, QuantumVariable
from qrisp.interface import StimBackend
from qrisp.interface.measurement_result import LazyDict


def _build_deterministic_circuit():
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.x(1)
    qc.measure([0, 1])
    return qc


def test_stim_backend_run_counts():
    qc = _build_deterministic_circuit()
    backend = StimBackend()
    counts = backend.run(qc, shots=200)
    assert sum(counts.values()) == 200
    assert counts == {"11": 200}


def test_stim_backend_batched_workflow():
    """StimBackend().batched() buffers the circuit and populates the result after dispatch()."""
    qv = QuantumVariable(2)
    qv[:] = "10"
    bb = StimBackend().batched()
    res = qv.get_measurement(backend=bb)
    assert isinstance(res, LazyDict)
    assert not res._populated
    bb.dispatch()
    assert res == {"10": 1.0}
