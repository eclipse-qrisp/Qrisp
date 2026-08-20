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

from __future__ import annotations

from qrisp.circuit import Instruction
from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.circuit.standard_operations import Barrier, QubitAlloc, QubitDealloc
from qrisp.simulator.preprocessing.measurement_handling import (
    count_measurements_and_treat_alloc,
    extract_measurements,
    insert_multiverse_measurements,
)

# ---------------------------------------------------------------------------
# count_measurements_and_treat_alloc
# ---------------------------------------------------------------------------


class TestCountMeasurementsAndTreatAlloc:
    def test_counts_measurements(self):
        """The returned counter matches the number of measure instructions."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)
        qc.h(1)
        qc.measure(1, 1)
        assert count_measurements_and_treat_alloc(qc) == 2

    def test_removes_barrier_and_alloc(self):
        """Barrier and qb_alloc instructions are dropped entirely."""
        qc = QuantumCircuit(1)
        qc.data.append(Instruction(QubitAlloc(), [qc.qubits[0]]))
        qc.h(0)
        qc.data.append(Instruction(Barrier(1), [qc.qubits[0]]))
        count_measurements_and_treat_alloc(qc)
        names = [instr.op.name for instr in qc.data]
        assert "qb_alloc" not in names
        assert "barrier" not in names
        assert names == ["h"]

    def test_dealloc_replaced_by_disentangler_when_insert_reset(self):
        """With insert_reset=True, qb_dealloc becomes a Disentangler."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.data.append(Instruction(QubitDealloc(), [qc.qubits[0]]))
        count_measurements_and_treat_alloc(qc, insert_reset=True)
        names = [instr.op.name for instr in qc.data]
        assert names == ["h", "disentangle"]

    def test_dealloc_removed_without_replacement_when_no_insert_reset(self):
        """With insert_reset=False, qb_dealloc is simply dropped."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.data.append(Instruction(QubitDealloc(), [qc.qubits[0]]))
        count_measurements_and_treat_alloc(qc, insert_reset=False)
        names = [instr.op.name for instr in qc.data]
        assert names == ["h"]


# ---------------------------------------------------------------------------
# extract_measurements
# ---------------------------------------------------------------------------


class TestExtractMeasurements:
    def test_trailing_measurements_are_extracted(self):
        """Measurements whose qubit/clbit are never used again are extracted."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)
        qc.h(1)
        qc.measure(1, 1)
        new_qc, mes_list = extract_measurements(qc)
        assert [instr.op.name for instr in new_qc.data] == ["h", "h"]
        assert len(mes_list) == 2

    def test_reused_qubit_measurement_is_not_extracted(self):
        """A measurement followed by further use of the same qubit stays in place."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.x(0)
        new_qc, mes_list = extract_measurements(qc)
        assert [instr.op.name for instr in new_qc.data] == ["h", "measure", "x"]
        assert len(mes_list) == 0

    def test_no_measurements(self):
        """A circuit without measurements returns an empty measurement list."""
        qc = QuantumCircuit(1)
        qc.h(0)
        new_qc, mes_list = extract_measurements(qc)
        assert [instr.op.name for instr in new_qc.data] == ["h"]
        assert mes_list == []


# ---------------------------------------------------------------------------
# insert_multiverse_measurements
# ---------------------------------------------------------------------------


class TestInsertMultiverseMeasurements:
    def test_mid_circuit_measurement_is_deferred_to_an_ancilla(self):
        """A measurement reused later is replaced by a CX to a fresh ancilla,
        with the actual measurement deferred to the returned list."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.x(0)

        result_qc, measurements = insert_multiverse_measurements(qc)

        names = [instr.op.name for instr in result_qc.data]
        assert "measure" not in names
        assert "cx" in names
        assert len(result_qc.qubits) == 2  # one ancilla added
        assert len(measurements) == 1
        assert measurements[0].op.name == "measure"

    def test_trailing_measurement_becomes_disentangler(self):
        """A measurement with no further use of its qubit/clbit is just disentangled."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)

        result_qc, measurements = insert_multiverse_measurements(qc)

        names = [instr.op.name for instr in result_qc.data]
        assert names == ["h", "disentangle"]
        assert len(measurements) == 1
