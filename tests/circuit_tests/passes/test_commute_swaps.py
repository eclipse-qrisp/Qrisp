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

from qrisp import Clbit, QuantumCircuit
from qrisp.circuit.pass_management.passes.commute_swaps import commute_swaps


class TestCommuteSwapsMeasurement:
    def test_measurement_commuted_past_swap(self):
        qc = QuantumCircuit(2)
        cb = Clbit("c0")
        qc.add_clbit(cb)
        qc.h(0)
        qc.measure(0, cb)
        qc.swap(0, 1)

        result = commute_swaps(qc)

        assert result.num_qubits() == 2
        measures = [i for i in result.data if i.op.name == "measure"]
        assert len(measures) == 1
        assert result.qubits.index(measures[0].qubits[0]) == 1
        assert result.data[-1].op.name == "measure"

    def test_single_qubit_gate_commuted_past_swap(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.swap(0, 1)

        result = commute_swaps(qc)

        x_gates = [i for i in result.data if i.op.name == "x"]
        assert len(x_gates) == 1
        assert result.qubits.index(x_gates[0].qubits[0]) == 1

    def test_multiple_measurements_commuted(self):
        qc = QuantumCircuit(3)
        cb1, cb2 = Clbit("c0"), Clbit("c1")
        qc.add_clbit(cb1)
        qc.add_clbit(cb2)
        qc.h(0)
        qc.x(1)
        qc.measure(0, cb1)
        qc.measure(1, cb2)
        qc.swap(0, 1)
        qc.h(2)

        result = commute_swaps(qc)

        assert len([i for i in result.data if i.op.name == "h"]) == 2
        assert len([i for i in result.data if i.op.name == "x"]) == 1
        assert len([i for i in result.data if i.op.name == "swap"]) == 1
        assert len([i for i in result.data if i.op.name == "measure"]) == 2
        last_two = [i.op.name for i in result.data[-2:]]
        assert "measure" in last_two


class TestCommuteSwapsNoChange:
    def test_no_single_qubit_before_swap(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.swap(0, 1)
        qc.cx(0, 1)

        original_ops = [i.op.name for i in qc.data]
        result = commute_swaps(qc)
        assert [i.op.name for i in result.data] == original_ops


class TestCommuteSwapsMultipleGates:
    def test_multiple_front_gates_all_commuted(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(0)
        qc.z(0)
        qc.swap(0, 1)

        result = commute_swaps(qc)

        op_names = [i.op.name for i in result.data]
        assert op_names == ["swap", "h", "x", "z"]
        for instr in result.data[1:]:
            assert result.qubits.index(instr.qubits[0]) == 1


# ---------------------------------------------------------------------------
# Correctness: compare_unitary + compare_measurement
# ---------------------------------------------------------------------------


class TestCommuteSwapsCorrectness:
    """Verify commute_swaps preserves the circuit unitary and measurement statistics."""

    def test_unitary_preserved_simple_swap(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        qc.swap(0, 1)
        assert commute_swaps.compare_unitary(qc)

    def test_unitary_preserved_multi_qubit(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.z(2)
        qc.swap(1, 2)
        qc.x(0)
        assert commute_swaps.compare_unitary(qc)

    def test_unitary_preserved_sequential_swaps(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.x(1)
        qc.swap(0, 1)
        qc.z(0)
        qc.swap(1, 2)
        qc.h(2)
        assert commute_swaps.compare_unitary(qc)

    def test_measurement_preserved(self):
        qc = QuantumCircuit(2)
        cb0 = Clbit("c0")
        cb1 = Clbit("c1")
        qc.add_clbit(cb0)
        qc.add_clbit(cb1)
        qc.h(0)
        qc.h(1)
        qc.x(0)
        qc.swap(0, 1)
        qc.measure(qc.qubits, [cb0, cb1])
        assert commute_swaps.compare_measurement(qc)

    def test_measurement_preserved_multiple(self):
        qc = QuantumCircuit(3)
        for _ in range(3):
            qc.add_clbit()
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.z(0)
        qc.x(1)
        qc.swap(0, 1)
        qc.swap(1, 2)
        qc.measure(qc.qubits, qc.clbits)
        assert commute_swaps.compare_measurement(qc)
