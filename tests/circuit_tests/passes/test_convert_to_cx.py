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

import pytest
from qrisp import QuantumCircuit, Qubit
from qrisp.circuit.pass_management.passes.convert_to_cx import convert_to_cx


def _make_circuit(n: int = 2):
    qc = QuantumCircuit()
    qubits = [Qubit(f"q{i}") for i in range(n)]
    for q in qubits:
        qc.add_qubit(q)
    return qc, qubits


class TestConvertToCXDecompositions:
    def test_cz_converts_to_h_cx_h(self):
        qc, qubits = _make_circuit(2)
        qc.cz(qubits[0], qubits[1])
        result = convert_to_cx()(qc)
        cx = [i for i in result.data if i.op.name == "cx"]
        h = [i for i in result.data if i.op.name == "h"]
        assert len(cx) == 1
        assert len(h) == 2

    def test_cy_converts_to_sdg_cx_s(self):
        qc, qubits = _make_circuit(2)
        qc.cy(qubits[0], qubits[1])
        result = convert_to_cx()(qc)
        cx = [i for i in result.data if i.op.name == "cx"]
        s = [i for i in result.data if i.op.name in ("s", "s_dg")]
        assert len(cx) == 1
        assert len(s) == 2

    def test_swap_converts_to_three_cx_gates(self):
        qc, qubits = _make_circuit(2)
        qc.swap(qubits[0], qubits[1])
        result = convert_to_cx()(qc)
        cx = [i for i in result.data if i.op.name == "cx"]
        assert len(cx) == 3


class TestConvertToCXPreservation:
    def test_cx_unchanged(self):
        qc, qubits = _make_circuit(2)
        qc.cx(qubits[0], qubits[1])
        result = convert_to_cx()(qc)
        assert len(result.data) == 1
        assert result.data[0].op.name == "cx"

    def test_single_qubit_gates_unchanged(self):
        qc, qubits = _make_circuit(2)
        qc.h(qubits[0])
        qc.x(qubits[1])
        result = convert_to_cx()(qc)
        assert len(result.data) == 2
        names = {i.op.name for i in result.data}
        assert names == {"h", "x"}

    def test_empty_circuit(self):
        qc, _ = _make_circuit(2)
        result = convert_to_cx()(qc)
        assert len(result.data) == 0
        assert result.num_qubits() == 2

    def test_unknown_two_qubit_gate_passes_through_by_default(self):
        from qrisp.circuit import Operation

        qc, qubits = _make_circuit(2)
        op = Operation(name="mystery_gate", num_qubits=2)
        qc.append(op, qubits)
        result = convert_to_cx()(qc)
        assert len(result.data) == 1
        assert result.data[0].op.name == "mystery_gate"


class TestConvertToCXStrict:
    def test_unknown_two_qubit_gate_raises_in_strict_mode(self):
        from qrisp.circuit import Operation

        qc, qubits = _make_circuit(2)
        op = Operation(name="mystery_gate", num_qubits=2)
        qc.append(op, qubits)
        with pytest.raises(Exception, match="mystery_gate"):
            convert_to_cx(strict=True)(qc)

    def test_known_gates_work_in_strict_mode(self):
        qc, qubits = _make_circuit(2)
        qc.cz(qubits[0], qubits[1])
        result = convert_to_cx(strict=True)(qc)
        cx = [i for i in result.data if i.op.name == "cx"]
        assert len(cx) == 1
