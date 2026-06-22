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

import numpy as np

from qrisp import QuantumCircuit, Qubit
from qrisp.circuit import XGate
from qrisp.circuit.pass_management.passes.gray_synth_toffoli import (
    _get_gray_toffoli_qc,
    gray_synth_toffoli,
    is_toffoli,
)
from qrisp.circuit.standard_operations import MCXGate

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_circuit(n: int = 3):
    qc = QuantumCircuit()
    qubits = [Qubit(f"q{i}") for i in range(n)]
    for q in qubits:
        qc.add_qubit(q)
    return qc, qubits


def _gate_names(qc) -> list[str]:
    return [instr.op.name for instr in qc.data]


# ------------------------------------------------------------------
# Tests for is_toffoli helper
# ------------------------------------------------------------------


class TestIsToffoli:
    def test_toffoli_detected(self):
        op = MCXGate(control_amount=2)
        assert is_toffoli(op)

    def test_single_controlled_x_rejected(self):
        op = MCXGate(control_amount=1)
        assert not is_toffoli(op)

    def test_triple_controlled_x_rejected(self):
        op = MCXGate(control_amount=3)
        assert not is_toffoli(op)

    def test_controlled_h_rejected(self):
        from qrisp.circuit.standard_operations import HGate

        op = HGate().control(2)
        assert not is_toffoli(op)

    def test_plain_x_rejected(self):
        op = XGate()
        assert not is_toffoli(op)


# ------------------------------------------------------------------
# Tests for the pre-built gray-synthesis Toffoli circuit
# ------------------------------------------------------------------


class TestGrayToffoliCircuit:
    def test_qubit_count(self):
        assert _get_gray_toffoli_qc().num_qubits() == 3

    def test_unitary_matches_toffoli(self):
        toffoli_qc = QuantumCircuit(3)
        toffoli_qc.ccx(toffoli_qc.qubits[0], toffoli_qc.qubits[1], toffoli_qc.qubits[2])
        U_gray = _get_gray_toffoli_qc().get_unitary()
        U_tof = toffoli_qc.get_unitary()

        idx = np.argmax(np.abs(U_tof.flatten()) > 1e-10)
        if np.abs(U_gray.flatten()[idx]) > 1e-10:
            phase = U_tof.flatten()[idx] / U_gray.flatten()[idx]
            np.testing.assert_allclose(
                U_gray * phase,
                U_tof,
                atol=1e-7,
                err_msg="Gray Toffoli unitary does not match Toffoli (up to global phase)",
            )


# ------------------------------------------------------------------
# Tests for gray_synth_toffoli pass
# ------------------------------------------------------------------


class TestGraySynthToffoliPass:
    def test_toffoli_is_replaced(self):
        qc, qubits = _make_circuit(3)
        qc.ccx(qubits[0], qubits[1], qubits[2])
        result = gray_synth_toffoli(qc)
        for instr in result.data:
            if is_toffoli(instr.op):
                assert instr.op.definition is not None
                assert instr.op.definition.num_qubits() == 3

    def test_non_toffoli_gates_preserved(self):
        qc, qubits = _make_circuit(3)
        qc.h(qubits[0])
        qc.cx(qubits[0], qubits[1])
        qc.rz(0.5, qubits[2])
        result = gray_synth_toffoli(qc)
        assert _gate_names(result) == ["h", "cx", "rz"]

    def test_mixed_circuit(self):
        qc, qubits = _make_circuit(3)
        qc.h(qubits[0])
        qc.ccx(qubits[0], qubits[1], qubits[2])
        qc.cx(qubits[1], qubits[2])
        qc.ccx(qubits[2], qubits[0], qubits[1])
        result = gray_synth_toffoli(qc)
        toffoli_count = sum(1 for instr in result.data if is_toffoli(instr.op))
        assert toffoli_count == 2
        assert result.data[0].op.name == "h"
        assert result.data[2].op.name == "cx"

    def test_empty_circuit(self):
        qc = QuantumCircuit(3)
        result = gray_synth_toffoli(qc)
        assert len(result.data) == 0

    def test_no_toffoli_circuit(self):
        qc, qubits = _make_circuit(3)
        qc.h(qubits[0])
        qc.cx(qubits[0], qubits[1])
        qc.t(qubits[2])
        result = gray_synth_toffoli(qc)
        assert _gate_names(result) == _gate_names(qc)

    def test_original_not_mutated(self):
        qc, qubits = _make_circuit(3)
        qc.ccx(qubits[0], qubits[1], qubits[2])
        original_op = qc.data[0].op
        original_def = original_op.definition
        _ = gray_synth_toffoli(qc)
        assert qc.data[0].op is original_op
        assert qc.data[0].op.definition is original_def

    def test_qubit_mapping_preserved(self):
        qc, qubits = _make_circuit(4)
        qc.ccx(qubits[1], qubits[3], qubits[2])
        result = gray_synth_toffoli(qc)
        tof_instr = [instr for instr in result.data if is_toffoli(instr.op)][0]
        assert tof_instr.qubits == [qubits[1], qubits[3], qubits[2]]

    def test_unitary_equivalence(self):
        qc, qubits = _make_circuit(3)
        qc.h(qubits[0])
        qc.ccx(qubits[0], qubits[1], qubits[2])
        qc.cx(qubits[1], qubits[2])
        result = gray_synth_toffoli(qc)
        U_orig = qc.get_unitary()
        U_result = result.get_unitary()
        np.testing.assert_allclose(
            U_orig,
            U_result,
            atol=1e-8,
            err_msg="Pass output is not unitarily equivalent to input",
        )

    def test_multiple_toffolis_unitary(self):
        qc, qubits = _make_circuit(3)
        qc.ccx(qubits[0], qubits[1], qubits[2])
        qc.ccx(qubits[2], qubits[0], qubits[1])
        qc.ccx(qubits[1], qubits[2], qubits[0])
        result = gray_synth_toffoli(qc)
        U_orig = qc.get_unitary()
        U_result = result.get_unitary()
        np.testing.assert_allclose(
            U_orig,
            U_result,
            atol=1e-8,
            err_msg="Multiple-Toffoli pass output is not unitarily equivalent",
        )


# ------------------------------------------------------------------
# Correctness: compare_unitary (CircuitPass API)
# ------------------------------------------------------------------


class TestGraySynthToffoliCorrectness:
    """Verify gray_synth_toffoli preserves the circuit unitary using the
    standard compare_unitary API.
    """

    def test_unitary_equivalence_single(self):
        qc, qubits = _make_circuit(3)
        qc.h(qubits[0])
        qc.ccx(qubits[0], qubits[1], qubits[2])
        qc.cx(qubits[1], qubits[2])
        assert gray_synth_toffoli.compare_unitary(qc)

    def test_unitary_equivalence_multiple(self):
        qc, qubits = _make_circuit(3)
        qc.ccx(qubits[0], qubits[1], qubits[2])
        qc.ccx(qubits[2], qubits[0], qubits[1])
        qc.ccx(qubits[1], qubits[2], qubits[0])
        assert gray_synth_toffoli.compare_unitary(qc)

    def test_unitary_equivalence_no_toffolis(self):
        qc, qubits = _make_circuit(3)
        qc.h(qubits[0])
        qc.cx(qubits[0], qubits[1])
        qc.t(qubits[2])
        assert gray_synth_toffoli.compare_unitary(qc)
