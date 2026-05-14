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

from __future__ import annotations

import numpy as np
import pytest
from qrisp import QuantumCircuit, Qubit, U3Gate
from qrisp.circuit import ControlledOperation, Operation, PTControlledOperation
from qrisp.circuit.pass_management.passes.cancel_zero_controls import cancel_zero_controls


def _make_circuit(n: int = 4):
    qc = QuantumCircuit()
    qubits = [Qubit(f"q{i}") for i in range(n)]
    for q in qubits:
        qc.add_qubit(q)
    return qc, qubits


def _gate_names(qc) -> list[str]:
    return [instr.op.name for instr in qc.data]


class TestCancelZeroControlsBasic:
    def test_cp_on_fresh_qubit_cancelled(self):
        qc, qubits = _make_circuit(2)
        qc.h(qubits[0])
        cp = Operation(name="cp", num_qubits=2, params=[np.pi / 4])
        qc.append(cp, [qubits[0], qubits[1]])
        result = cancel_zero_controls(qc)
        assert "cp" not in _gate_names(result)
        assert "h" in _gate_names(result)

    def test_cz_on_fresh_qubit_cancelled(self):
        qc, qubits = _make_circuit(2)
        qc.h(qubits[0])
        qc.cz(qubits[0], qubits[1])
        result = cancel_zero_controls(qc)
        assert "cz" not in _gate_names(result)

    def test_cp_neither_fresh_kept(self):
        qc, qubits = _make_circuit(2)
        qc.h(qubits[0])
        qc.h(qubits[1])
        cp = Operation(name="cp", num_qubits=2, params=[np.pi / 4])
        qc.append(cp, [qubits[0], qubits[1]])
        result = cancel_zero_controls(qc)
        assert "cp" in _gate_names(result)

    def test_cz_neither_fresh_kept(self):
        qc, qubits = _make_circuit(2)
        qc.h(qubits[0])
        qc.h(qubits[1])
        qc.cz(qubits[0], qubits[1])
        result = cancel_zero_controls(qc)
        assert "cz" in _gate_names(result)


class TestDiagonalPreservesFreshness:
    def test_diagonal_preserves_freshness(self):
        qc, qubits = _make_circuit(2)
        qc.s(qubits[0])
        cp = Operation(name="cp", num_qubits=2, params=[np.pi / 4])
        qc.append(cp, [qubits[0], qubits[1]])
        result = cancel_zero_controls(qc)
        assert "cp" not in _gate_names(result)

    def test_h_breaks_freshness_but_target_still_fresh(self):
        qc, qubits = _make_circuit(2)
        qc.h(qubits[0])
        cp = Operation(name="cp", num_qubits=2, params=[np.pi / 4])
        qc.append(cp, [qubits[0], qubits[1]])
        result = cancel_zero_controls(qc)
        # q1 still fresh → CP cancelled
        assert "cp" not in _gate_names(result)

    def test_h_on_both_keeps_cp(self):
        qc, qubits = _make_circuit(2)
        qc.h(qubits[0])
        qc.h(qubits[1])
        cp = Operation(name="cp", num_qubits=2, params=[np.pi / 4])
        qc.append(cp, [qubits[0], qubits[1]])
        result = cancel_zero_controls(qc)
        assert "cp" in _gate_names(result)


class TestControlledOperations:
    def test_controlled_op_ctrl_on_1_fresh_qubit_cancelled(self):
        qc, qubits = _make_circuit(2)
        defn = QuantumCircuit(1)
        defn.h(defn.qubits[0])
        base = Operation(name="my_h", num_qubits=1, definition=defn)
        ctrl_op = ControlledOperation(base, num_ctrl_qubits=1, ctrl_state=1)
        qc.append(ctrl_op, [qubits[0], qubits[1]])
        result = cancel_zero_controls(qc)
        controlled = [
            i for i in result.data
            if isinstance(i.op, ControlledOperation) or i.op.name.endswith("my_h")
        ]
        assert len(controlled) == 0

    def test_controlled_op_ctrl_on_0_fresh_qubit_kept(self):
        qc, qubits = _make_circuit(2)
        defn = QuantumCircuit(1)
        defn.h(defn.qubits[0])
        base = Operation(name="my_h", num_qubits=1, definition=defn)
        ctrl_op = ControlledOperation(base, num_ctrl_qubits=1, ctrl_state=0)
        qc.append(ctrl_op, [qubits[0], qubits[1]])
        result = cancel_zero_controls(qc)
        assert len(result.data) >= 1

    def test_pt_controlled_phase_on_fresh_cancelled(self):
        qc, qubits = _make_circuit(2)
        base = U3Gate(0, 0, np.pi / 4, name="p")
        pt_op = PTControlledOperation(base, num_ctrl_qubits=1)
        qc.append(pt_op, [qubits[0], qubits[1]])
        result = cancel_zero_controls(qc)
        meaningful = [
            i for i in result.data
            if i.op.name not in ("qb_alloc", "qb_dealloc", "barrier")
        ]
        assert len(meaningful) == 0


# ---------------------------------------------------------------------------
# Correctness: compare_measurement (not compare_unitary)
#
# cancel_zero_controls removes gates based on the |0...0⟩ initial state
# assumption.  This preserves measurement statistics (all qubits start in |0⟩)
# but changes the full unitary matrix.  Use compare_measurement for correctness
# verification and compare_unitary only for circuits where the pass is a no-op.
# ---------------------------------------------------------------------------


class TestCancelZeroControlsCorrectness:
    """Verify cancel_zero_controls preserves measurement statistics."""

    def test_measurement_preserved_simple_cp(self):
        """CP on a fresh qubit should be removed; measurement stats preserved."""
        qc = QuantumCircuit(2)
        for _ in range(2):
            qc.add_clbit()
        qc.h(0)
        qc.cp(np.pi / 4, 0, 1)
        qc.measure(qc.qubits, qc.clbits)
        assert cancel_zero_controls.compare_measurement(qc)

    def test_measurement_preserved_mixed(self):
        qc = QuantumCircuit(3)
        for _ in range(3):
            qc.add_clbit()
        qc.h(0)
        qc.cz(0, 1)
        qc.cx(1, 2)
        qc.measure(qc.qubits, qc.clbits)
        assert cancel_zero_controls.compare_measurement(qc)

    def test_unitary_preserved_when_no_fresh_qubits(self):
        """When all qubits are active, no gates are removed → unitary preserved."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.cp(np.pi / 4, 0, 1)
        qc.cz(0, 1)
        assert cancel_zero_controls.compare_unitary(qc)

    def test_unitary_preserved_empty(self):
        qc = QuantumCircuit(3)
        assert cancel_zero_controls.compare_unitary(qc)


class TestAllocDealloc:
    def test_qb_alloc_restores_freshness(self):
        qc, qubits = _make_circuit(2)
        qc.h(qubits[0])
        alloc = Operation(name="qb_alloc", num_qubits=1)
        qc.append(alloc, [qubits[0]])
        cp = Operation(name="cp", num_qubits=2, params=[np.pi / 4])
        qc.append(cp, [qubits[0], qubits[1]])
        result = cancel_zero_controls(qc)
        assert "cp" not in _gate_names(result)

    def test_qb_dealloc_drops_freshness_but_other_fresh(self):
        qc, qubits = _make_circuit(2)
        dealloc = Operation(name="qb_dealloc", num_qubits=1)
        qc.append(dealloc, [qubits[0]])
        cp = Operation(name="cp", num_qubits=2, params=[np.pi / 4])
        qc.append(cp, [qubits[0], qubits[1]])
        result = cancel_zero_controls(qc)
        # q1 still fresh → CP cancelled
        assert "cp" not in _gate_names(result)


class TestQFTStyle:
    def test_qft_all_cp_cancelled(self):
        n = 6
        qc, qubits = _make_circuit(n)
        for i in range(n):
            qc.h(qubits[i])
            for j in range(i + 1, n):
                angle = np.pi / (2 ** (j - i))
                cp = Operation(name="cp", num_qubits=2, params=[angle])
                qc.append(cp, [qubits[i], qubits[j]])
        total_cp = sum(1 for i in qc.data if i.op.name == "cp")
        result = cancel_zero_controls(qc)
        remaining_cp = sum(1 for i in result.data if i.op.name == "cp")
        assert remaining_cp < total_cp
        assert remaining_cp == 0


class TestEdgeCases:
    def test_barriers_preserved(self):
        qc, qubits = _make_circuit(2)
        barrier = Operation(name="barrier", num_qubits=2)
        qc.append(barrier, [qubits[0], qubits[1]])
        qc.h(qubits[0])
        result = cancel_zero_controls(qc)
        assert "barrier" in _gate_names(result)

    def test_empty_circuit(self):
        qc = QuantumCircuit()
        result = cancel_zero_controls(qc)
        assert len(result.data) == 0

    def test_swap_kept_and_breaks_freshness(self):
        qc, qubits = _make_circuit(3)
        qc.swap(qubits[0], qubits[1])
        cp = Operation(name="cp", num_qubits=2, params=[np.pi / 4])
        qc.append(cp, [qubits[0], qubits[1]])
        result = cancel_zero_controls(qc)
        assert "swap" in _gate_names(result)
