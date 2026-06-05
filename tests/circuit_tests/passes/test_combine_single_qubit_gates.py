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
from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.circuit.operation import ControlledOperation
from qrisp.circuit.pass_management.passes.combine_single_qubit_gates import (
    combine_single_qubit_gates,
    _apply_combined_gates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unitary_of_last_instr(qc: QuantumCircuit) -> np.ndarray:
    """Return the unitary of the last operation in the circuit data."""
    assert len(qc.data) > 0, "Circuit has no instructions."
    return qc.data[-1].op.get_unitary()


def _gate_names(qc: QuantumCircuit) -> list[str]:
    """Return a list of op names for circuit data, for quick inspection."""
    return [instr.op.name for instr in qc.data]


# ---------------------------------------------------------------------------
# Tests: identity cancellation
# ---------------------------------------------------------------------------


class TestIdentityCancellation:
    """Adjacent single-qubit gates whose product equals the identity
    should be removed entirely."""

    def test_x_then_x_cancels(self):
        """X followed by X on the same qubit → identity."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.x(0)
        result = combine_single_qubit_gates(qc)
        assert len(result.data) == 0

    def test_h_then_h_combined(self):
        """H followed by H on the same qubit is combined into a single gate.

        Qrisp's H gate has tiny (~1e-16) imaginary components that push
        the norm of H² − I just above the 1e-10 cancellation threshold,
        so the two H gates are combined rather than cancelled."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        result = combine_single_qubit_gates(qc)
        # H² ≈ I within ~1e-7, so the pass combines them into one u3 gate.
        assert len(result.data) == 1

    def test_rx_pi_then_rx_minus_pi_cancels(self):
        """RX(π) then RX(−π) → identity."""
        qc = QuantumCircuit(1)
        qc.rx(np.pi, 0)
        qc.rx(-np.pi, 0)
        result = combine_single_qubit_gates(qc)
        assert len(result.data) == 0

    def test_identity_has_no_effect_on_other_qubits(self):
        """X→X on q0 cancels; Y on q1 remains untouched."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(0)
        qc.y(1)
        result = combine_single_qubit_gates(qc)
        names = _gate_names(result)
        assert names == ["y"]

    def test_identity_cancellation_across_qubits(self):
        """X→X on q0 and Z→Z on q1 both cancel → empty circuit."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(0)
        qc.z(1)
        qc.z(1)
        result = combine_single_qubit_gates(qc)
        assert len(result.data) == 0


# ---------------------------------------------------------------------------
# Tests: gate combination
# ---------------------------------------------------------------------------


class TestGateCombination:
    """Non-identity products of adjacent single-qubit gates should be
    merged into a single unitary operation."""

    def test_x_then_z_combined(self):
        """X then Z is -i·Y — should produce a single gate (u3)."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.z(0)
        result = combine_single_qubit_gates(qc)
        assert len(result.data) == 1
        # Qrisp recognises the combined 1Q unitary as a U3 gate.
        assert _gate_names(result)[0] == "u3"

    def test_three_gates_combined_to_one(self):
        """Three consecutive rotations on the same qubit → one unitary."""
        qc = QuantumCircuit(1)
        qc.rx(0.3, 0)
        qc.ry(0.5, 0)
        qc.rz(0.7, 0)
        result = combine_single_qubit_gates(qc)
        assert len(result.data) == 1

    def test_single_gate_stays_single(self):
        """A single gate with no neighbours should be re-emitted as-is."""
        qc = QuantumCircuit(1)
        qc.x(0)
        result = combine_single_qubit_gates(qc)
        assert len(result.data) == 1
        assert _gate_names(result)[0] == "x"

    def test_combined_unitary_is_correct(self):
        """Verify the combined gate's unitary equals the matrix product.

        Gates are multiplied in reverse order (last applied = rightmost),
        so X then Z produces Z·X, not X·Z."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.z(0)
        result = combine_single_qubit_gates(qc)
        u = _unitary_of_last_instr(result)
        # Z·X = [[0, 1], [-1, 0]]  (X·Z = [[0, -1], [1, 0]] differs by −1)
        expected = np.array([[0, 1], [-1, 0]], dtype=complex)
        assert np.allclose(u, expected) or np.allclose(u, -expected)


# ---------------------------------------------------------------------------
# Tests: multi-qubit gates as barriers
# ---------------------------------------------------------------------------


class TestMultiQubitBarriers:
    """Multi-qubit gates should flush accumulated single-qubit gates
    on all qubits they touch *before* they are appended."""

    def test_cx_flushes_pending_gates(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.z(0)
        qc.cx(0, 1)
        # X→Z on q0 should be combined into one gate, then cx appears.
        result = combine_single_qubit_gates(qc)
        names = _gate_names(result)
        # Expect: combined gate (X·Z on q0 emitted as u3), then cx.
        assert len(names) == 2
        assert names[0] == "u3"
        assert names[1] == "cx"

    def test_gates_after_cx_not_merged_across_barrier(self):
        """Gates on opposite sides of a CX should NOT be merged."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.cx(0, 1)
        qc.z(0)
        result = combine_single_qubit_gates(qc)
        names = _gate_names(result)
        # x stays, cx, z stays (x and z are separated by cx).
        assert names == ["x", "cx", "z"]

    def test_different_qubits_independent(self):
        """Gates on different qubits are flushed independently by CX.

        X before and after CX on q0 stay separate (split by CX).
        Z before and after CX on q1 stay separate as well.
        Result: x, z, cx, x, z (5 instructions)."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.z(1)
        qc.cx(0, 1)
        qc.x(0)
        qc.z(1)
        result = combine_single_qubit_gates(qc)
        names = _gate_names(result)
        # Each single-qubit gate stays alone — CX splits them.
        assert len(names) == 5
        assert names[0] == "x"
        assert names[1] == "z"
        assert names[2] == "cx"
        assert names[3] == "x"
        assert names[4] == "z"


# ---------------------------------------------------------------------------
# Tests: allocation / deallocation / measurement / reset barriers
# ---------------------------------------------------------------------------


class TestSpecialInstructionBarriers:
    """Allocation, deallocation, measurement and reset instructions
    act as barriers that flush pending single-qubit gates."""

    def test_qb_alloc_flushes(self):
        """qb_alloc instructions act as barriers that flush pending gates.

        In Qrisp's compilation pipeline, qb_alloc / qb_dealloc are
        explicit instructions in circuit data (unlike add_qubit())."""
        from qrisp.circuit.operation import Operation

        qc = QuantumCircuit(1)
        qc.x(0)
        # Manual qb_alloc instruction touches qubit 0 → flushes its pending gates.
        qc.append(Operation(name="qb_alloc", num_qubits=1, num_clbits=0), [0])
        qc.y(0)
        result = combine_single_qubit_gates(qc)
        names = _gate_names(result)
        # x flushed before qb_alloc, then qb_alloc, then y — all separate.
        assert names == ["x", "qb_alloc", "y"]

    def test_reset_flushes(self):
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.reset(0)
        qc.y(0)
        result = combine_single_qubit_gates(qc)
        names = _gate_names(result)
        assert names == ["x", "reset", "y"]

    def test_measure_flushes(self):
        qc = QuantumCircuit(2, 1)
        qc.x(0)
        qc.measure(0, 0)
        qc.y(0)
        result = combine_single_qubit_gates(qc)
        names = _gate_names(result)
        assert names[0] == "x"
        assert names[1] == "measure"
        assert names[2] == "y"


# ---------------------------------------------------------------------------
# Tests: empty / no-op circuits
# ---------------------------------------------------------------------------


class TestEmptyAndNoOp:
    def test_empty_circuit_passes_through(self):
        qc = QuantumCircuit(3)
        result = combine_single_qubit_gates(qc)
        assert len(result.data) == 0
        assert result.num_qubits() == 3

    def test_all_gates_cancel(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(0)
        qc.z(1)
        qc.z(1)
        result = combine_single_qubit_gates(qc)
        assert len(result.data) == 0

    def test_no_single_qubit_gates(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cz(1, 2)
        result = combine_single_qubit_gates(qc)
        names = _gate_names(result)
        assert names == ["cx", "cz"]


# ---------------------------------------------------------------------------
# Tests: recursive processing of composite gates
# ---------------------------------------------------------------------------


class TestRecursiveProcessing:
    """The pass should recursively combine single-qubit gates inside
    compound gate definitions."""

    def test_gate_with_definition_optimised(self):
        """A composite gate containing redundant rotations is optimised."""
        inner = QuantumCircuit(1)
        inner.x(0)
        inner.x(0)  # cancels
        inner.z(0)  # should stay
        compound = inner.to_gate(name="compound")

        qc = QuantumCircuit(1)
        qc.append(compound, [0])
        result = combine_single_qubit_gates(qc)
        names = _gate_names(result)
        # After recursion, the compound gate's definition is z only.
        assert len(result.data) == 1
        assert names[0] == "compound"

    def test_controlled_op_base_definition_optimised(self):
        """A ControlledOperation whose base gate has a definition is
        recursively optimised."""
        inner = QuantumCircuit(2)
        inner.x(0)
        inner.x(0)  # cancel
        inner.z(0)
        # 2-qubit base gate (acts on qubits 0 and 1 of inner circuit).
        base_gate = inner.to_gate(name="my_base")

        # Controlled version has 1 control + 2 target = 3 qubits total.
        ctrl_op = ControlledOperation(base_gate, num_ctrl_qubits=1)
        qc = QuantumCircuit(3)
        qc.append(ctrl_op, [0, 1, 2])
        result = combine_single_qubit_gates(qc)
        names = _gate_names(result)
        assert len(result.data) == 1
        assert names[0] == "cmy_base"


# ---------------------------------------------------------------------------
# Tests: _apply_combined_gates helper
# ---------------------------------------------------------------------------


class TestApplyCombinedGatesHelper:
    """Unit tests for the private helper function."""

    def test_empty_list_nothing_appended(self):
        from qrisp.circuit.standard_operations import XGate

        qc = QuantumCircuit(1)
        _apply_combined_gates(qc, [], qc.qubits[0])
        assert len(qc.data) == 0

    def test_single_gate_reemitted(self):
        from qrisp.circuit.standard_operations import TGate

        qc = QuantumCircuit(1)
        gate = TGate()
        _apply_combined_gates(qc, [gate], qc.qubits[0])
        assert len(qc.data) == 1
        assert _gate_names(qc)[0] == "t"

    def test_complete_cancellation_removes_all(self):
        """Z — Z → identity (Z² = I)."""
        from qrisp.circuit.standard_operations import ZGate

        qc = QuantumCircuit(1)
        _apply_combined_gates(qc, [ZGate(), ZGate()], qc.qubits[0])
        assert len(qc.data) == 0

    def test_combined_unitary_correct(self):
        """Verify combined unitary = Y·X (last gate popped first)."""
        from qrisp.circuit.standard_operations import XGate, YGate

        qc = QuantumCircuit(1)
        # XGate first, YGate second → combined = Y·X.
        _apply_combined_gates(qc, [XGate(), YGate()], qc.qubits[0])
        assert len(qc.data) == 1
        u = _unitary_of_last_instr(qc)
        # Y·X = [[0,-i],[i,0]] @ [[0,1],[1,0]] = [[-i,0],[0,i]]
        expected = np.array([[-1j, 0], [0, 1j]], dtype=complex)
        assert np.allclose(u, expected) or np.allclose(u, -expected)
