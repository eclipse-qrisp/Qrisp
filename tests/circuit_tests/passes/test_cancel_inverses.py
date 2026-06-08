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
from qrisp.circuit.pass_management.passes.cancel_inverses import cancel_inverses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gate_names(qc: QuantumCircuit) -> list[str]:
    """Return a list of op names for circuit data."""
    return [instr.op.name for instr in qc.data]


def _num_gates(qc: QuantumCircuit) -> int:
    return len(qc.data)


# ---------------------------------------------------------------------------
# Tests: self-inverse cancellation
# ---------------------------------------------------------------------------


class TestSelfInverseCancellation:
    """Gates that are their own inverse should cancel when adjacent."""

    def test_cx_cx_cancels(self):
        """Two consecutive CX gates cancel."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(0, 1)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 0

    def test_x_x_cancels(self):
        """Two consecutive X gates cancel."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.x(0)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 0

    def test_h_h_cancels(self):
        """Two consecutive H gates cancel."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 0

    def test_z_z_cancels(self):
        """Two consecutive Z gates cancel."""
        qc = QuantumCircuit(1)
        qc.z(0)
        qc.z(0)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 0

    def test_cz_cz_cancels(self):
        """Two consecutive CZ gates cancel."""
        qc = QuantumCircuit(2)
        qc.cz(0, 1)
        qc.cz(0, 1)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 0

    def test_swap_swap_cancels(self):
        """Two consecutive SWAP gates cancel."""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        qc.swap(0, 1)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 0


# ---------------------------------------------------------------------------
# Tests: parameterised gate cancellation
# ---------------------------------------------------------------------------


class TestParameterisedCancellation:
    """Parametrised rotations Rz(θ)·Rz(−θ), Rx(θ)·Rx(−θ), etc."""

    def test_rz_theta_then_rz_minus_theta_cancels(self):
        qc = QuantumCircuit(1)
        qc.rz(np.pi / 3, 0)
        qc.rz(-np.pi / 3, 0)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 0

    def test_rz_cancelation_preserves_gphase(self):
        qc = QuantumCircuit(1)
        qc.rz(np.pi/2, 0)
        qc.rz(-np.pi/4, 0)
        result = cancel_inverses(qc)
        assert result.compare_unitary(qc, ignore_gphase = False)

    def test_controlled_preserves_gphase(self):
        from qrisp import QuantumVariable, control
        from qrisp.core import h, rz, p
        from qrisp.simulator import statevector_sim

        theta = 0.73

        qv = QuantumVariable(2)

        # Put the control qubit into superposition.
        h(qv[0])

        # RZ(theta) P(-theta) = exp(-i theta/2) I
        # That is only a global phase if it is NOT controlled.
        # Under control, it becomes a relative phase on the qv[0] = 1 branch.
        with control(qv[0]):
            rz(theta, qv[1])
            p(-theta, qv[1])

        # Convert the relative phase on the control branch into amplitudes/probabilities.
        h(qv[0])

        raw_qc = qv.qs.copy()
        raw_sv = statevector_sim(raw_qc)

        compiled_qc = qv.qs.compile()

        compiled_sv = compiled_qc.statevector_array()

        # Compare up to one global phase.
        idx = np.argmax(np.abs(raw_sv))
        phase = compiled_sv[idx] / raw_sv[idx]
        phase /= abs(phase)

        print(f"Raw statevector:\n{raw_sv}")
        print(f"Compiled statevector:\n{compiled_sv}")
        print(f"Compiled statevector, phase corrected:\n{compiled_sv / phase}")

        assert np.allclose(compiled_sv, phase * raw_sv, atol=1e-6)

    def test_rx_positive_then_negative_cancels(self):
        qc = QuantumCircuit(1)
        qc.rx(1.2, 0)
        qc.rx(-1.2, 0)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 0

    def test_ry_positive_then_negative_cancels(self):
        qc = QuantumCircuit(1)
        qc.ry(0.7, 0)
        qc.ry(-0.7, 0)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 0

    def test_p_theta_then_p_minus_theta_cancels(self):
        qc = QuantumCircuit(1)
        qc.p(np.pi / 2, 0)
        qc.p(-np.pi / 2, 0)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 0

    def test_rz_plus_p_cross_fusion(self):
        """Rz(θ)·P(−θ) on same qubit cancels into gphase."""
        qc = QuantumCircuit(1)
        qc.rz(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 1

    def test_different_qubits_no_cancel(self):
        """Rz(θ) on q0 and Rz(−θ) on q1 should NOT cancel."""
        qc = QuantumCircuit(2)
        qc.rz(np.pi / 3, 0)
        qc.rz(-np.pi / 3, 1)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 2


# ---------------------------------------------------------------------------
# Tests: gate separation prevents cancellation
# ---------------------------------------------------------------------------


class TestSeparationPreventsCancellation:
    """A gate in between should prevent cancellation."""

    def test_barrier_prevents_cancellation(self):
        """X · barrier · X — the barrier splits the flow."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(0)  # this cancels with the one above
        qc.cx(0, 1)
        qc.x(0)
        qc.barrier()
        qc.x(0)
        result = cancel_inverses(qc)
        # First two X cancel.  The barrier blocks the last two from fusing.
        # So we get: cx, x, barrier, x = 4 instructions.
        names = _gate_names(result)
        assert names.count("x") == 2

    def test_qubit_interaction_splits(self):
        """CX on q0,q1 means q0 gates on either side can't fuse."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.cx(0, 1)
        qc.x(0)
        result = cancel_inverses(qc)
        names = _gate_names(result)
        assert names.count("x") == 2
        assert names.count("cx") == 1

    def test_non_adjacent_not_cancelled(self):
        """X · Z · X — the Z prevents the two X from cancelling."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.z(0)
        qc.x(0)
        result = cancel_inverses(qc)
        names = _gate_names(result)
        assert names == ["x", "z", "x"]


# ---------------------------------------------------------------------------
# Tests: symmetric gate handling (CZ, SWAP, CP, RZZ)
# ---------------------------------------------------------------------------


class TestSymmetricGates:
    """Symmetric two-qubit gates match by qubit set, not order."""

    def test_cz_different_order_cancels(self):
        """CZ(0,1) then CZ(1,0) — same set → cancelled."""
        qc = QuantumCircuit(2)
        qc.cz(0, 1)
        qc.cz(1, 0)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 0

    def test_swap_different_order_cancels(self):
        """SWAP(0,1) then SWAP(1,0) — cancelled."""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        qc.swap(1, 0)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 0

    def test_cp_different_order_cancels(self):
        """Adjacent CP(θ)·CP(−θ) gates are fused or cancelled."""
        qc = QuantumCircuit(2)
        qc.cp(np.pi / 3, 0, 1)
        qc.cp(-np.pi / 3, 0, 1)
        result = cancel_inverses(qc)
        # Qrisp represents CP as a ControlledOperation(PGate),
        # so the pass reduces the pair to at most 1 gate.
        assert _num_gates(result) <= 1


# ---------------------------------------------------------------------------
# Tests: global phase accumulation
# ---------------------------------------------------------------------------


class TestGlobalPhase:
    """Global phase gates and phase tracking in parameterised gates."""

    def test_gphase_tracking(self):
        """Two gphase operations on the same qubit are fused."""
        from qrisp.circuit import GPhaseGate

        qc = QuantumCircuit(1)
        qc.append(GPhaseGate(np.pi), [0])
        qc.append(GPhaseGate(np.pi), [0])
        result = cancel_inverses(qc)
        # They should be fused into a single gphase of 2π.
        names = _gate_names(result)
        assert names.count("gphase") <= 1

    def test_rz_global_phase_tracking(self):
        """Rz(θ)·Rz(−θ) not only cancels but tracks global phase."""
        qc = QuantumCircuit(1)
        qc.rz(np.pi, 0)
        qc.rz(-np.pi, 0)
        result = cancel_inverses(qc)
        # Cancelled → no gates remain (global phase may be dropped).
        assert _num_gates(result) == 0


# ---------------------------------------------------------------------------
# Tests: empty / no-op circuits
# ---------------------------------------------------------------------------


class TestEmptyAndNoOp:
    def test_empty_circuit_passes_through(self):
        qc = QuantumCircuit(3)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 0

    def test_no_cancelable_gates(self):
        """All gates are non-cancellable — circuit unchanged in count."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.rx(0.3, 0)
        qc.rz(0.7, 1)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 3

    def test_single_gate_passes_through(self):
        qc = QuantumCircuit(1)
        qc.x(0)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 1
        assert _gate_names(result) == ["x"]


# ---------------------------------------------------------------------------
# Tests: controlled operation nesting
# ---------------------------------------------------------------------------


class TestControlledOperations:
    """Controlled operations whose bases cancel should cancel themselves."""

    def test_controlled_x_then_controlled_x_cancels(self):
        """C(X) then C(X) — the controlled X bases cancel."""
        from qrisp.circuit.operation import ControlledOperation
        from qrisp.circuit.standard_operations import XGate

        ctrl_x = ControlledOperation(XGate(), num_ctrl_qubits=1)
        qc = QuantumCircuit(2)
        qc.append(ctrl_x, [0, 1])
        qc.append(ctrl_x, [0, 1])
        result = cancel_inverses(qc)
        assert _num_gates(result) == 0

    def test_controlled_x_then_controlled_x_cancels(self):
        """Test non-trivial base operation cancels"""
        qc_a = QuantumCircuit(1)
        qc_a.x(0)
        qc_a.x(0)

        controlled_gate = qc_a.to_gate().control(1)

        qc_b = QuantumCircuit(2)

        qc_b.append(controlled_gate, qc_b.qubits)
        qc_b.append(controlled_gate, qc_b.qubits)

        result = cancel_inverses(qc_b)

        assert _num_gates(result) == 0

    def test_non_trivial_control_state(self):
        """Tests that XGates with differing control state
        are not canceling"""

        from qrisp.circuit import XGate

        qc = QuantumCircuit(2)

        gate_a = XGate().control(ctrl_state = "0")
        gate_b = XGate().control(ctrl_state = "1")

        qc.append(gate_a, qc.qubits)
        qc.append(gate_b, qc.qubits)

        result = cancel_inverses(qc)

        assert _num_gates(result) == 2

# ---------------------------------------------------------------------------
# Tests: circuit identity preservation
# ---------------------------------------------------------------------------


class TestIdentityPreservation:
    """The pass should not change the overall circuit unitary."""

    def test_identity_preserved(self):
        from qrisp.simulator import run

        # Build a non-trivial circuit with cancellable pairs.
        qc = QuantumCircuit(3, 1)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 1)  # cancels
        qc.rx(np.pi / 4, 1)
        qc.rx(-np.pi / 4, 1)  # cancels
        qc.cz(1, 2)
        qc.measure(0, 0)

        qc_opt = cancel_inverses(qc)

        # Both should yield the same measurement distribution.
        counts_orig = run(qc, shots=2000)
        counts_opt = run(qc_opt, shots=2000)

        assert set(counts_orig.keys()) == set(counts_opt.keys())
        for key in counts_orig:
            assert abs(counts_orig[key] - counts_opt[key]) < 150


class TestGidneyLogicalAND:
    """Cancellation behaviour of GidneyLogicalAND compute/uncompute pairs."""

    def test_compute_uncompute_cancels(self):
        """A GidneyLogicalAND compute followed by uncompute must cancel."""
        from qrisp.alg_primitives.mcx_algs.gidney import GidneyLogicalAND

        compute = GidneyLogicalAND(inv=False, ctrl_state="11")
        uncompute = GidneyLogicalAND(inv=True, ctrl_state="11")

        qc = QuantumCircuit(3)
        qc.append(compute, qc.qubits)
        qc.append(uncompute, qc.qubits)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 0

    def test_same_direction_cancels_via_transpile(self):
        ...
        from qrisp.alg_primitives.mcx_algs.gidney import GidneyLogicalAND

        compute = GidneyLogicalAND(inv=False, ctrl_state="11")

        qc = QuantumCircuit(3)
        qc.append(compute, qc.qubits)
        qc.append(compute, qc.qubits)
        result = cancel_inverses(qc)
        assert _num_gates(result) == 0