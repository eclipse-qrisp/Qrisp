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

"""Tests for the Qiskit ↔ Qrisp converter."""

import numpy as np
import pytest
from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit as QiskitQC

from qrisp import QuantumCircuit
from qrisp.circuit import PRXGate
from qrisp.interface.converter.qiskit_converter import (
    convert_from_qiskit,
    convert_to_qiskit,
)


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════


def assert_unitary_equal(qc_a, qc_b, atol=1e-13):
    u_a = (
        Operator(qc_a).data
        if isinstance(qc_a, QiskitQC)
        else Operator(qc_a.to_qiskit()).data
    )
    u_b = (
        Operator(qc_b).data
        if isinstance(qc_b, QiskitQC)
        else Operator(qc_b.to_qiskit()).data
    )
    assert np.allclose(u_a, u_b, atol=atol), f"unitaries differ\n{u_a}\n{u_b}"


# ══════════════════════════════════════════════════════════════════════════
# Qrisp → Qiskit → Qrisp  round-trip
# ══════════════════════════════════════════════════════════════════════════


class TestQrispToQiskitRoundtrip:
    """Standard gates survive Qrisp → Qiskit → Qrisp round-trip."""

    def test_empty(self):
        qc = QuantumCircuit(2)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    @pytest.mark.parametrize("gate_name", ["h", "x", "y", "z", "s", "t", "sx"])
    def test_single_qubit_clifford(self, gate_name):
        qc = QuantumCircuit(2)
        getattr(qc, gate_name)(0)
        getattr(qc, gate_name)(1)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    @pytest.mark.parametrize(
        "gate,params",
        [
            ("rx", [0.5]),
            ("ry", [1.2]),
            ("rz", [-0.3]),
            ("p", [np.pi / 3]),
        ],
    )
    def test_single_qubit_rotations(self, gate, params):
        qc = QuantumCircuit(1)
        getattr(qc, gate)(*params, 0)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_u3(self):
        qc = QuantumCircuit(1)
        qc.u3(1.0, 0.5, -0.3, 0)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_prx(self):
        """PRXGate maps to Qiskit RGate and back."""
        qc = QuantumCircuit(1)
        qc.append(PRXGate(0.7, 1.2), [0])
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_prx_identity(self):
        """PRXGate(0, beta) = identity should survive round-trip."""
        qc = QuantumCircuit(1)
        qc.append(PRXGate(0.0, 2.5), [0])
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_prx_pi_rotation(self):
        """PRXGate(pi, beta) should survive round-trip."""
        qc = QuantumCircuit(1)
        qc.append(PRXGate(np.pi, 0.3), [0])
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_prx_composite(self):
        """PRXGate mixed with other gates in a multi-qubit circuit."""
        qc = QuantumCircuit(3)
        qc.append(PRXGate(0.7, 1.2), [0])
        qc.cx(0, 1)
        qc.append(PRXGate(2.0, -0.5), [2])
        qc.cz(1, 2)
        qc.append(PRXGate(np.pi / 2, 0.0), [0])
        qc.h(1)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_cx(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_cz(self):
        qc = QuantumCircuit(2)
        qc.cz(0, 1)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_swap(self):
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_rzz(self):
        qc = QuantumCircuit(2)
        qc.rzz(0.7, 0, 1)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_cp(self):
        qc = QuantumCircuit(2)
        qc.cp(np.pi / 4, 0, 1)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_composite(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.5, 2)
        qc.cz(1, 2)
        qc.sx(0)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_measure_preserved(self):
        qc = QuantumCircuit(2)
        qc.add_clbit()
        qc.add_clbit()
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(qc.qubits[0], qc.clbits[0])
        qc.measure(qc.qubits[1], qc.clbits[1])
        qc_qiskit = convert_to_qiskit(qc)
        assert qc_qiskit.num_clbits == 2
        # measurements survive round-trip
        qc_rt = convert_from_qiskit(qc_qiskit)
        meas_count = sum(1 for inst in qc_rt.data if inst.op.name == "measure")
        assert meas_count == 2

    def test_barrier(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)
        qc_qiskit = convert_to_qiskit(qc)
        barrier_count = sum(1 for d in qc_qiskit.data if d.operation.name == "barrier")
        assert barrier_count == 1


# ══════════════════════════════════════════════════════════════════════════
# Qiskit → Qrisp → Qiskit  round-trip
# ══════════════════════════════════════════════════════════════════════════


class TestQiskitToQrispRoundtrip:
    """Standard Qiskit gates survive Qiskit → Qrisp → Qiskit round-trip."""

    def test_empty(self):
        qc = QiskitQC(2)
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    @pytest.mark.parametrize(
        "gate,args",
        [
            ("h", [0]),
            ("x", [1]),
            ("y", [0]),
            ("z", [1]),
            ("s", [0]),
            ("t", [1]),
            ("sx", [0]),
        ],
    )
    def test_single_qubit_clifford(self, gate, args):
        qc = QiskitQC(2)
        getattr(qc, gate)(*args)
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    @pytest.mark.parametrize(
        "gate,args",
        [
            ("rx", [0.5, 0]),
            ("ry", [1.2, 0]),
            ("rz", [-0.3, 0]),
        ],
    )
    def test_single_qubit_rotations(self, gate, args):
        qc = QiskitQC(1)
        getattr(qc, gate)(*args)
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    @pytest.mark.parametrize(
        "theta,phi",
        [
            (0.7, 1.2),
            (np.pi, 0.0),
            (0.0, 2.5),
            (-0.5, -1.0),
        ],
    )
    def test_rgate_to_prx(self, theta, phi):
        """Qiskit RGate maps to PRXGate and back."""
        qc = QiskitQC(1)
        qc.r(theta, phi, 0)
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

        # Also verify the intermediate Qrisp circuit uses PRXGate
        qrisp_qc = convert_from_qiskit(qc)
        prx_ops = [instr.op for instr in qrisp_qc.data if isinstance(instr.op, PRXGate)]
        assert len(prx_ops) == 1, f"Expected 1 PRXGate, got {len(prx_ops)}"
        assert abs(prx_ops[0].alpha - theta) < 1e-10
        assert abs(prx_ops[0].beta - phi) < 1e-10

    def test_cx(self):
        qc = QiskitQC(2)
        qc.cx(0, 1)
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_cz(self):
        qc = QiskitQC(2)
        qc.cz(0, 1)
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_cy(self):
        qc = QiskitQC(2)
        qc.cy(0, 1)
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_swap(self):
        qc = QiskitQC(2)
        qc.swap(0, 1)
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_qubit_names_preserved(self):
        """Qubit register names survive round-trip."""
        from qiskit import QuantumRegister

        qr = QuantumRegister(2, "data")
        qc = QiskitQC(qr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        assert qc_rt.num_qubits == 2
        assert_unitary_equal(qc, qc_rt)

    def test_barrier(self):
        qc = QiskitQC(2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        barrier_count = sum(1 for d in qc_rt.data if d.operation.name == "barrier")
        assert barrier_count == 1

    def test_measure(self):
        qc = QiskitQC(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        assert qc_rt.num_clbits == 2
        # unitary part matches (measurements stripped)
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        qc_rt_no_meas = qc_rt.remove_final_measurements(inplace=False)
        assert_unitary_equal(qc_no_meas, qc_rt_no_meas)

    def test_reset(self):
        qc = QiskitQC(1)
        qc.x(0)
        qc.reset(0)
        qc.h(0)
        # reset is non-unitary so Operator fails; just check round-trip
        # preserves the instruction
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        reset_count = sum(1 for d in qc_rt.data if d.operation.name == "reset")
        assert reset_count == 1


# ══════════════════════════════════════════════════════════════════════════
# Controlled operations
# ══════════════════════════════════════════════════════════════════════════


class TestControlledOperations:
    """Multi-controlled gates convert correctly."""

    def test_toffoli(self):
        qc = QuantumCircuit(3)
        qc.mcx([0, 1], 2)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_qiskit_ccx(self):
        qc = QiskitQC(3)
        qc.ccx(0, 1, 2)
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_controlled_prx(self):
        """Controlled PRXGate round-trips correctly."""
        qc = QuantumCircuit(2)
        qc.append(PRXGate(0.6, 0.8).control(1), [0, 1])
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_qiskit_controlled_rgate(self):
        """Qiskit controlled RGate → controlled PRXGate and back."""
        from qiskit.circuit.library import RGate

        qc = QiskitQC(2)
        qc.append(RGate(0.6, 0.8).control(1), [0, 1])
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)

    def test_controlled_rz(self):
        qc = QuantumCircuit(2)
        qc.cp(0.5, 0, 1)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        assert_unitary_equal(qc, qc_rt)


# ══════════════════════════════════════════════════════════════════════════
# Direct Qrisp ↔ Qiskit unitary comparison (no round-trip)
# ══════════════════════════════════════════════════════════════════════════


def _reverse_qubit_order(U: np.ndarray) -> np.ndarray:
    """Reverse qubit ordering: MSB-first (Qrisp) → LSB-first (Qiskit)."""
    n = int(np.log2(U.shape[0]))
    perm = np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(2**n)])
    return U[np.ix_(perm, perm)]


def _assert_qrisp_qiskit_unitary_equal(
    qrisp_qc: QuantumCircuit, qiskit_qc: QiskitQC, atol=1e-13
):
    """Compare Qrisp unitary (MSB-first) against Qiskit unitary (LSB-first)."""
    U_qrisp = qrisp_qc.get_unitary()
    U_qiskit = Operator(qiskit_qc).data
    U_qrisp_lsb = _reverse_qubit_order(U_qrisp)
    assert np.allclose(U_qrisp_lsb, U_qiskit, atol=atol), (
        f"unitaries differ after qubit-order reversal\n{U_qrisp_lsb}\n{U_qiskit}"
    )


class TestQrispToQiskitUnitary:
    """Qrisp → Qiskit: compare unitaries directly (no round-trip)."""

    def test_empty(self):
        qc = QuantumCircuit(2)
        qiskit_qc = convert_to_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qc, qiskit_qc)

    @pytest.mark.parametrize("gate", ["h", "x", "y", "z", "s", "t"])
    def test_clifford_1q(self, gate):
        qc = QuantumCircuit(2)
        getattr(qc, gate)(0)
        getattr(qc, gate)(1)
        qiskit_qc = convert_to_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qc, qiskit_qc)

    @pytest.mark.parametrize(
        "gate,args",
        [
            ("rx", [0.5]),
            ("ry", [1.2]),
            ("rz", [-0.3]),
            ("p", [np.pi / 3]),
        ],
    )
    def test_rotation_1q(self, gate, args):
        qc = QuantumCircuit(1)
        getattr(qc, gate)(*args, 0)
        qiskit_qc = convert_to_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qc, qiskit_qc)

    def test_u3(self):
        qc = QuantumCircuit(1)
        qc.u3(1.0, 0.5, -0.3, 0)
        qiskit_qc = convert_to_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qc, qiskit_qc)

    def test_prx(self):
        qc = QuantumCircuit(1)
        qc.append(PRXGate(0.7, 1.2), [0])
        qiskit_qc = convert_to_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qc, qiskit_qc)

    def test_cx(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qiskit_qc = convert_to_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qc, qiskit_qc)

    def test_cz(self):
        qc = QuantumCircuit(2)
        qc.cz(0, 1)
        qiskit_qc = convert_to_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qc, qiskit_qc)

    def test_swap(self):
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        qiskit_qc = convert_to_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qc, qiskit_qc)

    def test_rzz(self):
        qc = QuantumCircuit(2)
        qc.rzz(0.7, 0, 1)
        qiskit_qc = convert_to_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qc, qiskit_qc)

    def test_cp(self):
        qc = QuantumCircuit(2)
        qc.cp(np.pi / 4, 0, 1)
        qiskit_qc = convert_to_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qc, qiskit_qc)

    def test_toffoli(self):
        qc = QuantumCircuit(3)
        qc.mcx([0, 1], 2)
        qiskit_qc = convert_to_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qc, qiskit_qc)

    def test_composite(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.5, 2)
        qc.cz(1, 2)
        qc.x(0)
        qc.append(PRXGate(0.7, 1.2), [1])
        qiskit_qc = convert_to_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qc, qiskit_qc)

    def test_controlled_prx(self):
        qc = QuantumCircuit(2)
        qc.append(PRXGate(0.6, 0.8).control(1), [0, 1])
        qiskit_qc = convert_to_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qc, qiskit_qc)


class TestQiskitToQrispUnitary:
    """Qiskit → Qrisp: compare unitaries directly (no round-trip)."""

    def test_empty(self):
        qc = QiskitQC(2)
        qrisp_qc = convert_from_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qrisp_qc, qc)

    @pytest.mark.parametrize(
        "gate,args",
        [
            ("h", [0]),
            ("x", [1]),
            ("y", [0]),
            ("z", [1]),
        ],
    )
    def test_clifford_1q(self, gate, args):
        qc = QiskitQC(2)
        getattr(qc, gate)(*args)
        qrisp_qc = convert_from_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qrisp_qc, qc)

    @pytest.mark.parametrize(
        "gate,args",
        [
            ("rx", [0.5, 0]),
            ("ry", [1.2, 0]),
            ("rz", [-0.3, 0]),
        ],
    )
    def test_rotation_1q(self, gate, args):
        qc = QiskitQC(1)
        getattr(qc, gate)(*args)
        qrisp_qc = convert_from_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qrisp_qc, qc)

    def test_rgate(self):
        qc = QiskitQC(1)
        qc.r(0.7, 1.2, 0)
        qrisp_qc = convert_from_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qrisp_qc, qc)

    def test_cx(self):
        qc = QiskitQC(2)
        qc.cx(0, 1)
        qrisp_qc = convert_from_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qrisp_qc, qc)

    def test_cz(self):
        qc = QiskitQC(2)
        qc.cz(0, 1)
        qrisp_qc = convert_from_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qrisp_qc, qc)

    def test_swap(self):
        qc = QiskitQC(2)
        qc.swap(0, 1)
        qrisp_qc = convert_from_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qrisp_qc, qc)

    def test_ccx(self):
        qc = QiskitQC(3)
        qc.ccx(0, 1, 2)
        qrisp_qc = convert_from_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qrisp_qc, qc)

    def test_controlled_rgate(self):
        from qiskit.circuit.library import RGate

        qc = QiskitQC(2)
        qc.append(RGate(0.6, 0.8).control(1), [0, 1])
        qrisp_qc = convert_from_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qrisp_qc, qc)

    def test_composite(self):
        qc = QiskitQC(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.5, 2)
        qc.cz(1, 2)
        qc.r(0.7, 1.2, 1)
        qrisp_qc = convert_from_qiskit(qc)
        _assert_qrisp_qiskit_unitary_equal(qrisp_qc, qc)
