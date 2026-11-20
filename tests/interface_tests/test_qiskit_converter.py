# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Tests for the Qiskit ↔ Qrisp converter."""

import itertools

import numpy as np
import pytest
from qiskit import QuantumCircuit as QiskitQC
from qiskit.circuit.classical import expr
from qiskit.circuit.library import CUGate, SXGate, U1Gate, UGate
from qiskit.quantum_info import Operator

from qrisp import QuantumCircuit
from qrisp.circuit import PRXGate, U3Gate
from qrisp.interface.converter.qiskit_converter import (
    convert_from_qiskit,
    convert_to_qiskit,
)


def _assert_unitary_equal(
    qc_a: QiskitQC | QuantumCircuit, qc_b: QiskitQC | QuantumCircuit, atol: float = 1e-13
) -> None:
    u_a = Operator(qc_a).data if isinstance(qc_a, QiskitQC) else Operator(qc_a.to_qiskit()).data
    u_b = Operator(qc_b).data if isinstance(qc_b, QiskitQC) else Operator(qc_b.to_qiskit()).data
    assert np.allclose(u_a, u_b, atol=atol), f"unitaries differ\n{u_a}\n{u_b}"


class TestQrispToQiskitRoundtrip:
    """Standard gates survive Qrisp → Qiskit → Qrisp round-trip."""

    def test_empty(self):
        qc = QuantumCircuit(2)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    @pytest.mark.parametrize("gate_name", ["h", "x", "y", "z", "s", "t", "sx"])
    def test_single_qubit_clifford(self, gate_name):
        qc = QuantumCircuit(2)
        getattr(qc, gate_name)(0)
        getattr(qc, gate_name)(1)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

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
        _assert_unitary_equal(qc, qc_rt)

    def test_u3(self):
        qc = QuantumCircuit(1)
        qc.u3(1.0, 0.5, -0.3, 0)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    @pytest.mark.parametrize("num_ctrl_qubits", [0, 1, 2])
    def test_u3_with_global_phase(self, num_ctrl_qubits):
        gate = U3Gate(0.3, 0.5, 0.7, global_phase=1.1)
        if num_ctrl_qubits:
            gate = gate.control(num_ctrl_qubits)
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, qc.qubits)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    def test_prx(self):
        """PRXGate maps to Qiskit RGate and back."""
        qc = QuantumCircuit(1)
        qc.append(PRXGate(0.7, 1.2), [0])
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    def test_prx_identity(self):
        """PRXGate(0, beta) = identity should survive round-trip."""
        qc = QuantumCircuit(1)
        qc.append(PRXGate(0.0, 2.5), [0])
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    def test_prx_pi_rotation(self):
        """PRXGate(pi, beta) should survive round-trip."""
        qc = QuantumCircuit(1)
        qc.append(PRXGate(np.pi, 0.3), [0])
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

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
        _assert_unitary_equal(qc, qc_rt)

    def test_cx(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    def test_cz(self):
        qc = QuantumCircuit(2)
        qc.cz(0, 1)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    def test_swap(self):
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    def test_rzz(self):
        qc = QuantumCircuit(2)
        qc.rzz(0.7, 0, 1)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    def test_cp(self):
        qc = QuantumCircuit(2)
        qc.cp(np.pi / 4, 0, 1)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    def test_composite(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.5, 2)
        qc.cz(1, 2)
        qc.sx(0)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

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


class TestQiskitToQrispRoundtrip:
    """Standard Qiskit gates survive Qiskit → Qrisp → Qiskit round-trip."""

    def test_empty(self):
        qc = QiskitQC(2)
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

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
        _assert_unitary_equal(qc, qc_rt)

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
        _assert_unitary_equal(qc, qc_rt)

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
        _assert_unitary_equal(qc, qc_rt)

        # Also verify the intermediate Qrisp circuit uses PRXGate
        qrisp_qc = convert_from_qiskit(qc)
        prx_ops = [instr.op for instr in qrisp_qc.data if isinstance(instr.op, PRXGate)]
        assert len(prx_ops) == 1, f"Expected 1 PRXGate, got {len(prx_ops)}"
        assert abs(prx_ops[0].alpha - theta) < 1e-10
        assert abs(prx_ops[0].beta - phi) < 1e-10

    @pytest.mark.parametrize("gate", [U1Gate(0.7), SXGate()], ids=["u1", "sx"])
    @pytest.mark.parametrize("num_ctrl_qubits", [0, 1])
    def test_global_phase_survives_transpile(self, gate, num_ctrl_qubits):
        """Qiskit's u1 and sx keep their exact unitary, global phase included, through Qrisp.

        Controlling a gate turns its global phase into a relative phase. The circuit is
        transpiled in Qrisp before converting back, so the conversion goes through the gates'
        definitions rather than relying only on their names.
        """
        if num_ctrl_qubits:
            gate = gate.control(num_ctrl_qubits)
        qc = QiskitQC(2)
        qc.append(gate, list(range(gate.num_qubits)))
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc).transpile())
        _assert_unitary_equal(qc, qc_rt)

    @pytest.mark.parametrize("num_ctrl_qubits,ctrl_state", [(1, 1), (2, 3), (2, 1)])
    @pytest.mark.parametrize("transpile", [False, True], ids=["direct", "transpiled"])
    def test_cu_gamma(self, num_ctrl_qubits, ctrl_state, transpile):
        gate = CUGate(0.3, 0.5, 0.7, 1.1)
        if num_ctrl_qubits > 1:
            gate = gate.control(num_ctrl_qubits - 1)
        gate.ctrl_state = ctrl_state
        qc = QiskitQC(num_ctrl_qubits + 1)
        qc.append(gate, list(range(num_ctrl_qubits + 1)))

        qrisp_qc = convert_from_qiskit(qc)
        if transpile:
            qrisp_qc = qrisp_qc.transpile()
        _assert_unitary_equal(qc, convert_to_qiskit(qrisp_qc))

    @pytest.mark.parametrize("num_ctrl_qubits", [2, 3])
    def test_multi_controlled_ugate_without_gamma(self, num_ctrl_qubits):
        gate = UGate(0.3, 0.5, 0.7).control(num_ctrl_qubits)
        assert len(gate.params) == 3
        qc = QiskitQC(num_ctrl_qubits + 1)
        qc.append(gate, list(range(num_ctrl_qubits + 1)))
        _assert_unitary_equal(qc, convert_to_qiskit(convert_from_qiskit(qc)))

    def test_cx(self):
        qc = QiskitQC(2)
        qc.cx(0, 1)
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    def test_cz(self):
        qc = QiskitQC(2)
        qc.cz(0, 1)
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    def test_cy(self):
        qc = QiskitQC(2)
        qc.cy(0, 1)
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    def test_swap(self):
        qc = QiskitQC(2)
        qc.swap(0, 1)
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    def test_qubit_names_preserved(self):
        """Qubit register names survive round-trip."""
        from qiskit import QuantumRegister

        qr = QuantumRegister(2, "data")
        qc = QiskitQC(qr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        assert qc_rt.num_qubits == 2
        _assert_unitary_equal(qc, qc_rt)

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
        _assert_unitary_equal(qc_no_meas, qc_rt_no_meas)

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


class TestControlledOperations:
    """Multi-controlled gates convert correctly."""

    def test_toffoli(self):
        qc = QuantumCircuit(3)
        qc.mcx([0, 1], 2)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    def test_qiskit_ccx(self):
        qc = QiskitQC(3)
        qc.ccx(0, 1, 2)
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    def test_controlled_prx(self):
        """Controlled PRXGate round-trips correctly."""
        qc = QuantumCircuit(2)
        qc.append(PRXGate(0.6, 0.8).control(1), [0, 1])
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    def test_qiskit_controlled_rgate(self):
        """Qiskit controlled RGate → controlled PRXGate and back."""
        from qiskit.circuit.library import RGate

        qc = QiskitQC(2)
        qc.append(RGate(0.6, 0.8).control(1), [0, 1])
        qc_rt = convert_to_qiskit(convert_from_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)

    def test_controlled_rz(self):
        qc = QuantumCircuit(2)
        qc.cp(0.5, 0, 1)
        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        _assert_unitary_equal(qc, qc_rt)


def _reverse_qubit_order(U: np.ndarray) -> np.ndarray:
    """Reverse qubit ordering: MSB-first (Qrisp) → LSB-first (Qiskit)."""
    n = int(np.log2(U.shape[0]))
    perm = np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(2**n)])
    return U[np.ix_(perm, perm)]


def _assert_qrisp_qiskit_unitary_equal(qrisp_qc: QuantumCircuit, qiskit_qc: QiskitQC, atol=1e-13):
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

    @pytest.mark.parametrize("symbolic_alpha", [True, False])
    def test_prx_symbolic(self, symbolic_alpha):
        import sympy

        t = sympy.Symbol("t")
        alpha, beta = (t, 1.2) if symbolic_alpha else (0.7, t)
        qc = QuantumCircuit(1)
        qc.append(PRXGate(alpha, beta), [0])

        qiskit_qc = convert_to_qiskit(qc)
        qiskit_qc = qiskit_qc.assign_parameters({qiskit_qc.parameters[0]: 0.4})
        _assert_qrisp_qiskit_unitary_equal(qc.bind_parameters({t: 0.4}), qiskit_qc)

    def test_gphase_symbolic(self):
        import sympy

        t = sympy.Symbol("t")
        qc = QuantumCircuit(1)
        qc.gphase(t, 0)

        qiskit_qc = convert_to_qiskit(qc)
        qiskit_qc = qiskit_qc.assign_parameters({qiskit_qc.parameters[0]: 0.4})
        _assert_qrisp_qiskit_unitary_equal(qc.bind_parameters({t: 0.4}), qiskit_qc)

    def test_controlled_gphase(self):
        from qrisp.circuit.standard_operations import GPhaseGate

        qc = QuantumCircuit(2)
        qc.append(GPhaseGate(0.3).control(1), [0, 1])

        _assert_qrisp_qiskit_unitary_equal(qc, convert_to_qiskit(qc))

    @pytest.mark.parametrize("controlled", [False, True])
    def test_symbol_shared_with_definition(self, controlled):
        import sympy

        t = sympy.Symbol("t")
        sub = QuantumCircuit(1)
        sub.rx(t, 0)
        gate = sub.to_gate("g")

        # The same symbol appears inside the gate's definition and at the top
        # level, so both must become one Qiskit Parameter.
        qc = QuantumCircuit(2)
        if controlled:
            qc.append(gate.control(1), [0, 1])
        else:
            qc.append(gate, [1])
        qc.rz(t, 1)

        qiskit_qc = convert_to_qiskit(qc)
        assert len(qiskit_qc.parameters) == 1
        qiskit_qc = qiskit_qc.assign_parameters({qiskit_qc.parameters[0]: 0.4})
        _assert_qrisp_qiskit_unitary_equal(qc.bind_parameters({t: 0.4}), qiskit_qc)

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

    def test_unconvertible_symbolic_param_raises(self):
        from sympy import Symbol, sqrt

        qc = QuantumCircuit(1)
        qc.rx(sqrt(Symbol("theta")), 0)
        with pytest.raises(NotImplementedError, match="sqrt"):
            convert_to_qiskit(qc)

    def test_distinct_symbols_convert(self):
        import sympy

        a, b = sympy.symbols("a b")
        # Each expression must be evaluated on its own free symbols only, not
        # on every Parameter created so far, and in the matching order.
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.ry(b, 0)
        qc.rz(a - 2 * b, 0)

        qiskit_qc = convert_to_qiskit(qc)
        values = {"a": 0.3, "b": 1.1}
        qiskit_qc = qiskit_qc.assign_parameters({p: values[p.name] for p in qiskit_qc.parameters})
        _assert_qrisp_qiskit_unitary_equal(qc.bind_parameters({a: 0.3, b: 1.1}), qiskit_qc)


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


class TestQiskitControlFlow:
    """Qiskit control-flow operations are either converted faithfully or rejected."""

    @pytest.mark.parametrize("measured, expected", [(0, "10"), (1, "01")])
    def test_if_else_runs_the_matching_branch(self, measured, expected):
        from qiskit.circuit import IfElseOp

        true_body = QiskitQC(1)

        false_body = QiskitQC(1)
        false_body.x(0)

        qc = QiskitQC(2, 2)
        if measured:
            qc.x(0)
        qc.measure(0, 0)
        qc.append(IfElseOp((qc.clbits[0], 1), true_body, false_body), [1], [])
        qc.measure(1, 1)

        assert convert_from_qiskit(qc).run(shots=100) == {expected: 100}

    @pytest.mark.parametrize("measured, expected", [(0, "00"), (1, "11")])
    def test_if_without_else_keeps_condition(self, measured, expected):
        from qiskit.circuit import IfElseOp

        true_body = QiskitQC(1)
        true_body.x(0)

        qc = QiskitQC(2, 2)
        if measured:
            qc.x(0)
        qc.measure(0, 0)
        qc.append(IfElseOp((qc.clbits[0], 1), true_body), [1], [])
        qc.measure(1, 1)

        assert convert_from_qiskit(qc).run(shots=100) == {expected: 100}

    @pytest.mark.parametrize("measured, expected", [(0, "10"), (1, "01")])
    def test_if_else_from_if_test_builder(self, measured, expected):
        # The builder adds the condition clbit to both branch bodies.
        qc = QiskitQC(2, 2)
        if measured:
            qc.x(0)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], 1)) as else_:
            pass
        with else_:
            qc.x(1)
        qc.measure(1, 1)

        assert convert_from_qiskit(qc).run(shots=100) == {expected: 100}

    def test_if_body_writing_other_clbit(self):
        # The body measures into a clbit other than its condition bit. Qrisp's simulator
        # does not count measurements inside classically controlled operations, so the
        # converted structure is checked instead of the counts.
        qc = QiskitQC(2, 2)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], 1)):
            qc.measure(1, 1)

        qrisp_qc = convert_from_qiskit(qc)
        conditioned = qrisp_qc.data[-1]
        assert conditioned.clbits == qrisp_qc.clbits
        assert conditioned.op.base_op.num_clbits == 1

    def test_if_else_on_multiple_clbits_raises(self):
        from qiskit import ClassicalRegister, QuantumRegister
        from qiskit.circuit import IfElseOp

        creg = ClassicalRegister(2)
        qc = QiskitQC(QuantumRegister(1), creg)
        qc.append(IfElseOp((creg, 1), QiskitQC(1), QiskitQC(1)), [0], [])

        with pytest.raises(NotImplementedError):
            convert_from_qiskit(qc)

    def test_while_loop_raises(self):
        from qiskit.circuit import WhileLoopOp

        qc = QiskitQC(1, 1)
        qc.append(WhileLoopOp((qc.clbits[0], 1), QiskitQC(1)), [0], [])

        with pytest.raises(NotImplementedError):
            convert_from_qiskit(qc)


class TestQrispClassicalControl:
    """Classically controlled operations keep their control state in Qiskit."""

    @pytest.mark.parametrize(
        "ctrl_state, bits",
        [(state, bits) for state in ("0", "1", "01") for bits in itertools.product((0, 1), repeat=len(state))],
    )
    def test_condition_follows_ctrl_state(self, ctrl_state, bits):
        from qrisp.circuit import ClControlledOperation
        from qrisp.circuit.standard_operations import XGate

        qiskit_aer = pytest.importorskip("qiskit_aer")

        n = len(ctrl_state)
        qc = QuantumCircuit(n + 1, n + 1)
        for i, b in enumerate(bits):
            if b:
                qc.x(i)
            qc.measure(i, i)
        qc.append(ClControlledOperation(XGate(), n, ctrl_state=ctrl_state), [n], list(range(n)))
        qc.measure(n, n)

        counts = qiskit_aer.AerSimulator().run(convert_to_qiskit(qc), shots=100).result().get_counts()
        # Every clbit is its own register, and Qiskit lists registers last to first.
        flipped = int("".join(map(str, bits)) == ctrl_state)
        assert counts == {" ".join(map(str, [flipped, *reversed(bits)])): 100}

    @pytest.mark.parametrize("ctrl_state", ["0", "1", "01", "10", "110"])
    def test_condition_round_trip(self, ctrl_state):
        from qrisp.circuit import ClControlledOperation
        from qrisp.circuit.standard_operations import XGate

        n = len(ctrl_state)
        qc = QuantumCircuit(1, n)
        qc.append(ClControlledOperation(XGate(), n, ctrl_state=ctrl_state), [0], list(range(n)))

        qc_rt = convert_from_qiskit(convert_to_qiskit(qc))
        (instruction,) = qc_rt.data
        assert isinstance(instruction.op, ClControlledOperation)
        assert instruction.op.ctrl_state == ctrl_state
        assert [cb.identifier for cb in instruction.clbits] == [cb.identifier for cb in qc.clbits]

    @pytest.mark.parametrize(
        "make_condition",
        [
            expr.logic_or,
            lambda a, b: expr.logic_and(a, expr.logic_not(a)),
        ],
        ids=["or", "repeated_clbit"],
    )
    def test_unsupported_condition_raises(self, make_condition):
        qc = QiskitQC(1, 2)
        with qc.if_test(make_condition(*qc.clbits)):
            qc.x(0)

        with pytest.raises(NotImplementedError):
            convert_from_qiskit(qc)

    @pytest.mark.parametrize("ctrl_bit", [0, 1])
    @pytest.mark.parametrize("with_definition", [True, False])
    def test_base_op_clbits_are_kept(self, ctrl_bit, with_definition):
        from qrisp.circuit import ClControlledOperation
        from qrisp.circuit.standard_operations import Measurement

        qiskit_aer = pytest.importorskip("qiskit_aer")

        # The base op writes clbit 1: a measurement of |1>, either bare or
        # inside a definition.
        if with_definition:
            sub = QuantumCircuit(1, 1)
            sub.x(0)
            sub.measure(0, 0)
            base_op = sub.to_op("x_measure")
        else:
            base_op = Measurement()

        qc = QuantumCircuit(2, 2)
        if not with_definition:
            qc.x(0)
        if ctrl_bit:
            qc.x(1)
        qc.measure(1, 0)
        qc.append(ClControlledOperation(base_op, 1, ctrl_state="0"), [0], [0, 1])

        counts = qiskit_aer.AerSimulator().run(convert_to_qiskit(qc), shots=100).result().get_counts()
        assert counts == {f"{1 - ctrl_bit} {ctrl_bit}": 100}

    def test_measuring_q_cond_branches(self):
        from qrisp import QuantumVariable, h, measure, x
        from qrisp.jasp import make_jaspr, q_cond

        qiskit_aer = pytest.importorskip("qiskit_aer")

        # Each branch becomes a ClControlledOperation whose wrapped op measures
        # into a clbit of its own, after the condition clbit.
        def main():
            a = QuantumVariable(1)
            b = QuantumVariable(1)
            h(a[0])
            m = measure(a[0])

            def true_fn():
                x(b[0])
                measure(b[0])

            def false_fn():
                measure(b[0])

            q_cond(m, true_fn, false_fn)

        qc = make_jaspr(main)().to_qc()
        counts = qiskit_aer.AerSimulator().run(convert_to_qiskit(qc), shots=200).result().get_counts()
        # Clbits are (a, false-branch b, true-branch b), listed last to first.
        # a = 0 runs the false branch (b measured as 0), a = 1 the true branch (b = 1).
        assert set(counts) == {"0 0 0", "1 0 1"}

    def test_gphase_in_if_body(self):
        from qrisp.circuit import ClControlledOperation
        from qrisp.circuit.standard_operations import GPhaseGate

        qc = QuantumCircuit(1, 1)
        qc.append(ClControlledOperation(GPhaseGate(0.3)), [0], [0])

        if_op = convert_to_qiskit(qc).data[0].operation
        assert np.allclose(Operator(if_op.blocks[0]).data, np.exp(0.3j) * np.eye(2))

    def test_symbol_shared_with_if_body_definition(self):
        import sympy
        from qrisp.circuit import ClControlledOperation

        t = sympy.Symbol("t")
        sub = QuantumCircuit(1)
        sub.rx(t, 0)

        qc = QuantumCircuit(1, 1)
        qc.append(ClControlledOperation(sub.to_gate("g")), [0], [0])
        qc.rz(t, 0)

        # The if-body and the top-level rz must use the same Parameter.
        qiskit_qc = convert_to_qiskit(qc)
        assert len(qiskit_qc.parameters) == 1
        assert set(qiskit_qc.data[0].operation.blocks[0].parameters) == set(qiskit_qc.parameters)
