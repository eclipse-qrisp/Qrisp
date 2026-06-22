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

import numpy as np
from qrisp import QuantumCircuit, Qubit, U3Gate, convert_to_prx
from qrisp.circuit import PRXGate


def _make_circuit(n=2):
    """Create a circuit with *n* named qubits already added."""
    qc = QuantumCircuit()
    qubits = [Qubit(f"q{i}") for i in range(n)]
    for q in qubits:
        qc.add_qubit(q)
    return qc, qubits


class TestConvertToPRXDecompositions:
    """General U3 → PRX decompositions."""

    def test_general_u3_produces_at_least_one_prx(self):
        """Arbitrary U3 gate must be decomposed into PRX gate(s)."""
        qc, qubits = _make_circuit(2)
        qc.append(U3Gate(np.pi / 2, np.pi / 4, np.pi / 8), qubits[0])
        qc.cz(qubits[0], qubits[1])

        result = convert_to_prx(qc)

        prx_ops = [instr.op for instr in result.data if isinstance(instr.op, PRXGate)]
        assert len(prx_ops) >= 1

    def test_no_leftover_non_prx_u3_gates(self):
        """After conversion, no non-PRX U3 gates (except gphase) remain."""
        qc, qubits = _make_circuit(1)
        qc.append(U3Gate(0.5, 0.3, 0.7), qubits[0])

        result = convert_to_prx(qc)

        for instr in result.data:
            op = instr.op
            if (
                isinstance(op, U3Gate)
                and not isinstance(op, PRXGate)
                and op.name != "gphase"
            ):
                raise AssertionError(f"Leftover non-PRX U3Gate: {op.name}")

    def test_unitary_preserved_after_conversion(self):
        """The unitary must match exactly (global phase is tracked)."""
        qc, qubits = _make_circuit(2)
        qc.append(U3Gate(np.pi / 2, np.pi / 4, np.pi / 8), qubits[0])
        qc.cz(qubits[0], qubits[1])

        result = convert_to_prx(qc)
        assert np.allclose(qc.get_unitary(), result.get_unitary(), atol=1e-6)

    def test_single_prx_case_lam_equals_minus_phi(self):
        """When lam == -phi, a single PRX gate suffices (no gphase)."""
        theta, phi = np.pi / 3, np.pi / 6
        lam = -phi
        qc, qubits = _make_circuit(1)
        qc.append(U3Gate(theta, phi, lam), qubits[0])

        result = convert_to_prx(qc)
        prx_ops = [instr.op for instr in result.data if isinstance(instr.op, PRXGate)]

        assert len(prx_ops) == 1
        assert abs(prx_ops[0].alpha - theta) < 1e-10
        assert abs(prx_ops[0].beta - (phi + np.pi / 2)) < 1e-10

        gphase_ops = [instr.op for instr in result.data if instr.op.name == "gphase"]
        assert len(gphase_ops) == 0


class TestConvertToPRXPreservation:
    """Gates and circuit properties that must survive the pass unchanged."""

    def test_preserves_qubit_count(self):
        qc, _ = _make_circuit(3)
        result = convert_to_prx(qc)
        assert result.num_qubits() == 3

    def test_two_qubit_gates_preserved(self):
        qc, qubits = _make_circuit(2)
        qc.cx(qubits[0], qubits[1])
        qc.h(qubits[0])

        result = convert_to_prx(qc)
        cx_gates = [instr for instr in result.data if instr.op.name == "cx"]
        assert len(cx_gates) == 1

    def test_empty_circuit_stays_empty(self):
        qc, _ = _make_circuit(1)
        result = convert_to_prx(qc)
        assert len(result.data) == 0

    def test_h_gate_converted_to_prx(self):
        """H gate is a U3 gate, so it must be decomposed to PRX."""
        qc, qubits = _make_circuit(1)
        qc.h(qubits[0])

        result = convert_to_prx(qc)
        prx_ops = [instr.op for instr in result.data if isinstance(instr.op, PRXGate)]
        assert len(prx_ops) >= 1


class TestConvertToPRXGlobalPhase:
    """Global phase tracking during decomposition."""

    def test_emits_gphase_for_nonzero_global_phase(self):
        qc, qubits = _make_circuit(1)
        qc.append(U3Gate(0.5, 0.3, 0.7, global_phase=1.2), qubits[0])

        result = convert_to_prx(qc)
        gphase_ops = [instr for instr in result.data if instr.op.name == "gphase"]
        assert len(gphase_ops) == 1
        assert abs(gphase_ops[0].op.params[0]) > 1e-10
        assert np.allclose(qc.get_unitary(), result.get_unitary(), atol=1e-6)

    def test_identity_u3_with_phase_emits_gphase_only(self):
        """theta=0 with global phase: gphase emitted, no PRX gates."""
        qc, qubits = _make_circuit(1)
        qc.append(U3Gate(0, 0.3, -0.3, global_phase=1.5), qubits[0])

        result = convert_to_prx(qc)
        gphase_ops = [instr for instr in result.data if instr.op.name == "gphase"]
        prx_ops = [instr.op for instr in result.data if isinstance(instr.op, PRXGate)]

        assert len(gphase_ops) == 1
        assert len(prx_ops) == 0
        assert np.allclose(qc.get_unitary(), result.get_unitary(), atol=1e-6)


class TestConvertToPRXRandomCircuits:
    """Fuzz-testing: random circuits must preserve unitary equivalence."""

    def test_50_random_circuits_unitary_preserved(self):
        np.random.seed(123)
        for _ in range(50):
            n_qubits = np.random.randint(1, 4)
            qc = QuantumCircuit(n_qubits)

            for _ in range(np.random.randint(1, 10)):
                qb = np.random.randint(0, n_qubits)
                qc.u3(
                    np.random.uniform(0, 2 * np.pi),
                    np.random.uniform(0, 2 * np.pi),
                    np.random.uniform(0, 2 * np.pi),
                    qb,
                )
                if n_qubits >= 2:
                    qc.cz(qb, (qb + 1) % n_qubits)

            result = convert_to_prx(qc)
            assert np.allclose(qc.get_unitary(), result.get_unitary(), atol=1e-6)

    def test_random_circuit_single_qubit_gates_are_prx(self):
        """After conversion, every single-qubit gate must be PRX or gphase."""
        np.random.seed(123)
        for _ in range(25):
            n_qubits = np.random.randint(1, 4)
            qc = QuantumCircuit(n_qubits)
            for _ in range(np.random.randint(1, 6)):
                qb = np.random.randint(0, n_qubits)
                qc.u3(
                    np.random.uniform(0, 2 * np.pi),
                    np.random.uniform(0, 2 * np.pi),
                    np.random.uniform(0, 2 * np.pi),
                    qb,
                )

            result = convert_to_prx(qc)
            for instr in result.data:
                op = instr.op
                if op.name in ("qb_alloc", "qb_dealloc", "barrier"):
                    continue
                if op.num_qubits == 1 and op.num_clbits == 0:
                    assert isinstance(op, PRXGate) or op.name == "gphase", (
                        f"Non-PRX single-qubit gate: {op.name}"
                    )
