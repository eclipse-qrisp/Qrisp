"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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

"""
TODO:

-- Add test with parameters provided the the converter as argument
-- Add test with abstract parameters (sympy.Symbol) in the Qrisp circuit
-- Add test with `measure` operations in the circuit
-- Add test with all the imported gates

"""

# pylint: disable=line-too-long

import numpy as np
import pennylane as qml
import pytest

from qrisp import QuantumVariable
from qrisp.circuit.standard_operations import (
    Barrier,
    CPGate,
    CXGate,
    CYGate,
    CZGate,
    GPhaseGate,
    HGate,
    IDGate,
    MCRXGate,
    MCXGate,
    Measurement,
    PGate,
    QubitAlloc,
    QubitDealloc,
    Reset,
    RGate,
    RXGate,
    RXXGate,
    RYGate,
    RZGate,
    RZZGate,
    SGate,
    SwapGate,
    SXDGGate,
    SXGate,
    TGate,
    U1Gate,
    XGate,
    XXYYGate,
    YGate,
    ZGate,
    u3Gate,
)
from qrisp.interface import qml_converter

SINGLE_GATE_MAP = [
    (XGate, qml.PauliX, []),
    (YGate, qml.PauliY, []),
    (ZGate, qml.PauliZ, []),
    (RXGate, qml.RX, [np.pi / 4]),
    (RYGate, qml.RY, [np.pi / 3]),
    (RZGate, qml.RZ, [np.pi / 2]),
    (PGate, qml.PhaseShift, [np.pi / 2]),
    (SGate, qml.S, []),
    (TGate, qml.T, []),
    (HGate, qml.Hadamard, []),
    (SXGate, qml.SX, []),
    (u3Gate, qml.U3, [np.pi / 2, np.pi / 4, np.pi / 8]),
    (IDGate, qml.Identity, []),
    (GPhaseGate, qml.GlobalPhase, [np.pi / 5]),
    (U1Gate, qml.U1, [np.pi / 6]),
]

MULTI_GATE_MAP = [
    (SwapGate, qml.SWAP, 2, []),
    (RXXGate, qml.IsingXX, 2, [np.pi / 2]),
]

CONTROLLED_GATE_MAP = [
    (CXGate, qml.CNOT, []),
    (CYGate, qml.CY, []),
    (CZGate, qml.CZ, []),
    (CPGate, qml.ControlledPhaseShift, [np.pi / 2]),
]


def _get_qml_operations(qml_circuit):
    """Helper function to extract PennyLane operations from a PennyLane QNode."""
    qml_ops = []
    # pylint: disable=protected-access
    for op in qml_circuit._tape.operations:
        qml_ops.append(op)
    return qml_ops


def _create_qml_circuit(qrisp_circuit, inner_wires=None):
    """
    Helper function to create a PennyLane circuit from a Qrisp circuit.

    This functions uses qml.probs to return the measurement probabilities.

    Probabilities are returned for all qubits in the Qrisp circuit unless inner_wires are provided.
    If inner_wires are provided, probabilities are returned only for those wires.

    """

    pennylane_circuit = qml_converter(qc=qrisp_circuit.qs)
    qrisp_qubits = [qubit.identifier for qubit in qrisp_circuit]

    device_wires = qrisp_qubits
    if inner_wires is not None:
        device_wires += inner_wires

    @qml.qnode(device=qml.device("default.qubit", wires=device_wires))
    def circuit():
        pennylane_circuit(wires=inner_wires)
        return qml.probs(wires=inner_wires if inner_wires is not None else qrisp_qubits)

    return circuit


def check_qml_operations(qml_circuit, expected_ops):
    """Helper function to check if the PennyLane operations match the expected operations."""
    qml_ops = _get_qml_operations(qml_circuit)
    for op, expected_op in zip(qml_ops, expected_ops, strict=True):
        qml.assert_equal(op, expected_op)


def check_probs_equivalence(qrisp_probs, qml_probs, atol=1e-8):
    """Helper function to check if the measurement probabilities from Qrisp and PennyLane are equivalent."""

    qrisp_items = sorted(qrisp_probs.items(), key=lambda kv: int(kv[0], 2))

    for bitstring, q_prob in qrisp_items:
        idx = int(bitstring, 2)

        if idx >= qml_probs.size:
            raise AssertionError(
                f"Index {idx} out of range for qml_probs (size {qml_probs.size})"
            )

        qml_prob = qml_probs[idx]
        assert np.isclose(q_prob, qml_prob, atol=atol), (
            f"Probability mismatch for bitstring {bitstring}: "
            f"Qrisp={q_prob}, PennyLane={qml_prob}"
        )


class TestSingleGateConversion:
    """Test class for single gate conversions from Qrisp to PennyLane."""

    @pytest.mark.parametrize("qrisp_gate_class,qml_gate_class,params", SINGLE_GATE_MAP)
    def test_single_gate_conversion(self, qrisp_gate_class, qml_gate_class, params):
        """Test one-to-one conversion of simple gates from Qrisp to PennyLane."""
        qv = QuantumVariable(1)
        qs = qv.qs
        qs.append(qrisp_gate_class(*params), qv[0])

        qml_circ = _create_qml_circuit(qv)
        qml_res = qml_circ()  # populate tape

        expected_ops = [qml_gate_class(*params, wires=qv[0].identifier)]
        check_qml_operations(qml_circ, expected_ops)

        qrisp_res = qv.get_measurement()
        qml_res = np.asarray(qml_res, dtype=float)
        atol = 1e-5 if params else 1e-8
        check_probs_equivalence(qrisp_res, qml_res, atol=atol)

    def test_inner_wires(self):
        """Test conversion of a circuit with inner wires provided to the converter."""
        qrisp_qv = QuantumVariable(2)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(XGate(), qrisp_qv[0])
        qrisp_qs.append(HGate(), qrisp_qv[1])

        inner_wires = ["aux_0", "aux_1"]
        qml_converted_circuit = _create_qml_circuit(qrisp_qv, inner_wires=inner_wires)

        # We execute the circuit to populate the tape
        qml_res = qml_converted_circuit()

        expected_ops = [
            qml.X(wires="aux_0"),
            qml.Hadamard(wires="aux_1"),
        ]
        check_qml_operations(qml_converted_circuit, expected_ops)

        qrisp_res = qrisp_qv.get_measurement()
        qml_res = np.asarray(qml_res, dtype=float)
        check_probs_equivalence(qrisp_res, qml_res)

    def test_pauli_gates(self):
        """Test conversion of Pauli gates (X, Y, Z) from Qrisp to PennyLane."""
        qrisp_qv = QuantumVariable(3)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(XGate(), qrisp_qv[0])
        qrisp_qs.append(XGate(), qrisp_qv[0])
        qrisp_qs.append(YGate(), qrisp_qv[0])
        qrisp_qs.append(ZGate(), qrisp_qv[0])

        qrisp_qs.append(YGate(), qrisp_qv[1])

        qrisp_qs.append(ZGate(), qrisp_qv[2])
        qrisp_qs.append(XGate(), qrisp_qv[2])

        qrisp_qs.append(YGate(), qrisp_qv[0])
        qrisp_qs.append(ZGate(), qrisp_qv[0])

        qml_converted_circuit = _create_qml_circuit(qrisp_qv)

        # We execute the circuit to populate the tape
        qml_res = qml_converted_circuit()

        expected_ops = [
            qml.X(qrisp_qv[0].identifier),
            qml.X(qrisp_qv[0].identifier),
            qml.Y(qrisp_qv[0].identifier),
            qml.Z(qrisp_qv[0].identifier),
            qml.Y(qrisp_qv[1].identifier),
            qml.Z(qrisp_qv[2].identifier),
            qml.X(qrisp_qv[2].identifier),
            qml.Y(qrisp_qv[0].identifier),
            qml.Z(qrisp_qv[0].identifier),
        ]
        check_qml_operations(qml_converted_circuit, expected_ops)

        qrisp_res = qrisp_qv.get_measurement()
        qml_res = np.asarray(qml_res, dtype=float)
        check_probs_equivalence(qrisp_res, qml_res)

    def test_rotation_gates(self):
        """Test conversion of rotation gates (RX, RY, RZ) from Qrisp to PennyLane."""
        qrisp_qv = QuantumVariable(3)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(RXGate(np.pi / 2), qrisp_qv[0])
        qrisp_qs.append(RYGate(np.pi), qrisp_qv[0])

        qrisp_qs.append(RZGate(np.pi / 4), qrisp_qv[1])
        qrisp_qs.append(RXGate(np.pi / 3), qrisp_qv[1])

        qrisp_qs.append(RYGate(np.pi / 6), qrisp_qv[2])
        qrisp_qs.append(RZGate(np.pi / 2), qrisp_qv[2])
        qrisp_qs.append(RXGate(np.pi), qrisp_qv[2])

        qrisp_qs.append(RZGate(np.pi / 5), qrisp_qv[1])
        qrisp_qs.append(RXGate(np.pi / 7), qrisp_qv[1])

        qrisp_qs.append(RYGate(np.pi / 8), qrisp_qv[0])
        qrisp_qs.append(RZGate(np.pi / 9), qrisp_qv[0])

        qml_converted_circuit = _create_qml_circuit(qrisp_qv)

        # We execute the circuit to populate the tape
        qml_res = qml_converted_circuit()

        expected_ops = [
            qml.RX(np.pi / 2, wires=qrisp_qv[0].identifier),
            qml.RY(np.pi, wires=qrisp_qv[0].identifier),
            qml.RZ(np.pi / 4, wires=qrisp_qv[1].identifier),
            qml.RX(np.pi / 3, wires=qrisp_qv[1].identifier),
            qml.RY(np.pi / 6, wires=qrisp_qv[2].identifier),
            qml.RZ(np.pi / 2, wires=qrisp_qv[2].identifier),
            qml.RX(np.pi, wires=qrisp_qv[2].identifier),
            qml.RZ(np.pi / 5, wires=qrisp_qv[1].identifier),
            qml.RX(np.pi / 7, wires=qrisp_qv[1].identifier),
            qml.RY(np.pi / 8, wires=qrisp_qv[0].identifier),
            qml.RZ(np.pi / 9, wires=qrisp_qv[0].identifier),
        ]
        check_qml_operations(qml_converted_circuit, expected_ops)

        qrisp_res = qrisp_qv.get_measurement()
        qml_res = np.asarray(qml_res, dtype=float)
        check_probs_equivalence(qrisp_res, qml_res, atol=1e-5)

    def test_phase_gates(self):
        """Test conversion of phase gates (P, S, T) from Qrisp to PennyLane."""
        qrisp_qv = QuantumVariable(2)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(PGate(np.pi / 2), qrisp_qv[0])
        qrisp_qs.append(SGate(), qrisp_qv[0])

        qrisp_qs.append(SGate(), qrisp_qv[1])
        qrisp_qs.append(TGate(), qrisp_qv[1])
        qrisp_qs.append(PGate(np.pi / 4), qrisp_qv[1])

        qrisp_qs.append(TGate(), qrisp_qv[0])

        qrisp_qs.append(TGate(), qrisp_qv[1])

        qml_converted_circuit = _create_qml_circuit(qrisp_qv)

        # We execute the circuit to populate the tape
        qml_res = qml_converted_circuit()

        expected_ops = [
            qml.PhaseShift(np.pi / 2, wires=qrisp_qv[0].identifier),
            qml.S(wires=qrisp_qv[0].identifier),
            qml.S(wires=qrisp_qv[1].identifier),
            qml.T(wires=qrisp_qv[1].identifier),
            qml.PhaseShift(np.pi / 4, wires=qrisp_qv[1].identifier),
            qml.T(wires=qrisp_qv[0].identifier),
            qml.T(wires=qrisp_qv[1].identifier),
        ]
        check_qml_operations(qml_converted_circuit, expected_ops)

        qrisp_res = qrisp_qv.get_measurement()
        qml_res = np.asarray(qml_res, dtype=float)
        check_probs_equivalence(qrisp_res, qml_res)

    def test_h_sx_gates(self):
        """Test conversion of H and SX gates from Qrisp to PennyLane."""
        qrisp_qv = QuantumVariable(2)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(HGate(), qrisp_qv[0])
        qrisp_qs.append(SXGate(), qrisp_qv[0])

        qrisp_qs.append(SXGate(), qrisp_qv[1])
        qrisp_qs.append(HGate(), qrisp_qv[1])
        qrisp_qs.append(SXGate(), qrisp_qv[1])

        qrisp_qs.append(HGate(), qrisp_qv[0])
        qrisp_qs.append(HGate(), qrisp_qv[1])

        qml_converted_circuit = _create_qml_circuit(qrisp_qv)

        # We execute the circuit to populate the tape
        qml_res = qml_converted_circuit()

        expected_ops = [
            qml.Hadamard(wires=qrisp_qv[0].identifier),
            qml.SX(wires=qrisp_qv[0].identifier),
            qml.SX(wires=qrisp_qv[1].identifier),
            qml.Hadamard(wires=qrisp_qv[1].identifier),
            qml.SX(wires=qrisp_qv[1].identifier),
            qml.Hadamard(wires=qrisp_qv[0].identifier),
            qml.Hadamard(wires=qrisp_qv[1].identifier),
        ]
        check_qml_operations(qml_converted_circuit, expected_ops)

        qrisp_res = qrisp_qv.get_measurement()
        qml_res = np.asarray(qml_res, dtype=float)
        check_probs_equivalence(qrisp_res, qml_res)

    def test_adjoint_gates(self):
        """Test conversion of adjoint gates from Qrisp to PennyLane."""
        qrisp_qv = QuantumVariable(1)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(SGate().inverse(), qrisp_qv[0])
        qrisp_qs.append(SXGate().inverse(), qrisp_qv[0])
        qrisp_qs.append(TGate().inverse(), qrisp_qv[0])
        qrisp_qs.append(RXGate(0.5).inverse(), qrisp_qv[0])

        qml_converted_circuit = _create_qml_circuit(qrisp_qv)

        # We execute the circuit to populate the tape
        qml_res = qml_converted_circuit()

        expected_ops = [
            qml.adjoint(qml.S)(wires=qrisp_qv[0].identifier),
            qml.adjoint(qml.SX)(wires=qrisp_qv[0].identifier),
            qml.adjoint(qml.T)(wires=qrisp_qv[0].identifier),
            qml.RX(-0.5, wires=qrisp_qv[0].identifier),
        ]
        check_qml_operations(qml_converted_circuit, expected_ops)

        qrisp_res = qrisp_qv.get_measurement()
        qml_res = np.asarray(qml_res, dtype=float)
        check_probs_equivalence(qrisp_res, qml_res)


class TestMultiQubitGateConversion:
    """Test class for multi-qubit gate conversions from Qrisp to PennyLane."""

    @pytest.mark.parametrize(
        "qrisp_gate_class,qml_gate_class,n_qubits,params", MULTI_GATE_MAP
    )
    def test_multi_qubit_gate_conversion(
        self, qrisp_gate_class, qml_gate_class, n_qubits, params
    ):
        """Test conversion of multi-qubit gates from Qrisp to PennyLane."""
        qv = QuantumVariable(n_qubits)
        qs = qv.qs

        targets = [qv[i] for i in range(n_qubits)]

        qs.append(qrisp_gate_class(*params), targets)

        qml_circ = _create_qml_circuit(qv)
        qml_res = qml_circ()  # populate tape

        expected_ops = [qml_gate_class(*params, wires=[q.identifier for q in targets])]
        check_qml_operations(qml_circ, expected_ops)

        qrisp_res = qv.get_measurement()
        qml_res = np.asarray(qml_res, dtype=float)
        atol = 1e-5 if params else 1e-8
        check_probs_equivalence(qrisp_res, qml_res, atol=atol)

    def test_swap_rxx_rzz_gates(self):
        """Test conversion of SWAP, RXX, RZZ gates from Qrisp to PennyLane."""
        qrisp_qv = QuantumVariable(2)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(SwapGate(), [qrisp_qv[0], qrisp_qv[1]])
        qrisp_qs.append(RXXGate(np.pi / 2), [qrisp_qv[0], qrisp_qv[1]])
        qrisp_qs.append(RZZGate(np.pi / 4), [qrisp_qv[1], qrisp_qv[0]])

        qml_converted_circuit = _create_qml_circuit(qrisp_qv)

        # We execute the circuit to populate the tape
        qml_res = qml_converted_circuit()

        expected_ops = [
            qml.SWAP(wires=[qrisp_qv[0].identifier, qrisp_qv[1].identifier]),
            qml.IsingXX(
                np.pi / 2, wires=[qrisp_qv[0].identifier, qrisp_qv[1].identifier]
            ),
            qml.IsingZZ(
                np.pi / 4, wires=[qrisp_qv[1].identifier, qrisp_qv[0].identifier]
            ),
        ]
        check_qml_operations(qml_converted_circuit, expected_ops)

        qrisp_res = qrisp_qv.get_measurement()
        qml_res = np.asarray(qml_res, dtype=float)
        check_probs_equivalence(qrisp_res, qml_res)


class TestControlledGateConversion:
    """Test class for controlled gate conversions from Qrisp to PennyLane."""

    @pytest.mark.parametrize(
        "qrisp_gate_class,qml_gate_class,params", CONTROLLED_GATE_MAP
    )
    def test_controlled_gate_conversion(self, qrisp_gate_class, qml_gate_class, params):
        """Test conversion of controlled gates from Qrisp to PennyLane."""
        qv = QuantumVariable(2)
        qs = qv.qs
        qs.append(qrisp_gate_class(*params), [qv[0], qv[1]])

        qml_circ = _create_qml_circuit(qv)
        qml_res = qml_circ()  # populate tape

        expected_ops = [
            qml_gate_class(*params, wires=[q.identifier for q in [qv[0], qv[1]]])
        ]
        check_qml_operations(qml_circ, expected_ops)

        qrisp_res = qv.get_measurement()
        qml_res = np.asarray(qml_res, dtype=float)
        atol = 1e-5 if params else 1e-8
        check_probs_equivalence(qrisp_res, qml_res, atol=atol)

    def test_cx_cz_cy_gates(self):
        """Test conversion of CX, CZ, CY gates from Qrisp to PennyLane."""
        qrisp_qv = QuantumVariable(3)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(CXGate(), [qrisp_qv[0], qrisp_qv[1]])
        qrisp_qs.append(CZGate(), [qrisp_qv[1], qrisp_qv[2]])
        qrisp_qs.append(CYGate(), [qrisp_qv[2], qrisp_qv[0]])

        qml_converted_circuit = _create_qml_circuit(qrisp_qv)

        # We execute the circuit to populate the tape
        qml_res = qml_converted_circuit()

        expected_ops = [
            qml.CNOT(wires=[qrisp_qv[0].identifier, qrisp_qv[1].identifier]),
            qml.CZ(wires=[qrisp_qv[1].identifier, qrisp_qv[2].identifier]),
            qml.CY(wires=[qrisp_qv[2].identifier, qrisp_qv[0].identifier]),
        ]
        check_qml_operations(qml_converted_circuit, expected_ops)

        qrisp_res = qrisp_qv.get_measurement()
        qml_res = np.asarray(qml_res, dtype=float)
        check_probs_equivalence(qrisp_res, qml_res)

    def test_mcx_gate(self):
        """Test conversion of MCX gate from Qrisp to PennyLane."""
        qrisp_qv = QuantumVariable(4)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(
            MCXGate(control_amount=3, ctrl_state="101"),
            [qrisp_qv[0], qrisp_qv[1], qrisp_qv[2], qrisp_qv[3]],
        )

        qml_converted_circuit = _create_qml_circuit(qrisp_qv)

        # We execute the circuit to populate the tape
        qml_res = qml_converted_circuit()

        expected_ops = [
            qml.MultiControlledX(
                wires=[
                    qrisp_qv[0].identifier,
                    qrisp_qv[1].identifier,
                    qrisp_qv[2].identifier,
                    qrisp_qv[3].identifier,
                ],
                control_values=[1, 0, 1],
            ),
        ]
        check_qml_operations(qml_converted_circuit, expected_ops)

        qrisp_res = qrisp_qv.get_measurement()
        qml_res = np.asarray(qml_res, dtype=float)
        check_probs_equivalence(qrisp_res, qml_res)

    def test_mcrx_gate(self):
        """Test conversion of MCRX gate from Qrisp to PennyLane."""
        qrisp_qv = QuantumVariable(4)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(
            MCRXGate(0.5, control_amount=3),
            [qrisp_qv[0], qrisp_qv[1], qrisp_qv[2], qrisp_qv[3]],
        )

        qml_converted_circuit = _create_qml_circuit(qrisp_qv)

        # We execute the circuit to populate the tape
        qml_res = qml_converted_circuit()

        expected_ops = [
            qml.ctrl(
                op=qml.RX(0.5, wires=qrisp_qv[3].identifier),
                control=[
                    qrisp_qv[0].identifier,
                    qrisp_qv[1].identifier,
                    qrisp_qv[2].identifier,
                ],
                control_values=[1, 1, 1],
            ),
        ]
        check_qml_operations(qml_converted_circuit, expected_ops)

        qrisp_res = qrisp_qv.get_measurement()
        qml_res = np.asarray(qml_res, dtype=float)
        check_probs_equivalence(qrisp_res, qml_res)

    @pytest.mark.parametrize(
        "gate_class, qml_gate_class",
        [
            (RZGate, qml.RZ),
            (RYGate, qml.RY),
            (RXGate, qml.RX),
        ],
    )
    def test_custom_controlled_gate(self, gate_class, qml_gate_class):
        """Test conversion of custom controlled rotation gates (RZ, RY, RX)
        from Qrisp to PennyLane, including multi-controlled variants.
        """

        qrisp_qv = QuantumVariable(5)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(
            gate_class(0.5).control(num_ctrl_qubits=4, ctrl_state="0101"),
            [qrisp_qv[2], qrisp_qv[0], qrisp_qv[1], qrisp_qv[4], qrisp_qv[3]],
        )

        qml_converted_circuit = _create_qml_circuit(qrisp_qv)
        qml_res = qml_converted_circuit()  # Execute to populate tape

        expected_ops = [
            qml.ctrl(
                op=qml_gate_class(0.5, wires=qrisp_qv[3].identifier),
                control=[
                    qrisp_qv[2].identifier,
                    qrisp_qv[0].identifier,
                    qrisp_qv[1].identifier,
                    qrisp_qv[4].identifier,
                ],
                control_values=[0, 1, 0, 1],
            )
        ]

        check_qml_operations(qml_converted_circuit, expected_ops)

        qrisp_res = qrisp_qv.get_measurement()
        qml_res = np.asarray(qml_res, dtype=float)
        check_probs_equivalence(qrisp_res, qml_res)

    def test_controlled_adjoint_gate(self):
        """Test conversion of controlled adjoint gates from Qrisp to PennyLane."""
        qrisp_qv = QuantumVariable(4)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(
            SXGate().control(num_ctrl_qubits=3, ctrl_state="010").inverse(),
            [qrisp_qv[3], qrisp_qv[0], qrisp_qv[1], qrisp_qv[2]],
        )
        qrisp_qs.append(
            SXGate().inverse().control(num_ctrl_qubits=3, ctrl_state="010"),
            [qrisp_qv[3], qrisp_qv[0], qrisp_qv[1], qrisp_qv[2]],
        )
        qml_converted_circuit = _create_qml_circuit(qrisp_qv)
        qml_res = qml_converted_circuit()  # Execute to populate tape

        expected_ops = [
            qml.ctrl(
                op=qml.adjoint(qml.SX)(wires=qrisp_qv[2].identifier),
                control=[
                    qrisp_qv[3].identifier,
                    qrisp_qv[0].identifier,
                    qrisp_qv[1].identifier,
                ],
                control_values=[0, 1, 0],
            ),
            qml.ctrl(
                op=qml.adjoint(qml.SX)(wires=qrisp_qv[2].identifier),
                control=[
                    qrisp_qv[3].identifier,
                    qrisp_qv[0].identifier,
                    qrisp_qv[1].identifier,
                ],
                control_values=[0, 1, 0],
            ),
        ]
        check_qml_operations(qml_converted_circuit, expected_ops)
        qrisp_res = qrisp_qv.get_measurement()
        qml_res = np.asarray(qml_res, dtype=float)
        check_probs_equivalence(qrisp_res, qml_res)


# This was the previously existing test
def randomized_circuit_testing():

    ############ create the randomized circuit
    qvRand = QuantumVariable(10)
    qcRand = qvRand.qs

    single_gates = [XGate(), YGate(), ZGate(), HGate(), SXGate(), SGate()]

    rot_gates = [RXGate(0.5), RYGate(0.5), RZGate(0.5), PGate(0.5)]

    c_gates = [CXGate(), CYGate(), CZGate()]

    mc_gates = [SwapGate(), RXXGate(0.5), RZZGate(0.5)]

    special_gates = [MCXGate(control_amount=3), MCRXGate(0.5, control_amount=3)]

    op_list = [*mc_gates, *c_gates, *single_gates, *rot_gates, *special_gates]

    used_ops = []
    for index in range(30):
        used_ops.append(op_list[np.random.randint(0, len(op_list))])
    for op in used_ops:
        qubit_1 = qvRand[np.random.randint(7, 9)]
        qubit_2 = qvRand[np.random.randint(3, 6)]
        if op in single_gates or op in rot_gates:
            qcRand.append(op, qubit_1)
        # this is being called first due to mcx is subclass of cx reasons
        elif op in special_gates:
            qcRand.append(op, [qvRand[0], qvRand[2], qubit_2, qubit_1])
        elif op in c_gates or mc_gates:
            qcRand.append(op, [qubit_1, qubit_2])
    ######## done with the circuit

    # instanciate pennylane circuit
    qcRand_qubits = [qubit.identifier for qubit in qcRand.qubits]
    dev = qml.device("default.qubit", wires=qcRand_qubits)
    pl_qcRand = qml_converter(qc=qcRand)
    circ = pl_qcRand

    @qml.set_shots(1000)
    @qml.qnode(dev)
    def circuit():
        circ()
        return qml.probs(wires=qcRand_qubits)

    # save result from probability measurement
    # --> look up   qml.probs()   documentation for further insights
    # --> the result is an ordered array containing measurement probabilities (ordered in terms of binary strings representing the measurement results)
    qml_res_w_zeros = circuit().tolist()
    qml_res = qml_res_w_zeros

    # get qrisp-qv measurement
    qrisp_res = qvRand.get_measurement()
    # sort the keys in binary order
    qrisp_keySort = sorted(qrisp_res, key=lambda x: int(x, 2))
    temp = True

    for index in range(len(qrisp_keySort)):
        # get the index for qml_res array --> can trafo to int from binary
        index2 = int(qrisp_keySort[index], 2)
        if qrisp_res[qrisp_keySort[index]] < 0.13:
            # small results are problematic... might have to adjust for accuracy
            continue
        # we allow for slight deviations in result probabilities

        if (
            not qrisp_res[qrisp_keySort[index]] * 0.8
            <= qml_res[index2]
            <= qrisp_res[qrisp_keySort[index]] * 1.2
        ):

            temp = False
            break

    assert temp


def test_wrapper():

    for _ in range(20):
        randomized_circuit_testing()
