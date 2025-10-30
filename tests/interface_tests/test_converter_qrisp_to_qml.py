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

import numpy as np
import pennylane as qml
import pytest
from sympy import symbols

from qrisp import QuantumVariable
from qrisp.circuit.standard_operations import (  # Barrier,; Measurement,; QubitAlloc,; QubitDealloc,; Reset,; RGate,
    CPGate,
    CXGate,
    CYGate,
    CZGate,
    GPhaseGate,
    HGate,
    IDGate,
    MCRXGate,
    MCXGate,
    PGate,
    RXGate,
    RXXGate,
    RYGate,
    RZGate,
    RZZGate,
    SGate,
    SwapGate,
    SXGate,
    TGate,
    U1Gate,
    XGate,
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


# The circuit returned should be executed to populate the tape
def _create_qml_circuit(qrisp_circuit, inner_wires=None, subs_dic=None):
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
        pennylane_circuit(wires=inner_wires, subs_dic=subs_dic)
        return qml.probs(wires=inner_wires if inner_wires is not None else qrisp_qubits)

    return circuit


def check_qml_operations(qml_circuit, expected_ops):
    """Helper function to check if the PennyLane operations in a circuit match the expected operations."""
    qml_ops = _get_qml_operations(qml_circuit)
    for op, expected_op in zip(qml_ops, expected_ops, strict=True):
        qml.assert_equal(op, expected_op)


def check_probs_measurement_equivalence(qrisp_qv, qml_res, atol=1e-8):
    """Helper function to check if the measurement probabilities from Qrisp and PennyLane are equivalent."""

    qrisp_res = qrisp_qv.get_measurement()
    qml_probs = np.asarray(qml_res, dtype=float)

    qrisp_items = sorted(qrisp_res.items(), key=lambda kv: int(kv[0], 2))

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

        check_probs_measurement_equivalence(qv, qml_res, atol=1e-5 if params else 1e-8)

    def test_inner_wires(self):
        """Test conversion of a circuit with inner wires provided to the converter."""
        qrisp_qv = QuantumVariable(2)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(XGate(), qrisp_qv[0])
        qrisp_qs.append(HGate(), qrisp_qv[1])

        inner_wires = ["aux_0", "aux_1"]
        qml_converted_circuit = _create_qml_circuit(qrisp_qv, inner_wires=inner_wires)
        qml_res = qml_converted_circuit()

        expected_ops = [
            qml.X(wires="aux_0"),
            qml.Hadamard(wires="aux_1"),
        ]
        check_qml_operations(qml_converted_circuit, expected_ops)

        check_probs_measurement_equivalence(qrisp_qv, qml_res)

    @pytest.mark.parametrize("outer_wires", [[0, 1], [2, 3], [1, 4]])
    def test_nested_circuits(self, outer_wires):
        """Test conversion of nested circuits from Qrisp to PennyLane."""

        inner_qv = QuantumVariable(2)
        inner_qs = inner_qv.qs
        inner_qs.append(CXGate(), [0, 1])

        outer_qv = QuantumVariable(5)
        outer_qs = outer_qv.qs
        outer_qs.append(CYGate(), [0, 1])
        outer_qs.append(inner_qs.to_gate(name="inner_circuit"), outer_wires)

        qml_converted_circuit = _create_qml_circuit(outer_qv)
        qml_res = qml_converted_circuit()

        expected_ops = [
            qml.CY(wires=[outer_qv[0].identifier, outer_qv[1].identifier]),
            qml.CNOT(
                wires=[
                    outer_qv[outer_wires[0]].identifier,
                    outer_qv[outer_wires[1]].identifier,
                ]
            ),
        ]
        check_qml_operations(qml_converted_circuit, expected_ops)

        check_probs_measurement_equivalence(outer_qv, qml_res)

    def test_nested_circuits_multiple_levels(self):
        """Test conversion of nested circuits with multiple levels from Qrisp to PennyLane."""

        inner_qv = QuantumVariable(2)
        inner_qs = inner_qv.qs
        inner_qs.append(CXGate(), [0, 1])

        middle_qv = QuantumVariable(3)
        middle_qs = middle_qv.qs
        middle_qs.append(RXGate(np.pi / 4), middle_qv[0])
        middle_qs.append(inner_qs.to_gate(name="inner_circuit"), [1, 2])

        outer_qv = QuantumVariable(4)
        outer_qs = outer_qv.qs
        outer_qs.append(XGate(), outer_qv[0])
        outer_qs.append(middle_qs.to_gate(name="middle_circuit"), [1, 2, 3])

        qml_converted_circuit = _create_qml_circuit(outer_qv)
        qml_res = qml_converted_circuit()

        expected_ops = [
            qml.X(wires=outer_qv[0].identifier),
            qml.RX(np.pi / 4, wires=outer_qv[1].identifier),
            qml.CNOT(wires=[outer_qv[2].identifier, outer_qv[3].identifier]),
        ]
        check_qml_operations(qml_converted_circuit, expected_ops)

        check_probs_measurement_equivalence(outer_qv, qml_res, atol=1e-5)

    def test_abstract_parameters(self):
        """Test conversion of gates with abstract parameters from Qrisp to PennyLane."""
        qrisp_qv = QuantumVariable(1)
        qrisp_qs = qrisp_qv.qs

        abstract_parameters = symbols("a b c")

        qrisp_qs.append(RXGate(abstract_parameters[0]), qrisp_qv[0])
        qrisp_qs.append(RZGate(abstract_parameters[0]), qrisp_qv[0])
        qrisp_qs.append(PGate(1), qrisp_qv[0])
        qrisp_qs.append(
            RYGate(abstract_parameters[1] + abstract_parameters[2]), qrisp_qv[0]
        )

        subs_dic = {
            abstract_parameters[0]: np.pi / 4,
            abstract_parameters[1]: np.pi / 2,
            abstract_parameters[2]: np.pi / 3,
        }

        qml_converted_circuit = _create_qml_circuit(qrisp_qv, subs_dic=subs_dic)
        _ = qml_converted_circuit()

        expected_ops = [
            qml.RX(np.pi / 4, wires=qrisp_qv[0].identifier),
            qml.RZ(np.pi / 4, wires=qrisp_qv[0].identifier),
            qml.PhaseShift(1, wires=qrisp_qv[0].identifier),
            qml.RY(np.pi / 2 + np.pi / 3, wires=qrisp_qv[0].identifier),
        ]

        check_qml_operations(qml_converted_circuit, expected_ops)

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

        check_probs_measurement_equivalence(qrisp_qv, qml_res)

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

        check_probs_measurement_equivalence(qrisp_qv, qml_res, atol=1e-5)

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

        check_probs_measurement_equivalence(qrisp_qv, qml_res)

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

        check_probs_measurement_equivalence(qrisp_qv, qml_res)

    def test_adjoint_gates(self):
        """Test conversion of adjoint gates from Qrisp to PennyLane."""
        qrisp_qv = QuantumVariable(1)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(SGate().inverse(), qrisp_qv[0])
        qrisp_qs.append(SXGate().inverse(), qrisp_qv[0])
        qrisp_qs.append(TGate().inverse(), qrisp_qv[0])
        qrisp_qs.append(RXGate(0.5).inverse(), qrisp_qv[0])

        qml_converted_circuit = _create_qml_circuit(qrisp_qv)
        qml_res = qml_converted_circuit()

        expected_ops = [
            qml.adjoint(qml.S)(wires=qrisp_qv[0].identifier),
            qml.adjoint(qml.SX)(wires=qrisp_qv[0].identifier),
            qml.adjoint(qml.T)(wires=qrisp_qv[0].identifier),
            qml.RX(-0.5, wires=qrisp_qv[0].identifier),
        ]
        check_qml_operations(qml_converted_circuit, expected_ops)

        check_probs_measurement_equivalence(qrisp_qv, qml_res)


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
        qml_res = qml_circ()

        expected_ops = [qml_gate_class(*params, wires=[q.identifier for q in targets])]
        check_qml_operations(qml_circ, expected_ops)

        check_probs_measurement_equivalence(qv, qml_res, atol=1e-5 if params else 1e-8)

    def test_swap_rxx_rzz_gates(self):
        """Test conversion of SWAP, RXX, RZZ gates from Qrisp to PennyLane."""
        qrisp_qv = QuantumVariable(2)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(SwapGate(), [qrisp_qv[0], qrisp_qv[1]])
        qrisp_qs.append(RXXGate(np.pi / 2), [qrisp_qv[0], qrisp_qv[1]])
        qrisp_qs.append(RZZGate(np.pi / 4), [qrisp_qv[1], qrisp_qv[0]])

        qml_converted_circuit = _create_qml_circuit(qrisp_qv)
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

        check_probs_measurement_equivalence(qrisp_qv, qml_res)


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
        qml_res = qml_circ()

        expected_ops = [
            qml_gate_class(*params, wires=[q.identifier for q in [qv[0], qv[1]]])
        ]
        check_qml_operations(qml_circ, expected_ops)

        check_probs_measurement_equivalence(qv, qml_res, atol=1e-5 if params else 1e-8)

    def test_cx_cz_cy_gates(self):
        """Test conversion of CX, CZ, CY gates from Qrisp to PennyLane."""
        qrisp_qv = QuantumVariable(3)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(CXGate(), [qrisp_qv[0], qrisp_qv[1]])
        qrisp_qs.append(CZGate(), [qrisp_qv[1], qrisp_qv[2]])
        qrisp_qs.append(CYGate(), [qrisp_qv[2], qrisp_qv[0]])

        qml_converted_circuit = _create_qml_circuit(qrisp_qv)
        qml_res = qml_converted_circuit()

        expected_ops = [
            qml.CNOT(wires=[qrisp_qv[0].identifier, qrisp_qv[1].identifier]),
            qml.CZ(wires=[qrisp_qv[1].identifier, qrisp_qv[2].identifier]),
            qml.CY(wires=[qrisp_qv[2].identifier, qrisp_qv[0].identifier]),
        ]
        check_qml_operations(qml_converted_circuit, expected_ops)

        check_probs_measurement_equivalence(qrisp_qv, qml_res)

    def test_mcx_gate(self):
        """Test conversion of MCX gate from Qrisp to PennyLane."""
        qrisp_qv = QuantumVariable(4)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(
            MCXGate(control_amount=3, ctrl_state="101"),
            [qrisp_qv[0], qrisp_qv[1], qrisp_qv[2], qrisp_qv[3]],
        )

        qml_converted_circuit = _create_qml_circuit(qrisp_qv)
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

        check_probs_measurement_equivalence(qrisp_qv, qml_res)

    def test_mcrx_gate(self):
        """Test conversion of MCRX gate from Qrisp to PennyLane."""
        qrisp_qv = QuantumVariable(4)
        qrisp_qs = qrisp_qv.qs

        qrisp_qs.append(
            MCRXGate(0.5, control_amount=3),
            [qrisp_qv[0], qrisp_qv[1], qrisp_qv[2], qrisp_qv[3]],
        )

        qml_converted_circuit = _create_qml_circuit(qrisp_qv)
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

        check_probs_measurement_equivalence(qrisp_qv, qml_res)

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
        qml_res = qml_converted_circuit()

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

        check_probs_measurement_equivalence(qrisp_qv, qml_res)

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
        qml_res = qml_converted_circuit()

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

        check_probs_measurement_equivalence(qrisp_qv, qml_res)


def test_mixed_circuit():
    """Test conversion of a mixed circuit with various gate types from Qrisp to PennyLane."""

    qrisp_qv = QuantumVariable(10)
    qrisp_qs = qrisp_qv.qs

    single_gates = [XGate(), YGate(), ZGate(), HGate(), SXGate(), SGate()]
    rot_gates = [RXGate(0.5), RYGate(0.5), RZGate(0.5), PGate(0.5)]
    c_gates = [CXGate(), CYGate(), CZGate()]
    mc_gates = [SwapGate(), RXXGate(0.5), RZZGate(0.5)]
    special_gates = [MCXGate(control_amount=3), MCRXGate(0.5, control_amount=3)]

    for idx, gate in enumerate(single_gates):
        qrisp_qs.append(gate, qrisp_qv[idx])

    for offset, gate in enumerate(rot_gates, start=1):
        qrisp_qs.append(gate, qrisp_qv[offset])

    controlled_pairs = [(2, 3), (3, 4), (4, 5)]
    for gate, (a, b) in zip(c_gates, controlled_pairs):
        qrisp_qs.append(gate, [qrisp_qv[a], qrisp_qv[b]])

    qrisp_qs.append(mc_gates[0], [qrisp_qv[6], qrisp_qv[7]])  # SWAP
    qrisp_qs.append(mc_gates[1], [qrisp_qv[7], qrisp_qv[8]])  # RXX
    qrisp_qs.append(mc_gates[2], [qrisp_qv[8], qrisp_qv[9]])  # RZZ

    qrisp_qs.append(
        special_gates[0], [qrisp_qv[0], qrisp_qv[1], qrisp_qv[2], qrisp_qv[3]]
    )  # MCX
    qrisp_qs.append(
        special_gates[1], [qrisp_qv[3], qrisp_qv[4], qrisp_qv[5], qrisp_qv[6]]
    )  # MCRX

    qml_converted_circuit = _create_qml_circuit(qrisp_qv)
    qml_res = qml_converted_circuit()  # Execute to populate

    expected_ops = [
        qml.X(qrisp_qv[0].identifier),
        qml.Y(qrisp_qv[1].identifier),
        qml.Z(qrisp_qv[2].identifier),
        qml.H(qrisp_qv[3].identifier),
        qml.SX(qrisp_qv[4].identifier),
        qml.S(qrisp_qv[5].identifier),
        qml.RX(0.5, wires=qrisp_qv[1].identifier),
        qml.RY(0.5, wires=qrisp_qv[2].identifier),
        qml.RZ(0.5, wires=qrisp_qv[3].identifier),
        qml.PhaseShift(0.5, wires=qrisp_qv[4].identifier),
        qml.CNOT(wires=[qrisp_qv[2].identifier, qrisp_qv[3].identifier]),
        qml.CY(wires=[qrisp_qv[3].identifier, qrisp_qv[4].identifier]),
        qml.CZ(wires=[qrisp_qv[4].identifier, qrisp_qv[5].identifier]),
        qml.SWAP(wires=[qrisp_qv[6].identifier, qrisp_qv[7].identifier]),
        qml.IsingXX(0.5, wires=[qrisp_qv[7].identifier, qrisp_qv[8].identifier]),
        qml.IsingZZ(0.5, wires=[qrisp_qv[8].identifier, qrisp_qv[9].identifier]),
        qml.MultiControlledX(
            wires=[
                qrisp_qv[0].identifier,
                qrisp_qv[1].identifier,
                qrisp_qv[2].identifier,
                qrisp_qv[3].identifier,
            ],
            control_values=[1, 1, 1],
        ),
        qml.ctrl(
            op=qml.RX(0.5, wires=qrisp_qv[6].identifier),
            control=[
                qrisp_qv[3].identifier,
                qrisp_qv[4].identifier,
                qrisp_qv[5].identifier,
            ],
            control_values=[1, 1, 1],
        ),
    ]

    check_qml_operations(qml_converted_circuit, expected_ops)

    check_probs_measurement_equivalence(qrisp_qv, qml_res, atol=1e-4)
