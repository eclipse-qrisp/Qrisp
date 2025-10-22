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

-- Add test with wires=None
-- Add test with parameterized gates

"""

# pylint: disable=line-too-long

import numpy as np
import pennylane as qml

from qrisp import QuantumVariable
from qrisp.circuit.standard_operations import (
    CXGate,
    CYGate,
    CZGate,
    HGate,
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
    XGate,
    YGate,
    ZGate,
)
from qrisp.interface import qml_converter


def _get_qml_operations(qml_circuit):
    """Helper function to extract PennyLane operations from a PennyLane QNode."""
    qml_ops = []
    # pylint: disable=protected-access
    for op in qml_circuit._tape.operations:
        qml_ops.append(op)
    return qml_ops


def _create_qml_circuit(qrisp_circuit):
    """Helper function to create a PennyLane circuit from a Qrisp circuit."""
    qrisp_qubits = [qubit.identifier for qubit in qrisp_circuit]

    @qml.qnode(device=qml.device("default.qubit", wires=qrisp_qubits))
    def circuit():
        qml_converter(qc=qrisp_circuit.qs)()
        return qml.probs(wires=qrisp_qubits)

    return circuit


def check_qml_operations(qml_circuit, expected_ops):
    """Helper function to check if the PennyLane operations match the expected operations."""
    qml_ops = _get_qml_operations(qml_circuit)
    for op, expected_op in zip(qml_ops, expected_ops, strict=True):
        qml.assert_equal(op, expected_op)


def check_probs_equivalence(qrisp_probs, qml_probs, atol=1e-8):
    """Helper function to check if the measurement probabilities from Qrisp and PennyLane are equivalent."""
    # sort Qrisp entries by binary value so indices match qml_probs ordering
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


# TODO: continue from here

# create randomized qrisp circuit, convert to pennylane, measure outcomes and compare if they are equivalent.


def randomized_ciruit_testing():

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
        randomized_ciruit_testing()
