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

from qrisp.circuit import QuantumCircuit, Qubit, XGate


# This function takes a circuit and turns it into it's controlled version
def multi_controlled_circuit(
    input_circuit, control_amount=1, ctrl_state=-1, method=None
):
    # Create result circuit
    controlled_circuit = QuantumCircuit()

    # Add control qubits
    for i in range(control_amount):
        controlled_circuit.add_qubit(Qubit(identifier="ctrl_qb_" + str(i)))

    # Set alias for qubits
    control_qubits = list(controlled_circuit.qubits)

    # Add circuit qubits
    for qb in input_circuit.qubits:
        controlled_circuit.add_qubit(qb)

    # Apply controlled instructions
    for ins in input_circuit.data:
        if ins.op.name in ["qb_alloc", "qb_dealloc"]:
            # raise
            controlled_circuit.append(ins.op, ins.qubits)
            continue
        controlled_circuit.append(
            ins.op.control(control_amount, method=method, ctrl_state=ctrl_state),
            control_qubits + ins.qubits,
        )

    return controlled_circuit


# This function takes an U3Gate object and turns it into it's controlled version
def multi_controlled_u3_circ(u3_gate, control_amount, ctrl_state, method=None):
    from qrisp.alg_primitives.logic_synthesis import gray_phase_synth_qb_list
    from qrisp.alg_primitives.mcx_algs import multi_cx

    qc = QuantumCircuit(control_amount + 1)
    target_qubit = qc.qubits[-1]

    # Apply control state specification
    for i in range(len(ctrl_state)):
        if ctrl_state[i] == "0":
            qc.x(qc.qubits[i])

    # If the U3 gate is an rx , ry, rz or p gate, we can use gray phase synthesis on the
    # target qubit and wrap this in the corresponding gates
    # (for instance H for rx because RX = H RZ H)
    if u3_gate.name in ["p"]:
        # Synthesize phases using gray synthesis
        gray_phase_synth_qb_list(
            qc,
            qc.qubits,
            (2 ** (control_amount + 1) - 1) * [0] + [u3_gate.params[0]],
            phase_tolerant=method in ["gray_pt", "gray_pt_inv"],
        )

        if method == "gray_pt_inv":
            qc = qc.inverse()

    elif u3_gate.name == "rz":
        # Make sure to account for the possible global phase of rz
        gray_phase_synth_qb_list(
            qc,
            qc.qubits,
            (2 ** (control_amount + 1) - 2) * [0]
            + [-u3_gate.params[0] / 2, u3_gate.params[0] / 2],
            phase_tolerant=method in ["gray_pt", "gray_pt_inv"],
        )

        if method == "gray_pt_inv":
            qc = qc.inverse()

    elif u3_gate.name == "rx":
        # Same thing as with rz but now we use RX = H RZ H
        qc.h(target_qubit)
        gray_phase_synth_qb_list(
            qc,
            qc.qubits,
            (2 ** (control_amount + 1) - 2) * [0]
            + [-u3_gate.theta / 2, u3_gate.theta / 2],
            phase_tolerant=method in ["gray_pt", "gray_pt_inv"],
        )
        qc.h(target_qubit)

        if method == "gray_pt_inv":
            qc = qc.inverse()

    elif u3_gate.name == "ry":
        # Now we use RY = S RX S_DG
        qc.s(target_qubit)
        qc.h(target_qubit)
        gray_phase_synth_qb_list(
            qc,
            qc.qubits,
            (2 ** (control_amount + 1) - 2) * [0]
            + [u3_gate.theta / 2, -u3_gate.theta / 2],
            phase_tolerant=method in ["gray_pt", "gray_pt_inv"],
        )
        qc.h(target_qubit)
        qc.s_dg(target_qubit)

        if method == "gray_pt_inv":
            qc = qc.inverse()

    elif u3_gate.phi == 0 and u3_gate.lam == 0 and u3_gate.theta == 0:
        pass

    # Treat pauli gates
    elif u3_gate.name == "y":
        qc.s(target_qubit)
        qc.append(XGate().control(control_amount, method=method), qc.qubits)
        qc.s_dg(target_qubit)

    elif u3_gate.name == "z":
        qc.h(target_qubit)
        qc.append(XGate().control(control_amount, method=method), qc.qubits)
        qc.h(target_qubit)

    elif u3_gate.name == "x":
        qc.append(multi_cx(control_amount, method), qc.qubits)

    elif u3_gate.name == "h":
        qc.s(target_qubit)
        qc.h(target_qubit)
        qc.t(target_qubit)
        qc.append(XGate().control(control_amount, method=method), qc.qubits)
        qc.t_dg(target_qubit)
        qc.h(target_qubit)
        qc.s_dg(target_qubit)

    # Treat general U3Gates
    else:
        # Algorithm based on https://arxiv.org/pdf/quant-ph/9503016.pdf
        alpha = u3_gate.phi
        theta = -u3_gate.theta
        beta = u3_gate.lam

        A = QuantumCircuit(1)

        A.p(alpha, A.qubits[0])
        A.ry(theta / 2, A.qubits[0])

        B = QuantumCircuit(1)

        B.ry(-theta / 2, B.qubits[0])
        B.p(-(alpha + beta) / 2, B.qubits[0])

        C = QuantumCircuit(1)

        C.p((beta - alpha) / 2, C.qubits[0])

        # Treat global phases and the fact that X P(phi) X = exp(2 phi) P(-phi)
        if method not in ["gray_pt", "gray_pt_inv"]:
            control_phase = -u3_gate.global_phase / 2 - (alpha + beta)
            gray_phase_synth_qb_list(
                qc,
                qc.qubits[:-1],
                (2 ** (control_amount) - 1) * [0] + [-control_phase / 2],
            )

        qc.append(A.to_gate("A"), [qc.qubits[-1]])

        # To perform the controlled x gate, we can use the phase tolerant algorithm

        if control_amount == 2:
            from qrisp.alg_primitives import gray_pt_mcx

            mcx_gate = gray_pt_mcx(2, "11")
        else:
            mcx_gate = XGate().control(control_amount, method="gray_pt")

        qc.append(mcx_gate, qc.qubits)

        qc.append(B.to_gate("B"), [qc.qubits[-1]])

        qc.append(mcx_gate.inverse(), qc.qubits)

        qc.append(C.to_gate("C"), [qc.qubits[-1]])

    # Un-apply control state specification
    for i in range(len(ctrl_state)):
        if ctrl_state[i] == "0":
            qc.x(qc.qubits[i])

    return qc


def multi_controlled_gray_circ(gray_gate, control_amount, ctrl_state):
    from qrisp.alg_primitives.logic_synthesis.gray_synthesis import gray_synth_gate

    target_phases_old = gray_gate.target_phases

    ctrl_state = int(ctrl_state, 2)

    target_phases_new = []

    for i in range(2**control_amount):
        if i != ctrl_state:
            target_phases_new += [0] * len(target_phases_old)
        else:
            target_phases_new += target_phases_old

    new_gray_gate = gray_synth_gate(target_phases_new)
    print(new_gray_gate.definition)
    return new_gray_gate.definition, target_phases_new


def fredkin_qc(num_ctrl_qubits=1, ctrl_state=-1, method="gray"):
    from qrisp import QuantumCircuit, XGate

    mcx_gate = XGate().control().control(ctrl_state=ctrl_state, method=method)

    qc = QuantumCircuit(num_ctrl_qubits + 2)
    qc.cx(qc.qubits[-1], qc.qubits[-2])
    qc.append(mcx_gate, qc.qubits)
    qc.cx(qc.qubits[-1], qc.qubits[-2])

    return qc
