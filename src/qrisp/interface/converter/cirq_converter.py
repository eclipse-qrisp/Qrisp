import numpy as np

from cirq import Circuit, LineQubit
from qrisp.circuit import ControlledOperation

from cirq import (
    CNOT,
    H,
    X,
    Y,
    Z,
    S,
    T,
    SWAP,
    rx,
    ry,
    rz,
    inverse,
    I,
    M,
    R,
    CZ,
    CCNOT,
    ZPowGate,
    XPowGate,
)

qrisp_cirq_ops_dict = {
    "cx": CNOT,
    "cz": CZ,
    "swap": SWAP,
    "2cx": CCNOT,
    "h": H,
    "x": X,
    "y": Y,
    "z": Z,
    "rx": rx,
    "ry": ry,
    "rz": rz,
    "s": S,
    "t": T,
    "s_dg": inverse(S),
    "t_dg": inverse(T),
    "measure": M,
    "reset": R,
    "id": I,
    "p": ZPowGate,
    "sx": XPowGate,
    "sx_dg": XPowGate,
    # the converter skips adding the global phase gate
    # because the corresponding Cirq gate raises an error
    # for anything besides 1 and 1j as the coefficient
    # The qrisp global phase function does not accept
    # a complex value as input.
    "gphase": None,
    "xxyy": None,
    "rxx": None,
    "rzz": None,
    "qb_alloc": None,
    "qb_dealloc": None,
}


def convert_to_cirq(qrisp_circuit):
    """Function to convert a Qrisp circuit to a Cirq circuit."""
    # get data from Qrisp circuit
    qrisp_circ_num_qubits = qrisp_circuit.num_qubits()
    qrisp_circ_ops_data = qrisp_circuit.data

    # create generic 'ncx' keys in qrisp_cirq_ops_dict for the possibility of multicontrolled cx gates
    for n in range(3, qrisp_circ_num_qubits + 1):
        # start at 3 because 2cx is a Toffoli gate which already exists in Cirq
        qrisp_cirq_ops_dict[f"{n}cx"] = None

    # create an empty Cirq circuit
    cirq_circuit = Circuit()
    cirq_qubits = [LineQubit(i) for i in range(qrisp_circ_num_qubits)]

    # create a mapping of Qrisp qubits to Cirq qubits
    qubit_map = {}
    for i, q in enumerate(qrisp_circuit.qubits):
        qubit_map[q] = cirq_qubits[i]

    for instr in qrisp_circ_ops_data:
        # get the gate name, qubits it is acting on
        # and parameters if there are any
        op_i = instr.op.name
        op_qubits_i = instr.qubits

        if hasattr(instr.op, "params"):
            params = instr.op.params

        if op_i not in qrisp_cirq_ops_dict:
            raise ValueError(
                f"{op_i} gate is not supported by the Qrisp to Cirq converter."
            )

        if op_i in ["gphase", "rzz", "rxx"]:
            print(
                "Qrisp circuit contains a global phase gate which will be skipped in the Qrisp to Cirq conversion."
            )

        cirq_op_qubits = [qubit_map[q] for q in op_qubits_i]

        cirq_gate = qrisp_cirq_ops_dict[op_i]

        # for single qubit parametrized gates
        if (
            op_i != "id"
            and instr.op.params
            and not isinstance(instr.op, ControlledOperation)
        ):
            # added op_i != 'id' becasue the identity gate has parameters (0, 0, 0)
            if op_i == "p":
                # Cirq does not have a phase gate
                # for this reason, it has to be dealt with as a special case.
                # the ZPowGate has a global phase in addition to the
                # phase exponent. The default is to assume global_shift = 0 in cirq
                exp_param = params[0]
                cirq_circuit.append(ZPowGate(exponent=exp_param)(*cirq_op_qubits))

            elif op_i == "rxx":
                phi = params[0]
                # cirq_circuit.append(GlobalPhaseGate(-phi / 2).on())
                cirq_circuit.append(H(cirq_op_qubits[0]))
                cirq_circuit.append(H(cirq_op_qubits[1]))
                cirq_circuit.append(CNOT(cirq_op_qubits[0], cirq_op_qubits[1]))
                cirq_circuit.append(ZPowGate(exponent=phi)(cirq_op_qubits[1]))
                cirq_circuit.append(CNOT(cirq_op_qubits[0], cirq_op_qubits[1]))
                cirq_circuit.append(H(cirq_op_qubits[0]))
                cirq_circuit.append(H(cirq_op_qubits[1]))

            elif op_i == "rzz":
                phi = params[0]
                cirq_circuit.append(CNOT(cirq_op_qubits[0], cirq_op_qubits[1]))
                cirq_circuit.append(ZPowGate(exponent=phi)(cirq_op_qubits[1]))
                cirq_circuit.append(CNOT(cirq_op_qubits[0], cirq_op_qubits[1]))

            elif op_i == "xxyy":
                phi = params[0]
                beta = params[1]
                cirq_circuit.append(rz(beta).on(cirq_op_qubits[0]))
                cirq_circuit.append(rz(-np.pi / 2).on(cirq_op_qubits[1]))
                cirq_circuit.append(XPowGate(exponent=0.5)(cirq_op_qubits[1]))
                cirq_circuit.append(rz(np.pi / 2).on(cirq_op_qubits[1]))
                cirq_circuit.append(S(cirq_op_qubits[0]))
                cirq_circuit.append(CNOT(cirq_op_qubits[1], cirq_op_qubits[0]))
                cirq_circuit.append(ry(-phi / 2).on(cirq_op_qubits[1]))
                cirq_circuit.append(ry(-phi / 2).on(cirq_op_qubits[0]))
                cirq_circuit.append(CNOT(cirq_op_qubits[1], cirq_op_qubits[0]))
                cirq_circuit.append(inverse(S(cirq_op_qubits[0])))
                cirq_circuit.append(rz(-np.pi / 2).on(cirq_op_qubits[1]))
                cirq_circuit.append(inverse(XPowGate(exponent=0.5)(cirq_op_qubits[1])))
                cirq_circuit.append(rz(np.pi / 2).on(cirq_op_qubits[1]))
                cirq_circuit.append(rz(-beta).on(cirq_op_qubits[0]))

            # elif op_i == 'gphase':
            # global phase gate in Cirq cannot be applied to specific qubits
            # cirq_circuit.append(GlobalPhaseGate(*params).on())

            elif cirq_gate:
                gate_instance = cirq_gate(*params)
                cirq_circuit.append(gate_instance(*cirq_op_qubits))

        elif isinstance(instr.op, ControlledOperation):
            # control and target qubits from qrisp
            control_qubits = instr.qubits[: len(instr.op.ctrl_state)]
            target_qubits = instr.qubits[len(instr.op.ctrl_state) :]

            cirq_ctrl_qubits = [qubit_map[q] for q in control_qubits]
            cirq_target_qubits = [qubit_map[q] for q in target_qubits]

            # verify the contorl and target qubit mapping worked
            assert cirq_op_qubits == cirq_ctrl_qubits + cirq_target_qubits
            assert op_qubits_i == control_qubits + target_qubits

            # for 2-qubit controlled operations
            if len(cirq_op_qubits) <= 3:
                # if the controlled operation is also parametrized
                if instr.op.params:
                    gate_instance = cirq_gate(*params)
                    cirq_circuit.append(
                        gate_instance(*cirq_ctrl_qubits, *cirq_target_qubits)
                    )
                else:
                    cirq_circuit.append(
                        cirq_gate(*cirq_ctrl_qubits, *cirq_target_qubits)
                    )
            # for multi-qubit controlled operations mcx
            else:
                cirq_circuit.append(
                    X(*cirq_target_qubits).controlled_by(*cirq_ctrl_qubits)
                )

        else:
            # for simple single qubit gates
            if op_i == "sx":
                cirq_circuit.append(XPowGate(exponent=0.5)(*cirq_op_qubits))
            elif op_i == "sx_dg":
                cirq_circuit.append(inverse(XPowGate(exponent=0.5)(*cirq_op_qubits)))
            elif cirq_gate:
                cirq_circuit.append(cirq_gate(*cirq_op_qubits))

    return cirq_circuit
