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

from qrisp.circuit import ControlledOperation
import numpy as np

def convert_to_cirq(qrisp_circuit, cirq_qubits = None):
    
    try:
        from cirq import Circuit, LineQubit
    except (ModuleNotFoundError, ImportError):
        raise ImportError("Cirq must be installed to be able to use the Qrisp to Cirq converter.")

    
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
        ZPowGate,
        XPowGate,
        GlobalPhaseGate
    )
    
    qrisp_cirq_ops_dict = {
        "cx": CNOT,
        "cz": CZ,
        "swap": SWAP,
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
        # the converter skips adding the global phase gate for now
        "gphase": None,
        "xxyy": None,
        "rxx": None,
        "rzz": None,
        # skip qubit allocation and deallocation ops in the converter
        "qb_alloc": None,
        "qb_dealloc": None,
    }
    
    """Function to convert a Qrisp circuit to a Cirq circuit."""
    # get data from Qrisp circuit
    qrisp_circ_num_qubits = qrisp_circuit.num_qubits()
    qrisp_circ_ops_data = qrisp_circuit.data

    # create an empty Cirq circuit
    cirq_circuit = Circuit()

    if cirq_qubits is None:
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
            try:
                # this code block also ends up transpiling a mcx gate
                # qrisp allows for different types of control state but cirq does not
                def transpile_predicate(op):
                    if op.name == op_i:
                        return True
                    else:
                        return False

                transpiled_qc = qrisp_circuit.transpile(
                    transpile_predicate=transpile_predicate
                )
                return convert_to_cirq(transpiled_qc)

            except Exception:
                raise ValueError(
                    f"{op_i} gate is not supported by the Qrisp to Cirq converter."
                )

        if op_i == "gphase":
            cirq_circuit.append(GlobalPhaseGate(np.exp(1j * instr.op.params[0]))())
            continue
            
        # the filter is useful for composite gates that do not have a cirq equivalent and have instr.op.definition
        cirq_gates_filter = [
            "cx",
            "cz",
            "swap",
            "h",
            "x",
            "y",
            "z",
            "rx",
            "ry",
            "rz",
            "s",
            "t",
            "s_dg",
            "t_dg",
            "measure",
            "reset",
            "id",
            "p",
            "sx",
            "sx_dg",
        ]

        cirq_op_qubits = [qubit_map[q] for q in op_qubits_i]

        if (op_i not in cirq_gates_filter) and instr.op.definition:
            new_circ = instr.op.definition
            cirq_circuit.append(convert_to_cirq(new_circ, cirq_op_qubits))

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
                cirq_circuit.append(
                    ZPowGate(exponent=exp_param / np.pi)(*cirq_op_qubits)
                )

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

        else:
            # for simple single qubit gates
            if op_i == "sx":
                cirq_circuit.append(XPowGate(exponent=0.5)(*cirq_op_qubits))
            elif op_i == "sx_dg":
                cirq_circuit.append(inverse(XPowGate(exponent=0.5)(*cirq_op_qubits)))
            elif cirq_gate:
                cirq_circuit.append(cirq_gate(*cirq_op_qubits))

    return cirq_circuit
