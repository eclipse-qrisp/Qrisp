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

from qrisp.circuit import ControlledOperation
import numpy as np


def convert_to_cirq(qrisp_circuit, cirq_qubits=None):
    """Convert a Qrisp QuantumCircuit to a Cirq Circuit.

    Args:
        qrisp_circuit: The Qrisp QuantumCircuit to convert.
        cirq_qubits: Optional list of Cirq qubits to map to. If None,
            LineQubits are created automatically.

    Returns:
        A cirq.Circuit equivalent to the input Qrisp circuit.

    Raises:
        ImportError: If Cirq is not installed.
        ValueError: If a gate is not supported by the converter.
    """
    try:
        from cirq import Circuit, LineQubit
    except (ModuleNotFoundError, ImportError):
        raise ImportError(
            "Cirq must be installed to be able to use the Qrisp to Cirq converter."
        )

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
        GlobalPhaseGate,
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


def convert_from_cirq(cirq_circuit):
    """Convert a Cirq Circuit to a Qrisp QuantumCircuit.

    Args:
        cirq_circuit: The Cirq Circuit to convert.

    Returns:
        A Qrisp QuantumCircuit equivalent to the input Cirq circuit.

    Raises:
        ImportError: If Cirq is not installed.
        ValueError: If a gate is not supported by the converter.
    """
    try:
        import cirq
    except (ModuleNotFoundError, ImportError):
        raise ImportError(
            "Cirq must be installed to be able to use the Cirq to Qrisp converter."
        )

    import numpy as np
    from qrisp import QuantumCircuit
    from qrisp.circuit import ControlledOperation
    from qrisp.circuit import standard_operations as ops

    qc = QuantumCircuit(len(cirq_circuit.all_qubits()))

    cirq_qubits = sorted(cirq_circuit.all_qubits())
    qubit_map = {q: qc.qubits[i] for i, q in enumerate(cirq_qubits)}

    for op in cirq_circuit.all_operations():
        # Extract the gate, unwrapping ControlledOperation if present
        extra_controls = None
        if isinstance(op, cirq.ControlledOperation):
            sub_op = op.sub_operation
            gate = getattr(sub_op, "gate", None)
            if gate is None:
                raise ValueError(
                    f"Controlled sub-operation {sub_op} is not supported "
                    "by the Cirq to Qrisp converter."
                )
            extra_controls = (list(op.controls), list(sub_op.qubits))
        else:
            gate = getattr(op, "gate", None)
            if gate is None:
                raise ValueError(
                    f"Operation {op} without gate attribute is not supported "
                    "by the Cirq to Qrisp converter."
                )

        # Global phase in Cirq acts on 0 qubits; Qrisp requires a qubit
        # argument, so map it to the first available qubit.  Handled
        # outside the dict because of this special qubit-mapping logic.
        if isinstance(gate, cirq.GlobalPhaseGate):
            if cirq_qubits:
                coeff = gate.coefficient
                phi = np.angle(coeff)
                qrisp_op = ops.GPhaseGate(phi)
                qc.append(qrisp_op, [qubit_map[cirq_qubits[0]]])
            continue

        # Measurement uses the circuit's measure() method for automatic
        # classical-bit allocation.  Cannot go through the gate dict.
        if isinstance(gate, cirq.MeasurementGate):
            qrisp_qubits = [qubit_map[q] for q in op.qubits]
            if len(qrisp_qubits) == 1:
                qc.measure(qrisp_qubits[0])
            else:
                qc.measure(qrisp_qubits)
            continue

        # Unwrap nested ControlledGate layers (e.g. cirq.X.controlled()).
        # Cannot live in the dict because it needs to recursively peel
        # control layers off before converting the innermost gate.
        ctrl_layers = []
        inner_gate = gate
        while isinstance(inner_gate, cirq.ControlledGate):
            ctrl_values = inner_gate.control_values
            if ctrl_values is not None:
                ctrl_state = "".join(str(int(v)) for v in ctrl_values)
            else:
                ctrl_state = -1
            ctrl_layers.append((inner_gate.num_controls(), ctrl_state))
            inner_gate = inner_gate.sub_gate

        # Dict lookup by exact type for gates whose Qrisp equivalent is
        # fixed or follows a simple (attr, multiplier) pattern.
        cirq_qrisp_gate_map = {
            cirq.HPowGate: ops.HGate(),
            cirq.CXPowGate: ops.CXGate(),
            cirq.CZPowGate: ops.CZGate(),
            cirq.SwapPowGate: ops.SwapGate(),
            cirq.IdentityGate: ops.IDGate(),
            cirq.ResetChannel: ops.Reset(),
            cirq.MeasurementGate: ops.Measurement(),
            cirq.Rx: (ops.RXGate, "exponent", np.pi),
            cirq.Ry: (ops.RYGate, "exponent", np.pi),
            cirq.Rz: (ops.RZGate, "exponent", np.pi),
            cirq.GlobalPhaseGate: (ops.GPhaseGate, "coefficient", None),
        }

        converter = cirq_qrisp_gate_map.get(type(inner_gate))
        if converter is not None:
            if isinstance(converter, tuple):
                op_class, attr, mult = converter
                value = getattr(inner_gate, attr)
                qrisp_op = op_class(
                    np.angle(value) if mult is None else value * mult
                )
            else:
                qrisp_op = converter

        # XPowGate uses isinstance to catch cirq.X (type _PauliX, a
        # subclass).  The exponent selects X, SX, SX†, or generic RX.
        elif isinstance(inner_gate, cirq.XPowGate):
            exp = inner_gate.exponent
            if np.isclose(exp % 4, 1.0) or np.isclose(exp % 4, -3.0):
                qrisp_op = ops.XGate()
            elif np.isclose(exp % 4, 0.5) or np.isclose(exp % 4, -3.5):
                qrisp_op = ops.SXGate()
            elif np.isclose(exp % 4, -0.5) or np.isclose(exp % 4, 3.5):
                qrisp_op = ops.SXDGGate()
            else:
                qrisp_op = ops.RXGate(exp * np.pi)

        # YPowGate: same subclass issue with cirq.Y (type _PauliY).
        elif isinstance(inner_gate, cirq.YPowGate):
            exp = inner_gate.exponent
            if np.isclose(exp % 4, 1.0) or np.isclose(exp % 4, -3.0):
                qrisp_op = ops.YGate()
            else:
                qrisp_op = ops.RYGate(exp * np.pi)

        # ZPowGate: catches cirq.Z (type _PauliZ) and exponent-based
        # variants like S (exp=0.5), T (exp=0.25), etc.
        elif isinstance(inner_gate, cirq.ZPowGate):
            exp = inner_gate.exponent
            if np.isclose(exp % 4, 1.0) or np.isclose(exp % 4, -3.0):
                qrisp_op = ops.ZGate()
            elif np.isclose(exp % 4, 0.5) or np.isclose(exp % 4, -3.5):
                qrisp_op = ops.SGate()
            elif np.isclose(exp % 4, -0.5) or np.isclose(exp % 4, 3.5):
                qrisp_op = ops.SGate().inverse()
            elif np.isclose(exp % 4, 0.25) or np.isclose(exp % 4, -3.75):
                qrisp_op = ops.TGate()
            elif np.isclose(exp % 4, -0.25) or np.isclose(exp % 4, 3.75):
                qrisp_op = ops.TGate().inverse()
            else:
                qrisp_op = ops.PGate(exp * np.pi)
        else:
            raise ValueError(
                f"Gate {gate} is not supported by the Cirq to Qrisp converter."
            )

        # Wrap any ControlledGate layers (inside-out)
        for num_ctrl, ctrl_state in reversed(ctrl_layers):
            qrisp_op = ControlledOperation(
                base_operation=qrisp_op,
                num_ctrl_qubits=num_ctrl,
                ctrl_state=ctrl_state,
            )

        # Wrap any outer ControlledOperation and append
        if extra_controls is not None:
            controls, sub_qubits = extra_controls
            qrisp_op = ControlledOperation(
                base_operation=qrisp_op,
                num_ctrl_qubits=len(controls),
            )
            qrisp_qubits = [qubit_map[q] for q in controls + sub_qubits]
        else:
            qrisp_qubits = [qubit_map[q] for q in op.qubits]

        qc.append(qrisp_op, qrisp_qubits)

    return qc



