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

import types
import uuid

import numpy as np
import pennylane as qml

from qrisp import QuantumSession
from qrisp.circuit import ControlledOperation, Operation, QuantumCircuit

"""
TODO:

-- Implement all gates listed in `scr/qrisp/circuit/standard_operations.py`
-- Add support for control-flow operations (if, while, for)
-- Implement complex parameters handling for gates that support them.

"""

QRISP_PL_BASE_MAP = {
    "x": qml.X,
    "y": qml.Y,
    "z": qml.Z,
    "h": qml.Hadamard,
    "rx": qml.RX,
    "ry": qml.RY,
    "rz": qml.RZ,
    "p": qml.PhaseShift,
    "u1": qml.U1,
    "u3": qml.U3,
    "rxx": qml.IsingXX,
    "ryy": qml.IsingYY,
    "rzz": qml.IsingZZ,
    "xxyy": qml.IsingXY,
    "swap": qml.SWAP,
    "sx": qml.SX,
    "s": qml.S,
    "t": qml.T,
    "gphase": qml.GlobalPhase,
    "id": qml.Identity,
    "r": qml.Rot,
    "measure": qml.measurements.MidMeasureMP,
}


def _extract_name(name: str) -> tuple[str, bool]:
    """Extract the operation name and check if it's inverted."""
    is_inverse = name.endswith("_dg")
    name = name[:-3] if is_inverse else name
    return name, is_inverse


def _create_qml_instruction(op: Operation) -> tuple[qml.operation.Operator, bool, bool]:
    """
    Create a PennyLane instruction from a Qrisp operation.

    Parameters
    ----------
    op : Operation
        The Qrisp operation to convert.

    Returns
    -------
    tuple[qml.operation.Operator, bool, bool]
        A tuple containing the PennyLane operator class corresponding
        to the Qrisp operation, a boolean indicating if the operation
        is inverted, and a boolean indicating if the operation is controlled.
        For controlled Qrisp operations, it returns the base PennyLane operator class.

    """

    name, is_inverse = _extract_name(op.name)

    if name in QRISP_PL_BASE_MAP:
        return QRISP_PL_BASE_MAP[name], is_inverse, False

    if isinstance(op, ControlledOperation):
        base_name, is_inverse = _extract_name(op.base_operation.name)
        if base_name in QRISP_PL_BASE_MAP:
            return QRISP_PL_BASE_MAP[base_name], is_inverse, True

    raise NotImplementedError(
        f"Operation '{op.name}' is not supported in the PennyLane converter."
    )


def _process_qrisp_circuit(
    qc: QuantumCircuit, wire_map: dict[str, qml.wires.WiresLike], subs_dic: dict = None
) -> None:
    """Recursively process a Qrisp quantum circuit and apply PennyLane operations.

    Parameters
    ----------
    qc : QuantumCircuit
        The Qrisp quantum circuit to be processed.

    wire_map : dict[str, qml.wires.WiresLike]
        A mapping from Qrisp qubit identifiers to PennyLane wires.

    subs_dic : dict, optional
        A dictionary for substituting abstract parameters with concrete values.
    """

    for data in qc.data:
        op = data.op

        if op.name in {"qb_alloc", "qb_dealloc"}:
            continue

        # Can this check be improved?
        if (
            op.definition is not None
            and not isinstance(op, ControlledOperation)
            and op.name not in QRISP_PL_BASE_MAP
        ):
            sub_qc = op.definition
            sub_qubits = data.qubits
            sub_wire_map = {
                qb.identifier: wire_map[outer.identifier]
                for qb, outer in zip(sub_qc.qubits, sub_qubits)
            }
            _process_qrisp_circuit(sub_qc, sub_wire_map)
            continue

        qml_op_class, is_inverse, is_controlled = _create_qml_instruction(op)
        qml_wires = [wire_map[qb.identifier] for qb in data.qubits]

        n_ctrls = len(op.ctrl_state) if is_controlled else 0
        controls, targets = qml_wires[:n_ctrls], qml_wires[n_ctrls:]

        qml_params = []
        for param in op.params:
            if subs_dic is not None and hasattr(param, "subs"):
                qml_params.append(np.float64(param.subs(subs_dic)))
            else:
                qml_params.append(np.float64(param))

        # TODO: Find a more elegant way to handle special cases
        if qml_op_class.__name__ == "Identity":
            qml_params = []

        if qml_op_class.__name__ == "MidMeasureMP":
            mp = qml.measurements.MidMeasureMP(wires=targets, id=str(uuid.uuid4()))
            qml.measurements.MeasurementValue([mp])
            continue

        with qml.QueuingManager.stop_recording():

            qml_op = qml_op_class(*qml_params, wires=targets)

            if is_inverse:
                qml_op = qml.adjoint(qml_op)

            if is_controlled:
                qml_op = qml.ctrl(
                    op=qml_op,
                    control=controls,
                    control_values=[int(bit) for bit in op.ctrl_state],
                )

        qml.apply(qml_op)


def qml_converter(qc: QuantumCircuit | QuantumSession) -> types.FunctionType:
    """
    Convert a Qrisp quantum circuit to a PennyLane quantum function.

    Parameters
    ----------
    qc : QuantumCircuit | QuantumSession
        The Qrisp quantum circuit to be converted.

    Returns
    -------
    types.FunctionType
        A PennyLane quantum function reproducing the Qrisp circuit.

    """

    def circuit(wires: qml.wires.WiresLike = None, subs_dic: dict = None) -> None:
        """
        PennyLane quantum function representing the Qrisp circuit.

        This function should be used within a PennyLane QNode context.

        Parameters
        ----------
        wires : qml.wires.WiresLike, optional
            The wires to be used in the PennyLane circuit. If None, the qubit identifiers
            from the Qrisp circuit are used.

        subs_dic : dict, optional
            A dictionary for substituting abstract parameters with concrete values.

        """

        if wires is None:
            wire_map = {qb.identifier: qb.identifier for qb in qc.qubits}
        else:
            wire_map = {qb.identifier: wires[i] for i, qb in enumerate(qc.qubits)}

        _process_qrisp_circuit(qc, wire_map, subs_dic)

    return circuit
