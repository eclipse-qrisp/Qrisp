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
from dataclasses import dataclass
from typing import Callable, Optional

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


@dataclass
class QMLGateDescriptor:
    """Descriptor for mapping Qrisp operations to PennyLane gates."""

    gate_class: Callable
    param_fn: Callable = lambda op: op.params


QRISP_PL_BASE_MAP = {
    "x": QMLGateDescriptor(qml.X),
    "y": QMLGateDescriptor(qml.Y),
    "z": QMLGateDescriptor(qml.Z),
    "h": QMLGateDescriptor(qml.Hadamard),
    "rx": QMLGateDescriptor(qml.RX),
    "ry": QMLGateDescriptor(qml.RY),
    "rz": QMLGateDescriptor(qml.RZ),
    "p": QMLGateDescriptor(qml.PhaseShift),
    "u1": QMLGateDescriptor(qml.RZ),
    "u3": QMLGateDescriptor(qml.U3),
    "rxx": QMLGateDescriptor(qml.IsingXX),
    "ryy": QMLGateDescriptor(qml.IsingYY),
    "rzz": QMLGateDescriptor(qml.IsingZZ),
    # qml.IsingXY(phi, ...) is equivalent to XXYYGate(phi, beta=np.pi)
    "swap": QMLGateDescriptor(qml.SWAP),
    "s": QMLGateDescriptor(qml.S),
    "t": QMLGateDescriptor(qml.T),
    "id": QMLGateDescriptor(qml.Identity, param_fn=lambda _: []),
    "sx": QMLGateDescriptor(
        qml.RX, param_fn=lambda _: [np.pi / 2]
    ),  # SXGate() -> qml.RX(pi/2)
    "gphase": QMLGateDescriptor(
        qml.GlobalPhase, param_fn=lambda op: [-op.params[0]]
    ),  # GPhaseGate(phi) -> qml.GlobalPhase(-phi)
    "r": QMLGateDescriptor(
        qml.Rot, param_fn=lambda op: [-op.params[1], -op.params[0], op.params[1]]
    ),  # RGate(theta, phi) -> qml.Rot(-phi, -theta, phi)
    "measure": QMLGateDescriptor(qml.measurements.MidMeasureMP),
}


def _extract_name(name: str) -> tuple[str, bool]:
    """Extract the operation name and check if it's inverted."""
    is_inverse = name.endswith("_dg")
    name = name[:-3] if is_inverse else name
    return name, is_inverse


def _create_qml_instruction(op: Operation) -> tuple[QMLGateDescriptor, bool, bool]:
    """
    Create a PennyLane instruction from a Qrisp operation.

    Parameters
    ----------
    op : Operation
        The Qrisp operation to convert.

    Returns
    -------
    tuple[QMLGateDescriptor, bool, bool]
        A tuple containing the QMLGateDescriptor corresponding to the Qrisp operation,
        a boolean indicating if the operation is inverted, and a boolean indicating
        if the operation is controlled. For controlled Qrisp operations, it
        returns the base QMLGateDescriptor.

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


def _is_nested_circuit(op: Operation) -> bool:
    """Return True if this Qrisp operation is a subcircuit."""
    return (
        op.definition is not None
        and not isinstance(op, ControlledOperation)
        and op.name not in QRISP_PL_BASE_MAP
    )


def _evaluate_abstract_params(params: list, subs_dic: dict) -> list:
    """Return parameters with substitution for abstract parameters."""
    out = []
    for p in params:
        if hasattr(p, "subs"):
            p = p.subs(subs_dic)
        out.append(np.float64(p))
    return out


def _process_qrisp_circuit(
    qc: QuantumCircuit | QuantumSession, wire_map: dict, subs_dic: dict | None
) -> None:
    """Recursively process a Qrisp circuit into PennyLane operations."""

    for data in qc.data:
        op = data.op

        if op.name in {"qb_alloc", "qb_dealloc"}:
            continue

        if _is_nested_circuit(op):
            sub_qc = op.definition
            sub_qubits = data.qubits
            sub_wire_map = {
                inner.identifier: wire_map[outer.identifier]
                for inner, outer in zip(sub_qc.qubits, sub_qubits)
            }
            _process_qrisp_circuit(sub_qc, sub_wire_map, subs_dic)
            continue

        qml_gate_desc, is_inverse, is_controlled = _create_qml_instruction(op)
        qml_wires = [wire_map[qb.identifier] for qb in data.qubits]

        n_ctrls = len(op.ctrl_state) if is_controlled else 0
        controls, targets = qml_wires[:n_ctrls], qml_wires[n_ctrls:]

        qml_params = qml_gate_desc.param_fn(op)

        if subs_dic is not None:
            qml_params = _evaluate_abstract_params(qml_params, subs_dic)

        if qml_gate_desc.gate_class is qml.measurements.MidMeasureMP:
            qml.measure(wires=targets)
            continue

        with qml.QueuingManager.stop_recording():

            qml_op = qml_gate_desc.gate_class(*qml_params, wires=targets)

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

    def circuit(
        wires: Optional[qml.wires.WiresLike] = None, subs_dic: Optional[dict] = None
    ) -> None:
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
