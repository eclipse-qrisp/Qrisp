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

import numpy as np
import pennylane as qml
import sympy

from qrisp import QuantumSession
from qrisp.circuit import ControlledOperation, Operation, QuantumCircuit

"""
TODO:

-- Implement complex parameters handling for gates that support them.
-- Implement all gates listed in `scr/qrisp/circuit/standard_operations.py`

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

    symbols = list(qc.abstract_params)

    def circuit(
        *args, wires: qml.wires.WiresLike = None
    ) -> qml.measurements.CountsMP | None:
        """
        PennyLane quantum function representing the Qrisp circuit.

        This function should be used within a PennyLane QNode context.

        Parameters
        ----------
        *args : list
            Parameter values corresponding to Qrisp symbolic parameters.

        wires : qml.wires.WiresLike, optional
            The wires to be used in the PennyLane circuit. If None, the qubit identifiers
            from the Qrisp circuit are used.

        Returns
        -------
        qml.measurements.CountsMP or None
            The measurement results if measurements are present in the circuit, otherwise None.

        """

        if wires is None:
            wire_map = {qb.identifier: qb.identifier for qb in qc.qubits}
        else:
            wire_map = {qb.identifier: wires[i] for i, qb in enumerate(qc.qubits)}

        measure_wires = []

        for data in qc.data:

            op = data.op

            if op.name in {"qb_alloc", "qb_dealloc"}:
                continue

            if op.name == "measure":
                measure_wires.extend([wire_map[qb.identifier] for qb in data.qubits])
                continue

            qml_op_class, is_inverse, is_controlled = _create_qml_instruction(op)

            qml_wires = [wire_map[qb.identifier] for qb in data.qubits]
            n_ctrls = len(op.ctrl_state) if is_controlled else 0
            controls, targets = qml_wires[:n_ctrls], qml_wires[n_ctrls:]

            # If the operation has symbolic parameters, substitute them with the provided arguments.
            # Otherwise, convert them to float64 (with float128 there is a bug in PennyLane)
            qml_params = []
            for param in op.params:
                if isinstance(param, sympy.Symbol):
                    try:
                        # TODO: precomputing this in a dictionary
                        # might be more efficient (O(1) lookup instead of O(n))
                        idx = symbols.index(param)
                    except ValueError as exc:
                        raise ValueError(f"Unrecognized symbol: {param}") from exc
                    qml_params.append(np.float64(args[idx]))
                else:
                    qml_params.append(np.float64(param))

            # Special case for Identity gate
            if qml_op_class.__name__ == "Identity":
                qml_params = []

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

        if measure_wires:
            return qml.counts(wires=measure_wires)

        return None

    return circuit
