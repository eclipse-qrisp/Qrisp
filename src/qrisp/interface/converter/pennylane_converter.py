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

import re
import types

import numpy as np
import pennylane as qml
import sympy

from qrisp import QuantumSession
from qrisp.circuit import ControlledOperation, Operation, QuantumCircuit

"""
TODO:

-- Implement complex parameters handling for gates
-- Implement controlled operations properly
-- Implement all gates listed in `scr/qrisp/circuit/standard_operations.py`

"""
# Mapping for basic operations
QRISP_PL_OP_MAP = {
    "x": qml.X,
    "y": qml.Y,
    "z": qml.Z,
    "rxx": qml.IsingXX,
    "ryy": qml.IsingYY,
    "rzz": qml.IsingZZ,
    "p": qml.RZ,
    "rx": qml.RX,
    "ry": qml.RY,
    "rz": qml.RZ,
    "u1": qml.U1,
    "u3": qml.U3,
    "xxyy": qml.IsingXY,
    "swap": qml.SWAP,
    "h": qml.Hadamard,
    "sx": qml.SX,
    "sx_dg": qml.SX,
    "s": qml.S,
    "s_dg": qml.S,
    "t": qml.T,
    "t_dg": qml.T,
}

# Mapping for controlled operations

# TODO: handle controlled gates such as 'ccx', 'cccx', etc.

# To avoid different naming conventions between Qrisp and PennyLane,
# we map qrisp controlled gate names to their PennyLane base gate counterparts.
# Then, we use `qml.ctrl` to create the controlled versions in the main conversion function.
QRISP_PL_OP_MAP_CTRL = {
    "cx": qml.X,
    "cy": qml.Y,
    "cz": qml.Z,
    "cp": qml.PhaseShift,
    "crx": qml.RX,
    "cry": qml.RY,
    "crz": qml.RZ,
    "mcrx": qml.RX,
    "csx": qml.SX,
    "csx_dg": qml.SX,
}


def _create_qml_instruction(op: Operation) -> qml.operation.Operator:
    """Create a PennyLane instruction from a Qrisp operation."""

    if op.name in QRISP_PL_OP_MAP:
        return QRISP_PL_OP_MAP[op.name]

    if op.name in QRISP_PL_OP_MAP_CTRL:
        return QRISP_PL_OP_MAP_CTRL[op.name]

    # Handle custom multi-controlled gates, e.g., '2cx', '3cz', '3csx_dg', etc.
    gate_match = re.fullmatch(r"(\d+)(c[a-z_]+)", op.name)
    if gate_match:
        ctrl_gate_name = gate_match.group(2)
        if ctrl_gate_name in QRISP_PL_OP_MAP_CTRL:
            return QRISP_PL_OP_MAP_CTRL[ctrl_gate_name]

    raise NotImplementedError(
        f"Operation {op.name} not implemented in PennyLane converter."
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
        A PennyLane quantum function representing the Qrisp circuit.

    """

    def circuit(*args, wires: qml.wires.WiresLike = None):
        """
        PennyLane quantum function representing the Qrisp circuit.

        Parameters
        ----------
        *args : list
            The parameters for the quantum circuit.

        wires : qml.wires.WiresLike, optional
            The wires to be used in the PennyLane circuit. If None, the qubit identifiers
            from the Qrisp circuit are used.

        Returns
        -------
        qml.counts or None
            The measurement results if measurements are present in the circuit, otherwise None.

        """

        qubits = {}
        measurelist = []
        symbols = list(qc.abstract_params)

        if wires is None:
            for qubit in qc.qubits:
                qubits[qubit.identifier] = qubit.identifier
        else:
            for i, qubit in enumerate(qc.qubits):
                qubits[qubit.identifier] = wires[i]

        for data in qc.data:

            op = data.op
            qml_params = list(op.params)

            # data.qubits = [controls..., targets...]
            qml_wires = [qubits[qubit.identifier] for qubit in data.qubits]
            n_ctrls = len(op.ctrl_state) if isinstance(op, ControlledOperation) else 0
            controls = qml_wires[:n_ctrls]
            targets = qml_wires[n_ctrls:]

            if op.name in ["qb_alloc", "qb_dealloc"]:
                continue

            if op.name == "measure":
                measurelist.append(*qml_wires)
                continue

            # If the operation has symbolic parameters, substitute them with the provided arguments.
            # Otherwise, convert them to float64 (with float128 there is a bug in PennyLane)
            for i, param in enumerate(list(op.params)):
                if isinstance(param, sympy.Symbol):
                    qml_params[i] = args[symbols.index(param)]
                else:
                    qml_params[i] = np.float64(param)

            # if the operation is controlled, this returns the PL base operation
            qml_op_class = _create_qml_instruction(op)

            with qml.QueuingManager.stop_recording():

                qml_op = qml_op_class(*qml_params, wires=targets)

                if "_dg" in op.name:
                    qml_op = qml.adjoint(qml_op)

                if n_ctrls:
                    qml_op = qml.ctrl(
                        op=qml_op,
                        control=controls,
                        control_values=[int(bit) for bit in op.ctrl_state],
                    )

            qml.apply(qml_op)

        if len(measurelist) > 0:
            return qml.counts(wires=measurelist)

    return circuit
