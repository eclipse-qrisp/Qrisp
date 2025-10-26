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

from qrisp.circuit import ControlledOperation, QuantumCircuit, Operation

from qrisp import QuantumSession

"""
TODO:

-- Implement complex parameters handling for gates

"""

QRISP_PL_OP_MAP = {
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
    "x": qml.PauliX,
    "y": qml.PauliY,
    "z": qml.PauliZ,
    "sx": qml.SX,
    "sx_dg": qml.SX,
    "s": qml.S,
    "s_dg": qml.S,
    "t": qml.T,
    "t_dg": qml.T,
    "cx": qml.CNOT,
    "cy": qml.CY,
    "cz": qml.CZ,
    "cp": qml.ControlledPhaseShift,
}


def _create_qml_instruction(op: Operation) -> qml.operation.Operator:
    """Create a PennyLane instruction from a Qrisp operation."""

    if op.name in QRISP_PL_OP_MAP:
        return QRISP_PL_OP_MAP[op.name]

    # complex gate, with subcircuit definition we create a representative subcircuit
    # TODO: this is only used so far for MCXGate and MCRXGate
    # if op.definition is not None:
    #     return qml_converter(op.definition)

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
            qml_wires = [qubits[qubit.identifier] for qubit in data.qubits]

            if op.name in ["qb_alloc", "qb_dealloc"]:
                continue

            if op.name == "measure":
                measurelist.append(*qml_wires)
                continue

            #########################
            ### Parameter Handling
            #########################

            # If the operation has symbolic parameters, substitute them with the provided arguments.
            # Otherwise, convert them to float64 (with float128 there is a bug in PennyLane)
            for i, param in enumerate(list(op.params)):
                if isinstance(param, sympy.Symbol):
                    qml_params[i] = args[symbols.index(param)]
                else:
                    qml_params[i] = np.float64(param)

            qml_op_instance = _create_qml_instruction(op)

            # n_controls = 0
            # if isinstance(op, ControlledOperation):
            #     n_controls = len(op.ctrl_state)

            # controls = qml_wires[:n_controls]
            # targets = qml_wires[n_controls:]

            with qml.QueuingManager.stop_recording():

                qml_op = qml_op_instance(*qml_params, wires=qml_wires)

                if "_dg" in op.name:
                    qml_op = qml.adjoint(qml_op)

                # if isinstance(op, ControlledOperation):
                #     qml_op = qml.ctrl(
                #         op=qml_op,
                #         control=controls,
                #         control_values=[int(bit) for bit in op.ctrl_state],
                #     )

            ######################
            ### Apply Operation
            ######################

            qml.apply(qml_op)

        if len(measurelist) > 0:
            return qml.counts(wires=measurelist)

    return circuit
