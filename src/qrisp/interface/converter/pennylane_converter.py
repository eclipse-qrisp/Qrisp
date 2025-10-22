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

from qrisp.circuit import ControlledOperation

"""
TODO:

-- Change the structure of the mapping
-- Update mapping with more gates (ideally all the currently supported Qrisp gates)
-- Add support for all gates in the mapping
-- Polish the code with pylint/black standards
-- Improve efficiency if possible

-- clbit interaction with measurements 
-- conditional environment, i.e. mid-circuit measurement is possible, have to create this
    --> otherwise no clbit interaction?

"""

# mapping from Qrisp operation names to PennyLane gates
mapping = {
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
}


def create_qml_instruction(op):
    """Create a PennyLane instruction from a Qrisp operation."""

    if op.name in mapping:
        qml_ins = mapping[op.name]

    # complex gate, with subcircuit definition we create a representative subcircuit
    elif op.definition:
        return qml_converter(op.definition, circ=True)

    else:
        NotImplementedError(
            f"Operation {op.name} not implemented in PennyLane converter."
        )

    return qml_ins


def qml_converter(qc, circ=False):
    """Convert a Qrisp quantum circuit to a PennyLane quantum function."""

    def circuit(*args, wires=None):

        circ_flag = False
        qbit_dic = {}

        # Map Qrisp qubit identifiers to PennyLane wires
        if wires is None:
            for qubit in qc.qubits:
                qbit_dic[qubit.identifier] = qubit.identifier
        else:
            for i in range(len(qc.qubits)):
                qbit_dic[qc.qubits[i].identifier] = wires[i]

        # Save measurements
        measurelist = []
        symbols = list(qc.abstract_params)

        for index in range(len(qc.data)):

            op = qc.data[index].op
            params = list(op.params)
            for idx, param in enumerate(params):
                if isinstance(param, sympy.Symbol):
                    params[idx] = args[symbols.index(param)]
                else:
                    params[idx] = np.float64(param)

            qubit_list = [qbit_dic[qubit.identifier] for qubit in qc.data[index].qubits]
            op_qubits = qc.data[index].qubits

            if op.name in ["qb_alloc", "qb_dealloc"]:
                continue

            if op.name == "measure":
                ### below: possible implementation for nested circuit measurments
                # if circ == True:
                #   Exception("Measurements in nested circuits not supported (yet)")
                measurelist.append(*qubit_list)
                continue

            ###### BugCatcher + special gates ######
            if op.name in ["sx", "sx_dg", "id", "t", "t_dg", "s", "s_dg"]:
                # bugged -> params empty
                params = []
            if op.name == "xxyy":
                # deffed as RXXYY with angle pi
                params = [np.pi]

            if op.name == "cx":
                # maybe adjustment necessary here
                if hasattr(op, "ctrl_state"):
                    qml_ins = qml.CNOT([qubit_list[0], qubit_list[1]])
                else:
                    qml_ins = qml.CNOT([qubit_list[0], qubit_list[1]])

            elif op.name == "cy":
                if hasattr(op, "ctrl_state"):
                    qml_ins = qml.CY([qubit_list[0], qubit_list[1]])
                else:
                    qml_ins = qml.CY([qubit_list[0], qubit_list[1]])

            elif op.name == "cz":
                if hasattr(op, "ctrl_state"):
                    qml_ins = qml.CZ([qubit_list[0], qubit_list[1]])
                else:
                    qml_ins = qml.CZ([qubit_list[0], qubit_list[1]])

            elif op.name == "cp":
                if hasattr(op, "ctrl_state"):
                    qml_ins = qml.ControlledPhaseShift(
                        *params, [qubit_list[0], qubit_list[1]]
                    )
                else:
                    qml_ins = qml.ControlledPhaseShift(
                        *params, [qubit_list[0], qubit_list[1]]
                    )

            # return the controlled instruction or the nested circuit
            elif isinstance(op, ControlledOperation):
                qml_trafo = create_qml_instruction(op)
                if isinstance(qml_trafo, types.FunctionType):
                    circ_flag = True
                    wire_map = {}
                    for i in range(len(op_qubits)):
                        # wire_map[i] = op_qubits[i].identifier
                        wire_map[op.definition.qubits[i].identifier] = op_qubits[
                            i
                        ].identifier
                    mapped_quantum_function = qml.map_wires(qml_trafo, wire_map)
                    qml_ins = qml.ctrl(
                        mapped_quantum_function, control=op.ctrl_state[::-1]
                    )
                else:
                    qml_ins = qml.ctrl(qml_trafo, control=op.ctrl_state[::-1])

            # return the instruction or the nested circuit
            else:
                qml_trafo = create_qml_instruction(op)
                if isinstance(qml_trafo, types.FunctionType):
                    circ_flag = True
                    wire_map = {}
                    for idx, qubit in enumerate(op_qubits):
                        # map the wires as given for the nested circuit
                        wire_map[op.definition.qubits[idx].identifier] = (
                            qubit.identifier
                        )
                    mapped_quantum_function = qml.map_wires(qml_trafo, wire_map)

                else:
                    if "_dg" in op.name:
                        qml_ins = qml.adjoint(qml_trafo(*params, wires=qubit_list))
                    else:
                        qml_ins = qml_trafo(*params, wires=qubit_list)

            if not isinstance(circ, bool):
                circ()
            elif not circ_flag:
                qml_ins
            else:
                mapped_quantum_function()

        # add measurements at the end, i.e. return a count object
        if len(measurelist) > 0:
            return qml.counts(wires=measurelist)

    return circuit
