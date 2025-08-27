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

from qrisp.circuit import ControlledOperation

"""
TODO: 
-- clbit interaction with measurements 
-- conditional environment, i.e. mid-circuit measurement is possible, have to create this
    --> otherwise no clbit interaction?

BUGS:

"""


def create_qml_instruction(op):
    import pennylane as qml

    # create basic, straight forward instructions
    if op.name == "rxx":
        qml_ins = qml.IsingXX
    elif op.name == "ryy":
        qml_ins = qml.IsingYY
    elif op.name == "rzz":
        qml_ins = qml.IsingZZ

    elif op.name == "p":
        qml_ins = qml.RZ
    elif op.name == "rx":
        qml_ins = qml.RX
    elif op.name == "ry":
        qml_ins = qml.RY
    elif op.name == "rz":
        qml_ins = qml.RZ
    elif op.name == "u1":
        qml_ins = qml.U1

    elif op.name == "u3":
        qml_ins = qml.U3
    elif op.name == "xxyy":
        qml_ins = qml.IsingXY
    elif op.name == "swap":
        qml_ins = qml.SWAP
    elif op.name == "h":
        qml_ins = qml.Hadamard
    elif op.name == "x":
        qml_ins = qml.PauliX
    elif op.name == "y":
        qml_ins = qml.PauliY
    elif op.name == "z":
        qml_ins = qml.PauliZ
    elif op.name == "s" or op.name == "s_dg":
        qml_ins = qml.S
    elif op.name == "t" or op.name == "t_dg":
        qml_ins = qml.T
    elif op.name == "sx" or op.name == "sx_dg":
        qml_ins = qml.SX

    # complex gate, with subcircuit definition we create a representative subcircuit
    elif op.definition:
        circ = qml_converter(op.definition, circ=True)
        return circ
    else:
        Exception("Cannot convert")
    return qml_ins


import sympy


def qml_converter(qc, circ=False):
    import pennylane as qml

    def circuit(*args, wires=None):

        circFlag = False

        qbit_dic = {}
        # add qubits
        if wires is None:
            for qubit in qc.qubits:
                qbit_dic[qubit.identifier] = qubit.identifier
        else:
            for i in range(len(qc.qubits)):
                qbit_dic[qc.qubits[i].identifier] = wires[i]

        # save measurements
        measurelist = []
        symbols = list(qc.abstract_params)

        for index in range(len(qc.data)):

            op = qc.data[index].op
            params = list(op.params)
            for i in range(len(params)):
                if isinstance(params[i], sympy.Symbol):
                    params[i] = args[symbols.index(params[i])]

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
                    circFlag = True
                    wire_map = dict()
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
                    circFlag = True
                    wire_map = dict()
                    for i in range(len(op_qubits)):
                        # map the wires as given for the nested circuit
                        wire_map[op.definition.qubits[i].identifier] = op_qubits[
                            i
                        ].identifier
                    mapped_quantum_function = qml.map_wires(qml_trafo, wire_map)

                else:
                    if "_dg" in op.name:
                        qml_ins = qml.adjoint(qml_trafo(*params, wires=qubit_list))
                    else:
                        qml_ins = qml_trafo(*params, wires=qubit_list)

            if not isinstance(circ, bool):
                circ()
            elif not circFlag:
                qml_ins
            else:
                mapped_quantum_function()

        # add measurements at the end, i.e. return a count object
        if len(measurelist):
            return qml.counts(wires=measurelist)

    return circuit
