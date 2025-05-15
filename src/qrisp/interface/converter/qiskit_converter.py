"""
\********************************************************************************
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
********************************************************************************/
"""

import numpy as np
from sympy import lambdify, Expr

from qrisp.misc import bin_rep
from qrisp.circuit.standard_operations import op_list
from qrisp.circuit import ControlledOperation, ClControlledOperation


# Function to convert qrisp quantum circuits to Qiskit quantum circuits
def convert_to_qiskit(qc, transpile=False):
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit import Parameter

    if transpile:
        from qrisp.circuit.transpiler import transpile

        qc = transpile(qc)

    # Create circuit

    qiskit_qc = QuantumCircuit()

    # This dic gives the qiskit qubits/clbits when presented with their identifier
    bit_dic = {}

    # Add Qubits
    for i in range(len(qc.qubits)):
        qiskit_qc.add_register(QuantumRegister(1, name=qc.qubits[i].identifier))

        bit_dic[qc.qubits[i].identifier] = qiskit_qc.qubits[-1]

    # Add Clbits
    for i in range(len(qc.clbits)):
        qiskit_qc.add_register(ClassicalRegister(1, name=qc.clbits[i].identifier))

        bit_dic[qc.clbits[i].identifier] = qiskit_qc.clbits[-1]

    symbol_param_dic = {}

    for i in range(len(qc.data)):
        op = qc.data[i].op

        params = list(op.params)

        qiskit_params = []

        for p in params:
            if isinstance(p, Expr):

                free_symbols = list(p.free_symbols)
                lambd_expr = lambdify(free_symbols, p)

                qiskit_symbs = []

                for s in free_symbols:
                    if not s in symbol_param_dic:
                        symbol_param_dic[s] = Parameter(str(s))

                    qiskit_symbs.append(symbol_param_dic[s])
                qiskit_params.append(lambd_expr(*qiskit_symbs))

            else:
                qiskit_params.append(p)

        params = qiskit_params

        # Prepare qiskit qubits
        qubit_list = [bit_dic[qubit.identifier] for qubit in qc.data[i].qubits]
        clbit_list = [bit_dic[clbit.identifier] for clbit in qc.data[i].clbits]

        if op.name in ["qb_alloc", "qb_dealloc"]:
            # if op.name == "qb_alloc":
            # continue
            continue
            temp = QuantumCircuit(1)
            qiskit_ins = temp.to_gate()
            qiskit_ins.name = op.name

        elif op.name == "gphase":
            temp_qc = QuantumCircuit(1)
            qiskit_ins = temp_qc.to_gate()
            qiskit_ins.name = "gphase"

        elif isinstance(op, ClControlledOperation):
            q_reg = QuantumRegister(op.num_qubits)

            base_qiskit_ins = create_qiskit_instruction(op.base_op, params)

            # In Qiskit 1.3, the c_if interface was deprecated
            try:
                from qiskit.circuit import IfElseOp

                body_qc = QuantumCircuit(q_reg)
                body_qc.append(base_qiskit_ins, q_reg)
                qiskit_ins = IfElseOp((clbit_list[0], 1), true_body=body_qc)
                clbit_list = []
            except ImportError:
                cl_reg = ClassicalRegister(op.num_clbits)
                temp_qc = QuantumCircuit(q_reg, cl_reg)
                qiskit_ins = qiskit_ins.c_if(cl_reg, int(op.ctrl_state[::-1], 2))
                temp_qc.append(qiskit_ins, q_reg)
                qiskit_ins = temp_qc.to_instruction()
                qiskit_ins.name = op.name

        elif issubclass(op.__class__, ControlledOperation):
            base_name = op.base_operation.name

            if len(base_name) == 1:
                base_name = base_name.upper()

            if op.base_operation.definition:
                qiskit_definition = convert_to_qiskit(op.base_operation.definition)
                base_gate = qiskit_definition.to_gate()
                base_gate.name = op.base_operation.name
                qiskit_ins = base_gate.control(
                    len(op.controls), ctrl_state=op.ctrl_state[::-1]
                )

            else:

                if op.base_operation.name == "gphase":
                    qiskit_ins = create_qiskit_instruction(op, params)
                elif op.num_qubits == op.base_operation.num_qubits:
                    qiskit_ins = create_qiskit_instruction(op.base_operation, params)
                else:
                    base_gate = create_qiskit_instruction(op.base_operation, params)
                    qiskit_ins = base_gate.control(
                        len(op.controls), ctrl_state=op.ctrl_state[::-1]
                    )
        else:
            qiskit_ins = create_qiskit_instruction(op, params)

        # Append to qiskit circuit
        qiskit_qc.append(qiskit_ins, qubit_list, clbit_list)
    # Return result
    return qiskit_qc


def create_qiskit_instruction(op, params=[]):
    import qiskit.circuit.library.standard_gates as qsk_gates
    from qiskit.circuit import Measure, Reset, Barrier
    from qrisp.circuit import ControlledOperation
    from qiskit import QiskitError

    if op.name == "cx":
        if hasattr(op, "ctrl_state"):
            qiskit_ins = qsk_gates.XGate().control(ctrl_state=op.ctrl_state)
        else:
            qiskit_ins = qsk_gates.XGate().control()
    elif op.name == "cz":
        if hasattr(op, "ctrl_state"):
            qiskit_ins = qsk_gates.ZGate().control(ctrl_state=op.ctrl_state)
        else:
            qiskit_ins = qsk_gates.ZGate().control()
    elif op.name == "cy":
        if hasattr(op, "ctrl_state"):
            qiskit_ins = qsk_gates.YGate().control(ctrl_state=op.ctrl_state)
        else:
            qiskit_ins = qsk_gates.YGate().control()

    elif issubclass(op.__class__, ControlledOperation) and "gphase" not in op.name:
        base_name = op.base_operation.name

        if len(base_name) == 1:
            base_name = base_name.upper()

        if op.base_operation.definition:
            qiskit_definition = convert_to_qiskit(op.base_operation.definition)
            base_gate = qiskit_definition.to_gate()
            base_gate.name = op.base_operation.name
            qiskit_ins = base_gate.control(
                len(op.controls), ctrl_state=op.ctrl_state[::-1]
            )

        else:
            base_gate = create_qiskit_instruction(op.base_operation, params)
            qiskit_ins = base_gate.control(
                len(op.controls), ctrl_state=op.ctrl_state[::-1]
            )
    elif op.name == "rxx":
        qiskit_ins = qsk_gates.RXXGate(*params)
    elif op.name == "rzz":
        qiskit_ins = qsk_gates.RZZGate(*params)
    elif op.name == "ryy":
        qiskit_ins = qsk_gates.RYYGate(*params)

    elif op.name == "measure":
        qiskit_ins = Measure()
    elif op.name == "barrier":
        qiskit_ins = Barrier(op.num_qubits)
    elif op.name == "cp":
        qiskit_ins = qsk_gates.CPhaseGate(params[0])
    elif op.name == "x":
        qiskit_ins = qsk_gates.XGate()
    elif op.name == "y":
        qiskit_ins = qsk_gates.YGate()
    elif op.name == "z":
        qiskit_ins = qsk_gates.ZGate()
    elif op.name == "rx":
        qiskit_ins = qsk_gates.RXGate(params[0])
    elif op.name == "ry":
        qiskit_ins = qsk_gates.RYGate(params[0])
    elif op.name == "rz":
        qiskit_ins = qsk_gates.RZGate(params[0])
    elif op.name == "sx":
        qiskit_ins = qsk_gates.SXGate()
    elif op.name == "h":
        qiskit_ins = qsk_gates.HGate()

    elif op.name == "p":
        qiskit_ins = qsk_gates.PhaseGate(params[0])
    elif op.name == "u1":
        qiskit_ins = qsk_gates.PhaseGate(params[0])
    elif op.name == "s":
        qiskit_ins = qsk_gates.SGate()
    elif op.name == "s_dg":
        qiskit_ins = qsk_gates.SGate().inverse()
    elif op.name == "sx":
        qiskit_ins = qsk_gates.SXGate()
    elif op.name == "sx_dg":
        qiskit_ins = qsk_gates.SXdgGate()
    elif op.name == "rzz":
        qiskit_ins = qsk_gates.RZZGate(params[0])
    elif op.name == "xxyy":
        qiskit_ins = qsk_gates.XXPlusYYGate(*params, label="(XX+YY)")
    elif op.name == "t":
        qiskit_ins = qsk_gates.TGate()
    elif op.name == "t_dg":
        qiskit_ins = qsk_gates.TGate().inverse()
    elif op.name == "u3":
        qiskit_ins = qsk_gates.U3Gate(*params)
    elif op.name == "id":
        qiskit_ins = qsk_gates.IGate()
    elif op.name == "reset":
        qiskit_ins = Reset()
    elif op.definition:
        qiskit_definition = convert_to_qiskit(op.definition)
        try:
            qiskit_ins = qiskit_definition.to_gate()
        except QiskitError:
            qiskit_ins = qiskit_definition.to_instruction()

        qiskit_ins.name = op.name
    else:
        raise Exception("Could not convert operation " + str(op.name) + " to Qiskit")

    return qiskit_ins


op_dic = {op().name: op for op in op_list}
op_dic["u"] = op_dic["u3"]


def convert_from_qiskit(qiskit_qc):
    from qiskit.circuit import ControlledGate, ParameterExpression
    from qiskit import QuantumCircuit as QiskitQuantumCircuit

    from qrisp import Clbit, ControlledOperation, QuantumCircuit, Barrier, Qubit

    qc = QuantumCircuit()

    for i in range(qiskit_qc.num_qubits):
        q_reg = qiskit_qc.qubits[i]._register

        if q_reg is not None:

            if hasattr(q_reg, "_bits"):
                qb_list = q_reg._bits
            else:
                qb_list = list(q_reg)

            qubit_name = q_reg.name
            if q_reg.size > 1:
                qubit_name += "." + str(qb_list.index(qiskit_qc.qubits[i]))

            qc.add_qubit(Qubit(qubit_name))
        else:
            qc.add_qubit()

    for i in range(qiskit_qc.num_clbits):
        cl_reg = qiskit_qc.clbits[i]._register

        clbit_name = cl_reg.name
        if cl_reg.size > 1:
            clbit_name += "." + str(cl_reg._bits.index(qiskit_qc.clbits[i]))

        qc.add_clbit(Clbit(clbit_name))

    qb_dic = {qiskit_qc.qubits[i]: qc.qubits[i] for i in range(qiskit_qc.num_qubits)}
    cb_dic = {qiskit_qc.clbits[i]: qc.clbits[i] for i in range(qiskit_qc.num_clbits)}

    from sympy import sympify

    for i in range(len(qiskit_qc.data)):
        qiskit_op = qiskit_qc.data[i][0]

        if hasattr(qiskit_op, "condition_bits"):
            condition_bits = [cb_dic[cb] for cb in qiskit_op.condition_bits]
            if len(condition_bits):
                condition_value = qiskit_op.condition[1]
        elif hasattr(qiskit_op, "is_control_flow") and qiskit_op.is_control_flow():
            condition_bits = [cb_dic[cb] for cb in qiskit_qc.data[i].clbits]
            condition_value = qiskit_op.condition[1]
        else:
            condition_bits = []

        params = list(qiskit_op.params)

        qrisp_params = []

        while len(params):
            p = params.pop(0)
            if isinstance(p, np.ndarray):
                params = list(p.flatten()) + params
            elif isinstance(p, np.number):
                qrisp_params.append(p.item())
            elif isinstance(p, ParameterExpression):

                lambd_expr = sympify(ParameterExpression.sympify(p))

                qrisp_params.append(lambd_expr)

            elif isinstance(p, (float, int)):
                qrisp_params.append(float(p))
            elif isinstance(p, complex):
                qrisp_params.append(p)
            elif isinstance(p, QiskitQuantumCircuit):
                qiskit_op = p.to_gate()
                break
            else:
                print(p)
                raise Exception(f"Could not convert parameter type {type(p)}")

        params = qrisp_params

        if isinstance(qiskit_op, ControlledGate):
            controlled_gate = True
            qiskit_op = qiskit_op.base_gate
        else:
            controlled_gate = False

        try:
            op = op_dic[qiskit_op.name](*params)
        except KeyError:
            try:
                op = op_dic[qiskit_op.name.lower()](*params)
            except KeyError:
                if qiskit_op.definition is not None:
                    op = convert_from_qiskit(qiskit_op.definition).to_gate(
                        name=qiskit_op.name
                    )
                else:
                    raise Exception(
                        "Could not convert Qiskit operation "
                        + str(qiskit_op.name)
                        + " to Qrisp"
                    )

        if controlled_gate:
            qiskit_op = qiskit_qc.data[i][0]
            op = ControlledOperation(
                base_operation=op,
                num_ctrl_qubits=qiskit_op.num_ctrl_qubits,
                ctrl_state=bin_rep(qiskit_op.ctrl_state, qiskit_op.num_ctrl_qubits)[
                    ::-1
                ],
            )

        if op.name == "barrier":
            op = Barrier(qiskit_op.num_qubits)

        qubits = [qb_dic[qb] for qb in qiskit_qc.data[i][1]]
        clbits = [cb_dic[cb] for cb in qiskit_qc.data[i][2]]

        if condition_bits:
            clbits += condition_bits
            op = op.c_if(len(condition_bits), condition_value)

        qc.append(op, qubits, clbits)

    return qc
