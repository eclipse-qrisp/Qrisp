"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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


from qrisp.misc import bin_rep
from qrisp.circuit.standard_operations import op_list


def convert_circuit(qc, target_api="thrift", transpile=True):
    if transpile or target_api == "open_api":
        from qrisp.circuit import transpile

        qc = transpile(qc)

    if target_api == "thrift":
        from qrisp.interface.thrift_interface.interface_types import (
            PortableClbit,
            PortableInstruction,
            PortableOperation,
            PortableQuantumCircuit,
            PortableQubit,
        )
    elif target_api == "open_api":
        from qrisp.interface.openapi_interface.interface_types import (
            PortableClbit,
            PortableInstruction,
            PortableOperation,
            PortableQuantumCircuit,
            PortableQubit,
        )
    elif target_api == "qrisp":
        return convert_to_qrisp(qc)
    elif target_api == "qiskit":
        return convert_to_qiskit(qc, transpile)

    circuit_qubits = [PortableQubit(identifier=qb.identifier) for qb in qc.qubits]
    circuit_clbits = [PortableClbit(identifier=cb.identifier) for cb in qc.clbits]

    res_qc = PortableQuantumCircuit(
        qubits=circuit_qubits, clbits=circuit_clbits, data=[]
    )

    for i in range(len(qc.data)):
        instr = qc.data[i]
        if instr.op.name in ["qb_alloc", "qb_dealloc"]:
            # continue
            pass

        params = []
        for j in range(len(instr.op.params)):
            value = float(instr.op.params[j])
            params.append(value)

        if instr.op.definition is not None:
            definition = convert_circuit(
                instr.op.definition, target_api, transpile=False
            )
            op = PortableOperation(
                name=instr.op.name,
                num_qubits=instr.op.num_qubits,
                num_clbits=instr.op.num_clbits,
                params=params,
                definition=definition,
            )
        else:
            op = PortableOperation(
                name=instr.op.name,
                num_qubits=instr.op.num_qubits,
                num_clbits=instr.op.num_clbits,
                params=params,
            )

        qubits = [PortableQubit(identifier=qb.identifier) for qb in instr.qubits]
        clbits = [PortableClbit(identifier=cb.identifier) for cb in instr.clbits]

        res_qc.data.append(PortableInstruction(op=op, qubits=qubits, clbits=clbits))

    return res_qc


def convert_to_qrisp(qc):
    from qrisp.circuit import (
        Clbit,
        Instruction,
        Operation,
        QuantumCircuit,
        Qubit,
        fast_append,
        op_list,
    )

    op_dict = {op().name: op for op in op_list}

    res_qc = QuantumCircuit()

    # This dic gives the qiskit qubits/clbits when presented with their identifier
    bit_dic = {}

    # Add Qubits
    for i in range(len(qc.qubits)):
        res_qc.add_qubit(Qubit(qc.qubits[i].identifier))
        bit_dic[qc.qubits[i].identifier] = res_qc.qubits[-1]

    # Add Clbits
    for i in range(len(qc.clbits)):
        res_qc.add_clbit(Clbit(qc.clbits[i].identifier))
        bit_dic[qc.clbits[i].identifier] = res_qc.clbits[-1]

    with fast_append():
        for i in range(len(qc.data)):
            instr = qc.data[i]
            # Prepare qubits
            qubit_list = [bit_dic[qubit.identifier] for qubit in instr.qubits]
            clbit_list = [bit_dic[clbit.identifier] for clbit in instr.clbits]

            if not instr.op.definition is None:
                op = convert_to_qrisp(instr.op.definition).to_op()
            elif instr.op.name in op_dict.keys():
                op = op_dict[instr.op.name](*instr.op.params)
            elif instr.op.name[5:] in op_dict.keys():
                if instr.op.name[:5] == "c_if_":
                    op = op_dict[instr.op.name[5:]](*instr.op.params).c_if()
            elif instr.op.name[-3:] == "_dg" and instr.op.name[:-3] in op_dict.keys():
                op = op_dict[instr.op.name[:-3]](*instr.op.params).inverse()
            else:
                raise Exception(
                    "Don't know how to convert operation "
                    + str(instr.op.name)
                    + " to qrisp"
                )

            res_qc.data.append(Instruction(op, qubit_list, clbit_list))

    return res_qc


# Function to convert qrisp quantum circuits to Qiskit quantum circuits
def convert_to_qiskit(qc, transpile=False):
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

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

    # Add Instructions
    import qiskit.circuit.library.standard_gates as qsk_gates
    from qiskit.circuit import Barrier, ControlledGate, Gate, Instruction, Parameter
    from qiskit.circuit.measure import Measure

    from qrisp.circuit import ControlledOperation
    from sympy import Symbol, lambdify, Expr
    
    
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
            continue
            temp = QuantumCircuit(1)
            qiskit_ins = temp.to_gate()
            qiskit_ins.name = op.name

        elif op.name == "gphase":
            temp_qc = QuantumCircuit(1)
            qiskit_ins = temp_qc.to_gate()
            qiskit_ins.name = "gphase"

        elif op.name == "cx":
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
    from qiskit.circuit import Barrier
    from qiskit.circuit.measure import Measure
    
    

    if op.name == "rxx":
        qiskit_ins = qsk_gates.RXXGate(*params)
    elif op.name == "rzz":
        qiskit_ins = qsk_gates.RZZGate(*params)
    elif op.name == "ryy":
        qiskit_ins = qsk_gates.RYYGate(*params)
    
    elif op.name == "measure":
        qiskit_ins = Measure()

    elif op.name == "swap":
        qiskit_ins = qsk_gates.SwapGate()
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
        qiskit_ins = qsk_gates.XXPlusYYGate(*params, label='(XX+YY)')
    elif op.name == "t":
        qiskit_ins = qsk_gates.TGate()
    elif op.name == "t_dg":
        qiskit_ins = qsk_gates.TGate().inverse()
    elif op.name == "u3":
        qiskit_ins = qsk_gates.U3Gate(*params)
    elif op.name == "id":
        qiskit_ins = qsk_gates.IGate()
    elif op.definition:
        qiskit_definition = convert_to_qiskit(op.definition)
        if len(op.definition.clbits):
            qiskit_ins = qiskit_definition.to_instruction()
        else:
            qiskit_ins = qiskit_definition.to_gate()
        qiskit_ins.name = op.name
    else:
        raise Exception("Could not convert operation " + str(op.name) + " to Qiskit")

    return qiskit_ins


op_dic = {op().name: op for op in op_list}
op_dic["u"] = op_dic["u3"]


def convert_from_qiskit(qiskit_qc):
    from qiskit import transpile
    from qiskit.circuit import ControlledGate, ParameterExpression, Parameter

    from qrisp import Clbit, ControlledOperation, QuantumCircuit, Qubit, op_list
    from sympy import Symbol, lambdify, Expr
    

    qc = QuantumCircuit()

    for i in range(qiskit_qc.num_qubits):
        q_reg = qiskit_qc.qubits[i]._register

        qubit_name = q_reg.name
        if q_reg.size > 1:
            qubit_name += "." + str(q_reg._bits.index(qiskit_qc.qubits[i]))

        qc.add_qubit(Qubit(qubit_name))

    for i in range(qiskit_qc.num_clbits):
        cl_reg = qiskit_qc.clbits[i]._register

        clbit_name = cl_reg.name
        if cl_reg.size > 1:
            clbit_name += "." + str(cl_reg._bits.index(qiskit_qc.clbits[i]))

        qc.add_clbit(Clbit(clbit_name))

    qb_dic = {qiskit_qc.qubits[i]: qc.qubits[i] for i in range(qiskit_qc.num_qubits)}
    cb_dic = {qiskit_qc.clbits[i]: qc.clbits[i] for i in range(qiskit_qc.num_clbits)}

    symbol_param_dic = {}
    
    from sympy import sympify
    
    for i in range(len(qiskit_qc.data)):
        qiskit_op = qiskit_qc.data[i][0]

        params = list(qiskit_op.params)
        
        qrisp_params = []

        for p in params:
            if isinstance(p, ParameterExpression):
                
                lambd_expr = sympify(ParameterExpression.sympify(p))
                
                qrisp_params.append(lambd_expr)
                
            elif isinstance(p, (float,int)):
                qrisp_params.append(float(p))
            else:
                raise Exception(f"Could not convert parameter type {type(p)}")

        params = qrisp_params


        if isinstance(qiskit_op, ControlledGate):
            controlled_gate = True
            qiskit_op = qiskit_op.base_gate
        else:
            controlled_gate = False



            """ for j in range(len(qiskit_op.params)):
            if not isinstance(qiskit_op.params[j], float):  
                if isinstance(qiskit_op.params[j], Expr):
                    pass    
                else:
                    params[j] = float(qiskit_op.params[j]) """

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

        qubits = [qb_dic[qb] for qb in qiskit_qc.data[i][1]]
        clbits = [cb_dic[cb] for cb in qiskit_qc.data[i][2]]

        qc.append(op, qubits, clbits)

    return qc
