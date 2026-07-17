"""********************************************************************************
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

import numpy as np

from qrisp import ControlledOperation


def create_tket_instruction(op):
    try:
        from pytket import OpType
        from pytket.circuit import CircBox
    except (ModuleNotFoundError, ImportError) as exc:
        raise ImportError("PyTket must be installed to be able to use the Qrisp to PyTket converter.") from exc

    if op.name == "rxx":
        tket_ins = OpType.XXPhase
    elif op.name == "rzz":
        tket_ins = OpType.ZZPhase
    elif op.name == "ryy":
        tket_ins = OpType.YYPhase
    elif op.name == "measure":
        tket_ins = OpType.Measure
    elif op.name == "swap":
        tket_ins = OpType.SWAP
    elif op.name == "h":
        tket_ins = OpType.H
    elif op.name == "p":
        tket_ins = OpType.Rz
    elif op.name == "x":
        tket_ins = OpType.X
    elif op.name == "y":
        tket_ins = OpType.Y
    elif op.name == "z":
        tket_ins = OpType.Z
    elif op.name == "rx":
        tket_ins = OpType.Rx
    elif op.name == "ry":
        tket_ins = OpType.Ry
    elif op.name == "rz":
        tket_ins = OpType.Rz
    elif op.name == "s":
        tket_ins = OpType.S
    elif op.name == "s_dg":
        tket_ins = OpType.Sdg
    elif op.name == "t":
        tket_ins = OpType.T
    elif op.name == "t_dg":
        tket_ins = OpType.Tdg
    elif op.name == "u3":
        tket_ins = OpType.U3

    elif op.definition:
        # if complex definition we create an abstract circBox for the section
        tket_definition = pytket_converter(op.definition, boxFlag=True)
        # Defensive check: the boxed definition should span the same number of
        # qubits as the operation it represents.
        if tket_definition.n_qubits != op.num_qubits:  # pragma: no cover
            raise Exception("Converted definition of '" + str(op.name) + "' has a mismatched qubit count")

        tket_ins = CircBox(tket_definition)

    else:
        raise Exception("Could not convert operation " + str(op.name) + " to PyTket")

    return tket_ins


def pytket_converter(qc, boxFlag=False):
    try:
        from pytket import Circuit, OpType, Qubit
        from pytket.circuit import CircBox
    except (ModuleNotFoundError, ImportError) as exc:
        raise ImportError("PyTket must be installed to be able to use the Qrisp to PyTket converter.") from exc

    # This dic gives the qiskit qubits/clbits when presented with their identifier
    qubit_dic = {}
    tket_qc = Circuit()
    # stringListQubs = []
    tketQubits = []
    for i in range(len(qc.qubits)):
        # add a named qubit
        tketQubits.append(Qubit(name=str(qc.qubits[i].identifier), index=i))
        qubit_dic[qc.qubits[i].identifier] = tketQubits[-1]
        tket_qc.add_qubit(tketQubits[-1])

    # Flag for alternative qubit assignment if we try to create an abstract CircBox
    if boxFlag:
        tket_qc = Circuit(len(qc.qubits))
        qubit_dic = dict()
        for i in range(len(qc.qubits)):
            qubit_dic[qc.qubits[i].identifier] = i

    clbit_dic = {}
    # Add Clbits
    tketClbits = []
    if len(qc.clbits):
        c_reg = tket_qc.add_c_register(name="creg_std", size=len(qc.clbits))
    for i in range(len(qc.clbits)):
        clbit_dic[qc.clbits[i].identifier] = c_reg[i]
        # this will hopefully be used one day, when other simulators other than Aer are used with this backend, or... quantinuum decides to fix their backend integration
        # will throw an error on Aer backend and QASM converter, since they apparently only supports a single classical register, which is a lie
        """ tketClbits.append(Bit( name = str(qc.clbits[i].identifier)))
        clbit_dic[qc.clbits[i].identifier] = tketClbits[-1]
        tket_qc.add_bit(tketClbits[-1]) """

    for i in range(len(qc.data)):
        op = qc.data[i].op

        params = list(op.params)
        # Prepare qubits
        qubit_list = [qubit_dic[qubit.identifier] for qubit in qc.data[i].qubits]
        clbit_list = [clbit_dic[clbit.identifier] for clbit in qc.data[i].clbits]

        if op.name in [
            "cp",
            "p",
            "rx",
            "rz",
            "ry",
            "rxx",
            "rzz",
            "ryy",
            "u1",
            "u3",
        ]:  # and not boxFlag:
            # pytket expects angles in pi multiples
            params = [index / np.pi for index in params]

        # add_gate
        if op.name in ["qb_alloc", "qb_dealloc"]:
            continue

        elif op.name == "gphase":
            # Global phase: pytket tracks it circuit-wide. add_phase takes
            # half-turns, so params[0] (radians, not pi-scaled) is divided by pi.
            tket_qc.add_phase(params[0] / np.pi)
            continue

        elif op.name == "cx":
            tket_ins = OpType.CX

        elif op.name == "cy":
            tket_ins = OpType.CY

        elif op.name == "cz":
            tket_ins = OpType.CZ

        elif op.name == "cp":
            # cp is the controlled-phase gate diag(1,1,1,e^{i*theta}); pytket's
            # CU1 is exactly controlled-U1 (CRz is a different gate). (#630)
            tket_ins = OpType.CU1

        elif op.name == "sx":
            # bugged -> params empty
            params = []
            tket_ins = OpType.SX
        elif op.name == "sx_dg":
            # bugged -> params empty
            params = []
            tket_ins = OpType.SXdg

        elif op.name == "u1":
            # angle already converted to pi-units in the block above (#631)
            tket_ins = OpType.Rz
        elif op.name == "id":
            params = []
            # bugged
            tket_ins = OpType.noop

        elif issubclass(op.__class__, ControlledOperation):
            base_name = op.base_operation.name

            if len(base_name) == 1:
                base_name = base_name.upper()

            # pytket_converter always returns a Circuit, so a controlled operation
            # is always emitted as an abstract CircBox of its definition.
            tket_definition = pytket_converter(op.definition, boxFlag=True)
            tket_definition.name = base_name
            tket_ins = CircBox(tket_definition)

        else:
            tket_ins = create_tket_instruction(op)

        if isinstance(tket_ins, CircBox):
            tket_qc.add_circbox(tket_ins, qubit_list)

        elif clbit_list:
            # add other isinstance checks from above here aswell?
            tket_qc.add_gate(tket_ins, params, qubit_list + clbit_list)

        else:
            tket_qc.add_gate(tket_ins, params, qubit_list)

    return tket_qc
