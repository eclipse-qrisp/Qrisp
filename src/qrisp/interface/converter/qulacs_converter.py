"""
********************************************************************************
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


def qulacs_converter(qc):

    from qrisp.simulator import extract_measurements

    from qulacs import QuantumCircuit
    import qulacs.gate as gt

    temp_qc, measurements = extract_measurements(qc.copy())

    transpiled_qc = temp_qc.transpile(
        basis_gates=[
            "cx",
            "x",
            "y",
            "z",
            "h",
            "rz",
            "ry",
            "rx",
            "swap",
            " id",
            "mcx",
            "S",
            "Sdg",
            "t",
        ]
    )

    qulacs_qc = QuantumCircuit(len(qc.qubits))

    for instr in transpiled_qc.data:
        if instr.op.name == "x":
            qulacs_qc.add_gate(gt.X(transpiled_qc.qubits.index(instr.qubits[0])))

        elif instr.op.name == "y":
            qulacs_qc.add_gate(gt.Y(transpiled_qc.qubits.index(instr.qubits[0])))

        elif instr.op.name == "z":
            qulacs_qc.add_gate(gt.Z(transpiled_qc.qubits.index(instr.qubits[0])))

        elif instr.op.name == "h":
            qulacs_qc.add_gate(gt.H(transpiled_qc.qubits.index(instr.qubits[0])))

        elif instr.op.name == "cx":
            qulacs_qc.add_gate(
                gt.CNOT(
                    transpiled_qc.qubits.index(instr.qubits[0]),
                    transpiled_qc.qubits.index(instr.qubits[1]),
                )
            )

        elif instr.op.name == "rz":
            qulacs_qc.add_gate(
                gt.RZ(transpiled_qc.qubits.index(instr.qubits[0]), instr.op.params[0])
            )

        elif instr.op.name == "ry":
            qulacs_qc.add_gate(
                gt.RY(transpiled_qc.qubits.index(instr.qubits[0]), instr.op.params[0])
            )

        elif instr.op.name == "rx":
            qulacs_qc.add_gate(
                gt.RX(transpiled_qc.qubits.index(instr.qubits[0]), instr.op.params[0])
            )

        elif instr.op.name == "swap":
            qulacs_qc.add_gate(gt.SWAP(transpiled_qc.qubits.index(instr.qubits[0])))

        elif instr.op.name == "id":
            qulacs_qc.add_gate(gt.Identity(transpiled_qc.qubits.index(instr.qubits[0])))

        # elif instr.op.name == "mcx":
        # qulacs_qc.add_gate(gt.TOFFOLI(transpiled_qc.qubits.index(instr.qubits[0]), transpiled_qc.qubits.index(instr.qubits[1])))

        elif instr.op.name == "S":
            qulacs_qc.add_gate(gt.S(transpiled_qc.qubits.index(instr.qubits[0])))

        elif instr.op.name == "Sdg":
            qulacs_qc.add_gate(gt.Sdag(transpiled_qc.qubits.index(instr.qubits[0])))

        elif instr.op.name == "t":
            qulacs_qc.add_gate(gt.T(transpiled_qc.qubits.index(instr.qubits[0])))

        else:
            raise Exception(f"Don't know how to convert gate {instr.op.name}")

    return qulacs_qc
