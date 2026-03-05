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

import numpy as np

from qrisp.circuit.instruction import Instruction
from qrisp.circuit.operation import (
    Operation,
)
from qrisp.circuit import fast_append


# This function dissolves any Operation objects that have a definition circuit such
# that the result only consists of elementary gates
def transpile(qc, transpilation_level=np.inf, transpile_predicate=None, **kwargs):
    from qrisp.circuit import QuantumCircuit, Clbit, Qubit

    with fast_append():

        transpiled_qc = QuantumCircuit()

        # [transpiled_qc.add_qubit(Qubit(qb.identifier)) for qb in qc.qubits]
        # [transpiled_qc.add_clbit(Clbit(cb.identifier)) for cb in qc.clbits]

        for qb in qc.qubits:
            if isinstance(qb, Qubit):
                transpiled_qc.add_qubit(qb)
            else:
                transpiled_qc.add_qubit(Qubit(qb.identifier))

        for cb in qc.clbits:
            if isinstance(qb, Qubit):
                transpiled_qc.add_clbit(cb)
            else:
                transpiled_qc.add_clbit(Clbit(cb.identifier))

        translation_dic = {
            qc.qubits[i].identifier: transpiled_qc.qubits[i]
            for i in range(len(qc.qubits))
        }
        translation_dic.update(
            {
                qc.clbits[i].identifier: transpiled_qc.clbits[i]
                for i in range(len(qc.clbits))
            }
        )

        if transpile_predicate is None:
            transpile_predicate_ = lambda i, op: i < transpilation_level
        else:
            transpile_predicate_ = (
                lambda i, op: i < transpilation_level and transpile_predicate(op)
            )

        transpile_inner(qc, transpiled_qc, translation_dic, transpile_predicate_)

        QuantumCircuit.fast_append = False

        if not kwargs or not hasattr(qc, "to_qiskit"):
            return transpiled_qc
        else:
            from qrisp import QuantumCircuit

            qiskit_qc = transpiled_qc.to_qiskit()

            from qiskit import transpile as qiskit_transpile

            transpiled_qiskit_qc = qiskit_transpile(qiskit_qc, **kwargs)

            qrisp_qc = QuantumCircuit.from_qiskit(transpiled_qiskit_qc)

            return qrisp_qc


def transpile_inner(
    transpilation_qc,
    target_qc,
    translation_dic,
    transpile_predicate,
    transpilation_level=0,
):
    for i in range(len(transpilation_qc.data)):
        instr = transpilation_qc.data[i]
        if instr.op.definition:
            if transpile_predicate(transpilation_level, instr.op):
                definition = instr.op.definition

                new_translation_dic = {
                    definition.qubits[j].identifier: translation_dic[
                        instr.qubits[j].identifier
                    ]
                    for j in range(len(instr.qubits))
                }

                new_translation_dic.update(
                    {
                        definition.clbits[j].identifier: translation_dic[
                            instr.clbits[j].identifier
                        ]
                        for j in range(len(instr.clbits))
                    }
                )

                transpile_inner(
                    instr.op.definition,
                    target_qc,
                    new_translation_dic,
                    transpile_predicate,
                    transpilation_level + 1,
                )
                continue

        if not isinstance(instr.op, Operation):
            op = Operation(init_op=instr.op)
        else:
            op = instr.op

        target_qc.data.append(
            Instruction(
                op,
                [translation_dic[qb.identifier] for qb in instr.qubits],
                [translation_dic[cb.identifier] for cb in instr.clbits],
            )
        )
        # target_qc.append()


def extend(qc_0, qc_1, translation_dic="id"):
    if translation_dic == "id":
        translation_dic = {}
        for qb in qc_1.qubits:
            translation_dic[qb] = qb
            if qb not in qc_0.qubits:
                qc_0.add_qubit(qb)

        for cb in qc_1.clbits:
            translation_dic[qb] = qb
            if cb not in qc_0.clbits:
                qc_0.add_clbit(cb)

    # Copy in order to prevent modification
    translation_dic = dict(translation_dic)

    for key in list(translation_dic.keys()):
        if not isinstance(key, str):
            translation_dic[key.identifier] = translation_dic[key]

    for i in range(len(qc_1.data)):
        instruction_other = qc_1.data[i]
        qubits = []
        for qb in instruction_other.qubits:
            qubits.append(translation_dic[qb.identifier])

        clbits = []

        for cb in instruction_other.clbits:
            clbits.append(translation_dic[cb.identifier])

        instr_type = type(instruction_other)

        qc_0.data.append(instr_type(instruction_other.op, qubits, clbits))
