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

from __future__ import annotations

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.circuit.operation import Operation
from qrisp.circuit.qubit import Qubit
from qrisp.circuit.pass_management.circuit_pass import CircuitPass


@CircuitPass
def commute_swaps(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Commute single-qubit instructions past SWAP gates.

    This pass walks the circuit and moves any single-qubit instruction
    (gates *and* measurements) to the other side of an adjacent SWAP on
    the same qubit.  The commuted instruction is placed on the swapped
    partner qubit immediately after the SWAP.

    This is valid because SWAP simply relabels the two qubit lines, so a
    single-qubit operation applied before the SWAP on qubit *a* is
    equivalent to the same operation applied after the SWAP on qubit *b*
    (and vice versa).

    Parameters
    ----------
    qc : QuantumCircuit
        Input quantum circuit.

    Returns
    -------
    QuantumCircuit
        Circuit with single-qubit instructions commuted past SWAPs.

    Examples
    --------
    Commute a Hadamard past a SWAP::

        >>> from qrisp import QuantumCircuit, PassManager
        >>> from qrisp import commute_swaps
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0)
        >>> qc.swap(0, 1)
        >>> print(qc)
               ┌───┐   
        qb_65: ┤ H ├─X─
               └───┘ │ 
        qb_66: ──────X─

        >>> pm = PassManager()
        >>> pm += commute_swaps
        >>> optimized_qc = pm.run(qc)
        >>> print(optimized_qc)
        <BLANKLINE>                          
        qb_65: ─X──────
                │ ┌───┐
        qb_66: ─X─┤ H ├
                  └───┘
    """
    qc_new = qc.clearcopy()
    last_instruction_dic: dict[Qubit, list[int]] = {qb: [] for qb in qc_new.qubits}

    id_op = Operation(name="id", num_qubits=1)

    for instr in qc.data:
        qc_new.append(instr)

        if instr.op.name == "swap":

            dic_update: dict[Qubit, list[int]] = {qb: [] for qb in instr.qubits}

            for j in range(2):
                qubit = instr.qubits[j]
                partner = instr.qubits[(j + 1) % 2]
                commutable_indices = last_instruction_dic[qubit]

                if not commutable_indices:
                    continue

                for last_instr_index in commutable_indices:
                    last_instr = qc_new.data[last_instr_index]

                    if last_instr.op.num_qubits != 1:
                        continue

                    id_instr = last_instr.copy()
                    id_instr.op = id_op
                    id_instr.clbits = []
                    qc_new.data[last_instr_index] = id_instr

                    qc_new.append(last_instr.op, [partner], last_instr.clbits)
                    dic_update[partner].append(len(qc_new.data) - 1)

            for qb in instr.qubits:
                last_instruction_dic[qb] = []

            last_instruction_dic.update(dic_update)

        else:
            instr_index = len(qc_new.data) - 1

            if instr.op.num_qubits == 1 and len(instr.qubits) == 1:
                qb = instr.qubits[0]
                last_instruction_dic[qb].append(instr_index)
            else:
                for qb in instr.qubits:
                    last_instruction_dic[qb] = []

    # Remove identity placeholders
    qc_new.data = [instr for instr in qc_new.data if instr.op.name != "id"]

    return qc_new
