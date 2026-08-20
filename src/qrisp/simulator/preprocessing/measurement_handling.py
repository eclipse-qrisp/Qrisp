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

================================================================================
Measurement and Allocation Management
================================================================================

- `extract_measurements` and `count_measurements_and_treat_alloc` optimize
  how classical measurements and temporary qubit allocations are handled.
- `insert_multiverse_measurements` handles deferred measurement patterns by
  introducing ancilla qubits and CNOT gates, ensuring probability distributions
  are correctly captured without breaking coherence prematurely.
================================================================================
"""

from qrisp.circuit import ClControlledOperation, CXGate, Instruction, Measurement, QuantumCircuit
from qrisp.permeability.type_checker import is_permeable
from qrisp.simulator.preprocessing.disentangling import Disentangler


def count_measurements_and_treat_alloc(qc: QuantumCircuit, insert_reset: bool = True) -> int:
    """Counts the number of measurement instructions in the circuit and handles qubit allocation/deallocation."""
    counter = 0
    i = 0
    while i < len(qc.data):
        instr = qc.data[i]
        if instr.op.name == "barrier":
            qc.data.pop(i)
            continue
        if instr.op.name == "measure":
            counter += 1
        elif instr.op.name == "qb_alloc":
            qc.data.pop(i)
            continue
        elif instr.op.name == "qb_dealloc":
            qc.data.pop(i)
            if insert_reset:
                qc.data.insert(i, Instruction(Disentangler(True), qubits=instr.qubits))
            else:
                continue
        i += 1
    return counter


def extract_measurements(qc: QuantumCircuit) -> tuple[QuantumCircuit, list[Instruction]]:
    """Extracts measurement instructions from the circuit and returns a new circuit without them."""
    qubits = list(qc.qubits)
    clbits = list(qc.clbits)
    mes_list = []
    data = []
    for instr in qc.data[::-1]:
        if instr.op.name == "measure" and instr.qubits[0] in qubits and instr.clbits[0] in clbits:
            mes_list.append(instr)
        else:
            data.append(instr)

        for qb in instr.qubits:
            try:
                qubits.remove(qb)
            except:
                pass
        for cb in instr.clbits:
            try:
                clbits.remove(cb)
            except:
                pass

    new_qc = qc.clearcopy()
    new_qc.data = data[::-1]
    return new_qc, mes_list


def insert_multiverse_measurements(qc: QuantumCircuit) -> tuple[QuantumCircuit, list[Instruction]]:
    """Inserts multiverse measurements into the circuit to handle deferred measurement patterns."""
    new_data = []
    new_measurements = []
    cb_to_qb_dic = {}
    data = list(qc.data)

    while data:
        instr = data.pop(0)

        if instr.op.name == "measure":
            meas_qubit = instr.qubits[0]
            meas_clbit = instr.clbits[0]

            next_instr_is_reset = False
            for j in range(len(data)):
                if meas_qubit in data[j].qubits:
                    if data[j].op.name == "reset":
                        next_instr_is_reset = True
                        data.pop(j)
                        break
                    if not is_permeable(data[j].op, [data[j].qubits.index(meas_qubit)]):
                        break
                if meas_clbit in data[j].clbits and not isinstance(data[j], ClControlledOperation):
                    break
                # This treats the case that two measurements with the same outcome are performed
                # in this case we break the loop to make the first measurement appear as a
                # separate qubit.
                if data[j].op.name == "measure" and data[j].qubits[0] == meas_qubit:
                    break
            else:
                new_data.append(Instruction(Disentangler(), [meas_qubit]))
                new_measurements.append((instr.qubits[0], instr.clbits[0]))
                continue

            qb = qc.add_qubit()
            new_data.append(Instruction(CXGate(), instr.qubits + [qb]))

            if next_instr_is_reset:
                new_data.append(Instruction(CXGate(), [qb] + instr.qubits))
                new_data.append(Instruction(Disentangler(), [meas_qubit]))

            cb_to_qb_dic[instr.clbits[0]] = qb
            mes_instr = instr.copy()
            mes_instr.qubits = [qb]

        elif instr.op.name == "reset":
            meas_qubit = instr.qubits[0]
            new_data.append(Instruction(Disentangler(), [meas_qubit]))

            for j in range(len(data)):
                if meas_qubit in data[j].qubits:
                    if not is_permeable(data[j].op, [data[j].qubits.index(meas_qubit)]):
                        break
            else:
                continue

            qb = qc.add_qubit()
            new_data.append(Instruction(CXGate(), instr.qubits + [qb]))
            new_data.append(Instruction(CXGate(), [qb] + instr.qubits))
            new_data.append(Instruction(Disentangler(), [qb]))

        elif isinstance(instr.op, ClControlledOperation):
            new_qubits = []
            ctrl_state = instr.op.ctrl_state
            control_qubits = []
            for j, cb in enumerate(instr.clbits):
                if cb not in cb_to_qb_dic:
                    if ctrl_state[j] == "1":
                        break

                    qb = qc.add_qubit()
                    new_qubits.append(qb)
                    control_qubits.append(qb)
                else:
                    control_qubits.append(cb_to_qb_dic[cb])
            else:
                new_data.append(
                    Instruction(
                        instr.op.base_op.control(len(control_qubits), ctrl_state=ctrl_state),
                        control_qubits + instr.qubits,
                    )
                )

            for qb in control_qubits:
                new_data.append(Instruction(Disentangler(), [qb]))
        else:
            new_data.append(instr)

    for cb, qb in cb_to_qb_dic.items():
        new_measurements.append((qb, cb))

    new_measurements = list(set(new_measurements))
    measurements = []
    for qb, cb in new_measurements:
        measurements.append(Instruction(Measurement(), [qb], [cb]))

    qc.data = new_data
    return qc, measurements
