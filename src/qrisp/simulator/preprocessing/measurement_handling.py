# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Optimize measurement extraction and temporary qubit allocation.

Measurement and Allocation Management
=====================================

- `_extract_measurements` and `_count_measurements_and_treat_alloc` optimize
  how classical measurements and temporary qubit allocations are handled.
- `_insert_multiverse_measurements` handles deferred measurement patterns by
  introducing ancilla qubits and CNOT gates, ensuring probability distributions
    are correctly captured without breaking coherence prematurely.
"""

from typing import Any

from qrisp.circuit import ClControlledOperation, CXGate, Instruction, Measurement, QuantumCircuit
from qrisp.permeability.type_checker import is_permeable
from qrisp.simulator.preprocessing.disentangling import _Disentangler


def _count_measurements_and_treat_alloc(qc: QuantumCircuit, insert_reset: bool = True) -> int:
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
                qc.data.insert(i, Instruction(_Disentangler(True), qubits=instr.qubits))
            else:
                continue
        i += 1
    return counter


def _extract_measurements(qc: QuantumCircuit) -> tuple[QuantumCircuit, list[Instruction]]:
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
            except ValueError:
                pass
        for cb in instr.clbits:
            try:
                clbits.remove(cb)
            except ValueError:
                pass

    new_qc = qc.clearcopy()
    new_qc.data = data[::-1]
    return new_qc, mes_list


def _find_measurement_follow_up(data: list[Instruction], meas_qubit: Any, meas_clbit: Any) -> bool | None:
    """Find a blocking operation after a measurement and remove an adjacent reset."""
    for j, instr in enumerate(data):
        if meas_qubit in instr.qubits:
            if instr.op.name == "reset":
                data.pop(j)
                return True
            if not is_permeable(instr.op, [instr.qubits.index(meas_qubit)]):
                return False
        if meas_clbit in instr.clbits and not isinstance(instr, ClControlledOperation):
            return False
        if instr.op.name == "measure" and instr.qubits[0] == meas_qubit:
            return False

    return None


def _handle_deferred_measurement(
    instr: Instruction,
    qc: QuantumCircuit,
    new_data: list[Instruction],
    clbit_to_ancilla: dict[Any, Any],
    next_instr_is_reset: bool,
) -> None:
    """Replace a measurement that is used later with an ancilla-based copy."""
    ancilla = qc.add_qubit()
    new_data.append(Instruction(CXGate(), instr.qubits + [ancilla]))

    if next_instr_is_reset:
        new_data.append(Instruction(CXGate(), [ancilla] + instr.qubits))
        new_data.append(Instruction(_Disentangler(), [instr.qubits[0]]))

    clbit_to_ancilla[instr.clbits[0]] = ancilla


def _handle_reset(
    instr: Instruction,
    data: list[Instruction],
    qc: QuantumCircuit,
    new_data: list[Instruction],
) -> None:
    """Replace a reset that is used later with an ancilla-based reset."""
    reset_qubit = instr.qubits[0]
    new_data.append(Instruction(_Disentangler(), [reset_qubit]))

    for following_instr in data:
        if reset_qubit in following_instr.qubits:
            if not is_permeable(following_instr.op, [following_instr.qubits.index(reset_qubit)]):
                break
    else:
        return

    ancilla = qc.add_qubit()
    new_data.append(Instruction(CXGate(), instr.qubits + [ancilla]))
    new_data.append(Instruction(CXGate(), [ancilla] + instr.qubits))
    new_data.append(Instruction(_Disentangler(), [ancilla]))


def _handle_classical_control(
    instr: Instruction,
    qc: QuantumCircuit,
    new_data: list[Instruction],
    clbit_to_ancilla: dict[Any, Any],
) -> None:
    """Replace classical controls with quantum controls where an ancilla exists."""
    ctrl_state = instr.op.ctrl_state
    control_qubits = []
    for j, clbit in enumerate(instr.clbits):
        if clbit not in clbit_to_ancilla:
            if ctrl_state[j] == "1":
                break

            control_qubits.append(qc.add_qubit())
        else:
            control_qubits.append(clbit_to_ancilla[clbit])
    else:
        new_data.append(
            Instruction(
                instr.op.base_op.control(len(control_qubits), ctrl_state=ctrl_state),
                control_qubits + instr.qubits,
            )
        )

    for qubit in control_qubits:
        new_data.append(Instruction(_Disentangler(), [qubit]))


def _make_measurement_instructions(measurements: list[tuple[Any, Any]]) -> list[Instruction]:
    """Create measurement instructions from deferred qubit/classical-bit pairs."""
    return [Instruction(Measurement(), [qubit], [clbit]) for qubit, clbit in set(measurements)]


def _insert_multiverse_measurements(qc: QuantumCircuit) -> tuple[QuantumCircuit, list[Instruction]]:
    """Inserts multiverse measurements into the circuit to handle deferred measurement patterns."""
    new_data = []
    new_measurements = []
    clbit_to_ancilla = {}
    data = list(qc.data)

    while data:
        instr = data.pop(0)

        if instr.op.name == "measure":
            meas_qubit = instr.qubits[0]
            meas_clbit = instr.clbits[0]

            next_instr_is_reset = _find_measurement_follow_up(data, meas_qubit, meas_clbit)
            if next_instr_is_reset is None:
                new_data.append(Instruction(_Disentangler(), [meas_qubit]))
                new_measurements.append((instr.qubits[0], instr.clbits[0]))
                continue

            _handle_deferred_measurement(instr, qc, new_data, clbit_to_ancilla, next_instr_is_reset)

        elif instr.op.name == "reset":
            _handle_reset(instr, data, qc, new_data)

        elif isinstance(instr.op, ClControlledOperation):
            _handle_classical_control(instr, qc, new_data, clbit_to_ancilla)
        else:
            new_data.append(instr)

    new_measurements.extend((ancilla, clbit) for clbit, ancilla in clbit_to_ancilla.items())

    qc.data = new_data
    return qc, _make_measurement_instructions(new_measurements)
