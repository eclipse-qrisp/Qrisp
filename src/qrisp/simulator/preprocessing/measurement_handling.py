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
from qrisp.simulator.preprocessing.disentangling import _DISENTANGLER, _DISENTANGLER_WITH_WARNING


def _count_measurements_and_treat_alloc(qc: QuantumCircuit, insert_reset: bool = True) -> int:
    """Counts the number of measurement instructions in the circuit and handles qubit allocation/deallocation."""
    # Three kinds of book-keeping instruction are removed here, none of which the
    # statevector simulator can execute as dynamics:
    #
    # "barrier" is purely a compiler/visual hint, so it is dropped unconditionally.
    #
    # "qb_alloc" marks a qubit entering the computation. The QuantumState is built with
    # one single-qubit TensorFactor per circuit qubit, all in |0>, before the simulation
    # starts, so the marker is already satisfied and is dropped as well.
    #
    # "qb_dealloc" marks a qubit leaving the computation. By deallocating, the calling
    # program asserts that the qubit has been uncomputed back into |0>. If that holds,
    # the qubit is separable and can be split off into its own tensor factor, which is
    # exactly what a disentangler asks for and what keeps the simulated state small.
    # The marker is therefore replaced by one -- unless insert_reset is False, in which
    # case it is simply discarded.
    #
    # This is the only insertion site in this module that uses the warning-carrying
    # disentangler. A deallocated qubit that turns out not to be separable means the
    # caller's uncomputation was faulty, which is a bug in their program and worth
    # reporting: the simulator prints "Faulty uncomputation found during simulation".
    # Every other disentangler below is an internal optimisation, where failing to
    # separate is merely a missed opportunity and must stay silent.
    #
    # The measurement count is returned so the caller can stop simulating as soon as
    # every measurement has been evaluated.
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
                qc.data.insert(i, Instruction(_DISENTANGLER_WITH_WARNING, qubits=instr.qubits))
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


# Deferred ("multiverse") measurements
# ------------------------------------
#
# A measurement in the middle of a circuit is awkward for a statevector simulator: the
# classical outcome would be needed before the following instructions can be applied,
# which forces the simulation to commit to a value early. The functions below instead
# move every measurement to the end of the circuit, using the deferred measurement
# principle: the outcome is copied into a fresh ancilla qubit with a CNOT rather than
# being read out.
#
#       ┌───┐┌─┐
# qb_0: ┤ H ├┤M├───────────────────────
#       └───┘└╥┘┌────── ┌───┐ ───────┐
# qb_1: ──────╫─┤ If-0  ┤ X ├  End-0 ├─
#             ║ └──╥─── └───┘ ───────┘
# cb_0: ══════╩════■═══════════════════
#
# becomes
#
#          ┌───┐
#    qb_0: ┤ H ├──■──────────
#          └───┘  │  ┌───┐
#    qb_1: ───────┼──┤ X ├───
#               ┌─┴─┐└─┬─┘┌─┐
# ancilla: ─────┤ X ├──■──┤M├
#               └───┘     └╥┘
#    cb_0: ════════════════╩═
#
# The ancilla now carries what the classical bit would have carried, so a later
# classically controlled operation becomes an ordinary quantum-controlled operation on
# that ancilla, and the measurement itself can be appended at the very end. The
# resulting state holds every outcome at once -- hence "multiverse" -- and it is left to
# the disentangler machinery to split it into independent branches wherever that turns
# out to be cheap (see the disentangling module).
#
# The three helpers below implement the instruction kinds that need rewriting -- a
# measurement whose result is used later, a reset, and a classically controlled
# operation -- and _insert_multiverse_measurements drives them.


def _find_measurement_follow_up(data: list[Instruction], meas_qubit: Any, meas_clbit: Any) -> bool | None:
    """Find a blocking operation after a measurement and remove an adjacent reset."""
    # Scans forward from a measurement to decide whether it has to be deferred at all,
    # and distinguishes three outcomes:
    #
    #   True  -- the next thing touching the qubit is a reset. That reset is popped from
    #            the remaining data here, because the caller reproduces its effect as
    #            part of the rewrite instead of executing it separately.
    #   False -- something later depends on this measurement (a non-permeable operation
    #            on the qubit, a later use of the classical bit, or a second measurement
    #            of the same qubit), so the outcome has to be kept available and the
    #            measurement must be deferred onto an ancilla.
    #   None  -- nothing in the rest of the circuit depends on it. The measurement is
    #            effectively terminal and needs no ancilla; the caller simply
    #            disentangles the qubit and records the measurement for the end.
    #
    # A second measurement of the same qubit counts as blocking so that the earlier one
    # is materialised on its own ancilla. Otherwise both would end up sharing a qubit
    # and their outcomes could no longer be told apart.
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
    # Copy the measured qubit into a fresh ancilla. The ancilla starts in |0>, so the
    # CNOT leaves it holding exactly the value the classical bit would have held, while
    # the measurement itself is postponed to the end of the circuit.
    ancilla = qc.add_qubit()
    new_data.append(Instruction(CXGate(), instr.qubits + [ancilla]))

    # If the measurement was immediately followed by a reset (already removed from the
    # remaining data by _find_measurement_follow_up), reproduce that reset here. The
    # second CNOT, now controlled on the ancilla, XORs the value back out of the
    # original qubit and thereby returns it to |0>. Disentangling it afterwards lets the
    # simulator split it off as its own tensor factor.
    if next_instr_is_reset:
        new_data.append(Instruction(CXGate(), [ancilla] + instr.qubits))
        new_data.append(Instruction(_DISENTANGLER, [instr.qubits[0]]))

    # Record which qubit now stands in for this classical bit, so that later classically
    # controlled operations can be controlled on the ancilla instead, and so that the
    # deferred measurement can be emitted once the walk is finished.
    clbit_to_ancilla[instr.clbits[0]] = ancilla


def _handle_reset(
    instr: Instruction,
    data: list[Instruction],
    qc: QuantumCircuit,
    new_data: list[Instruction],
) -> None:
    """Replace a reset that is used later with an ancilla-based reset."""
    # A reset is first of all an opportunity to disentangle: it declares that whatever
    # the qubit held is no longer of interest, which is precisely the condition under
    # which the simulator may split the qubit off into its own tensor factor. This
    # marker is emitted unconditionally, because it is the cheap half of the rewrite.
    reset_qubit = instr.qubits[0]
    new_data.append(Instruction(_DISENTANGLER, [reset_qubit]))

    # Whether the qubit also has to be physically returned to |0> depends on what comes
    # afterwards. A permeable operation does not depend on the qubit's value -- it only
    # picks up a phase per computational basis state, see the permeability module -- so
    # if every later operation on this qubit is permeable, the disentangler above is the
    # whole story and there is nothing left to do.
    for following_instr in data:
        if reset_qubit in following_instr.qubits:
            if not is_permeable(following_instr.op, [following_instr.qubits.index(reset_qubit)]):
                break
    else:
        return

    # Something later genuinely depends on the qubit being |0>, so perform the reset
    # coherently rather than by measuring: the first CNOT copies the value into a fresh
    # ancilla, the second XORs it back out of the original qubit, leaving that qubit in
    # |0>. The discarded value now sits in the ancilla, which is disentangled in turn.
    ancilla = qc.add_qubit()
    new_data.append(Instruction(CXGate(), instr.qubits + [ancilla]))
    new_data.append(Instruction(CXGate(), [ancilla] + instr.qubits))
    new_data.append(Instruction(_DISENTANGLER, [ancilla]))


def _handle_classical_control(
    instr: Instruction,
    qc: QuantumCircuit,
    new_data: list[Instruction],
    clbit_to_ancilla: dict[Any, Any],
) -> None:
    """Replace classical controls with quantum controls where an ancilla exists."""
    # Because the measurements feeding this operation have been deferred, their outcomes
    # live on ancilla qubits rather than in classical bits. The classical control is
    # therefore rewritten into a quantum control over those ancillas.
    ctrl_state = instr.op.ctrl_state
    control_qubits = []
    for j, clbit in enumerate(instr.clbits):
        if clbit not in clbit_to_ancilla:
            # A classical bit missing from the map was never written by a deferred
            # measurement, so it still holds its initial value of 0. If this control
            # position requires a 1 the condition can never be satisfied, making the
            # whole operation dead code: break out so the else branch is skipped and the
            # instruction is dropped from the circuit entirely.
            if ctrl_state[j] == "1":
                break

            # The control position requires a 0, which an unwritten bit already
            # satisfies. A freshly allocated qubit is in |0>, so it can stand in for the
            # bit and keep the control pattern the same width as the original.
            control_qubits.append(qc.add_qubit())
        else:
            control_qubits.append(clbit_to_ancilla[clbit])
    else:
        # Reached only when every control position turned out to be satisfiable:
        # re-issue the wrapped operation as a quantum-controlled gate over the
        # substituted qubits.
        new_data.append(
            Instruction(
                instr.op.base_op.control(len(control_qubits), ctrl_state=ctrl_state),
                control_qubits + instr.qubits,
            )
        )

    # Once a control qubit has been consumed it carries nothing but classical
    # information, which makes this a good place to let the simulator branch. This also
    # runs on the dropped-operation path, where it disentangles the substitute qubits
    # that were allocated before the break.
    for qubit in control_qubits:
        new_data.append(Instruction(_DISENTANGLER, [qubit]))


def _make_measurement_instructions(measurements: list[tuple[Any, Any]]) -> list[Instruction]:
    """Create measurement instructions from deferred qubit/classical-bit pairs."""
    # These are the postponed measurements, handed back as a separate list rather than
    # as circuit data. The set() removes duplicates, which arise because a qubit/clbit
    # pair can be registered both as a terminal measurement and through the
    # clbit-to-ancilla map. The order they come out in does not matter, since the
    # measurements are independent of one another.
    return [Instruction(Measurement(), [qubit], [clbit]) for qubit, clbit in set(measurements)]


def _insert_multiverse_measurements(qc: QuantumCircuit) -> tuple[QuantumCircuit, list[Instruction]]:
    """Inserts multiverse measurements into the circuit to handle deferred measurement patterns."""
    # Walks the circuit once, rebuilding it into new_data while rewriting the three
    # instruction kinds that would otherwise force an early classical decision.
    # Instructions are consumed from the front of `data` so that the helpers can look
    # ahead at -- and, for a reset directly following a measurement, remove -- the
    # instructions that come after the one being handled.
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
                new_data.append(Instruction(_DISENTANGLER, [meas_qubit]))
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
