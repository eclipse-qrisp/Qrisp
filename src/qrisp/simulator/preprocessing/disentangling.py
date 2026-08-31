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

"""Insert operations that split independent statevector branches.

State Disentangling
===================

Simulating a 50+ qubit statevector is practically impossible if fully entangled.
However, many algorithms naturally "disentangle" certain qubits during execution
(e.g., via measurements, resets, or specific uncomputations).

- `insert_disentangling` identifies points in the circuit where wave-function
  branches no longer interact.
- It inserts a custom `disentangle` instruction. The simulator catches this and
  splits the massive simulation into smaller, separate, parallelizable wave-functions,
  effectively turning an intractable problem into a solvable one.
"""

from qrisp.circuit import Instruction, Operation, QuantumCircuit
from qrisp.permeability.type_checker import is_permeable

# ==============================================================================
# DISENTANGLING AND MEASUREMENT OPERATIONS
# =============================================================================
#
# This function inserts disentangling operations into the circuit, if suited
# Consider the following circuit
#             ┌───┐
# qubit_3362: ┤ H ├──■──────────────────────
#             └───┘┌─┴─┐┌───┐
# qubit_3363: ─────┤ x ├┤ H ├──■────────────
#                  └───┘└───┘┌─┴─┐┌────────┐
# qubit_3364: ───────────────┤ x ├┤ P(0.3) ├
#                            └───┘└────────┘
# If we denote the gates after the first CNOT with U,
# the resulting state is
# 2**-0.5*(|0>U|0> + |1>U|1>)
# Since we are not interested in the wave-function but in the resulting
# probabilities, it is now safe to evaluate both summands of the wave function
# separately, ie. evaluate |0>U|0> and |1>U|1> and multiply the resulting,
# probabilities with 2**-0.5.
# This is possible because there is not going to be any interaction between the to
# states.
# (TO-DO better proof required)
# Therefore the workflow for performing the disentangling is
# 1. Identify position to insert disentangler (here: directly behind the first CNOT)
# 2. Perform measurement (to acquire classical probabilities)
# 3. Continue the simulation on the two decoherent states

# Simulating both states separately has two advantages
# 1. If one of the branches has probability zero, this will show up in the measurement
#   and we don't have to continue the simulation of this branch
# 2. Simulating two states that don't interact is easier to parallelize

# As it turns out, it is even possible to disentangle this circuit right
# after the first H gate on qubit zero


#              ┌───┐     ┌────────┐
# qubit_10119: ┤ H ├──■──┤ P(0.2) ├─────────────────■──
#              └───┘┌─┴─┐└─┬───┬──┘                 │
# qubit_10120: ─────┤ x ├──┤ H ├─────■──────────────┼──
#                   └───┘  └───┘   ┌─┴─┐┌────────┐┌─┴─┐
# qubit_10121: ────────────────────┤ x ├┤ P(0.3) ├┤ x ├
#                                  └───┘└────────┘└───┘


# This is because all the gates after the H-gate are permeable on this qubit
# (for an elaboration on permeability check the uncomputation module)
# Roughly said, a gate U which is permeable on the first qubit behaves like this
# U|0>|a> = exp(i*phi_0) |0> U_0 |a>
# U|1>|b> = exp(i*phi_1) |0> U_1 |b>


class Disentangler(Operation):
    """A custom operation that indicates where the circuit can be split into separate branches for simulation."""

    def __init__(self, warning: bool = False):
        """Initialize a disentangling operation."""
        super().__init__("disentangle", num_qubits=1)
        self.definition = QuantumCircuit(1)
        self.permeability = {0: False}
        self.warning = warning


def insert_disentangling(qc: QuantumCircuit) -> QuantumCircuit:
    """Inserts disentangling operations into the circuit where appropriate."""
    for qb in qc.qubits:
        qc.reset(qb)

    reversed_data = list(qc.data)[::-1]
    disentangling_counter = 0
    i = 0

    while i < len(reversed_data):
        if reversed_data[i].op.name not in ["measure", "reset"]:
            i += 1
            continue

        qubit = reversed_data[i].qubits[0]
        j = int(i)

        while j < len(reversed_data):
            instr = reversed_data[j]
            if qubit in instr.qubits:
                if instr.op.name in ["measure", "reset", "disentangle"]:
                    j += 1
                    continue

                qubit_index = instr.qubits.index(qubit)

                if is_permeable(instr.op, [qubit_index]):
                    reversed_data.insert(j + 1, Instruction(Disentangler(), [qubit]))
                    disentangling_counter += 1
                    j += 1
                else:
                    break
            j += 1
        i += 1

    new_qc = qc.clearcopy()
    new_qc.data = reversed_data[::-1]
    return new_qc
