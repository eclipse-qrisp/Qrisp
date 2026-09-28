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

"""Insert markers for qubits that can be factored out of the statevector.

State Disentangling
===================

Simulating a 50+ qubit statevector is practically impossible if it is fully
entangled. Many algorithms do however return individual qubits to a separable
state while running -- most importantly ancillae, which are uncomputed once they
have served their purpose.

- A `disentangle` instruction marks a qubit that the simulator should try to split
  off into its own tensor factor. The split is kept only if the qubit really is
  separable at that point; otherwise the state is left untouched.
- Peeling a qubit off an n-qubit tensor factor halves the number of amplitudes
  carried from then on, which is what can turn an intractable simulation into a
  solvable one.
- `_insert_disentangling` looks for positions where such a split may pay off, using
  permeability as its criterion. That is a necessary but not a sufficient condition,
  so the markers it places are speculative.
"""

from qrisp.circuit import Instruction, Operation, QuantumCircuit
from qrisp.permeability.type_checker import is_permeable

# ==============================================================================
# DISENTANGLING AND MEASUREMENT OPERATIONS
# =============================================================================
#
# A disentangler marks a qubit that the simulator should try to factor out of the tensor
# factor it currently belongs to. It is not a branching instruction:
# TensorFactor.disentangle splits the factor's state on that qubit and keeps the split
# only if the qubit genuinely is separable at that point, which is the case when
#
#   - only one measurement outcome has non-zero probability, i.e. the qubit sits in a
#     definite |0> or |1>. An uncomputed qubit is exactly this case; or
#   - the two conditional states are proportional, i.e. the state has the product form
#     (a|0> + b|1>) (x) |psi>.
#
# If neither holds, the qubit cannot be factored out and disentangle returns the factor
# unchanged.
#
# The pay-off is the width of the tensor factors: an n-qubit factor carries 2**n
# amplitudes, so peeling one qubit off halves the work for every later operation on that
# factor, and a factor that collapses to a single outcome drops out of the simulation
# altogether.
#
# The case this is built for is an ancilla that the algorithm uncomputes. Below, anc
# holds qb_0 AND qb_1, is used to phase qb_2, and is then uncomputed back to |0>:
#
#       ┌───┐
# qb_0: ┤ H ├──■───────■──
#       ├───┤  │       │
# qb_1: ┤ H ├──■───────■──
#       └───┘┌─┴─┐   ┌─┴─┐
#  anc: ─────┤ X ├─■─┤ X ├
#       ┌───┐└───┘ │ └───┘
# qb_2: ┤ H ├──────■──────
#       └───┘
#
# Disentangling anc after the second MCX succeeds: the four-qubit factor splits into
# (qb_0, qb_1, qb_2) and (anc). Drop the second MCX and the same marker fails instead,
# which is the faulty-uncomputation case that the warning-carrying disentangler reports
# (see measurement_handling).
#
# Where the markers go
# --------------------
#
# _insert_disentangling walks backwards from a measurement or reset and moves the marker
# as early as it can, using permeability as its criterion. In the circuit below the
# marker travels all the way back to just after the H on the first qubit:
#

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
#
# Permeability is weaker than separability, though: it says the qubit's value stops
# being moved between basis states, not that the qubit factors out. Insertion is
# therefore speculative -- a marker goes wherever a split might pay off, and the
# simulator silently abandons the ones that cannot succeed. Those markers carry
# warning=False precisely because failure is expected; the cost is one attempted split.
# On the ancilla circuit above, for instance, this pass places markers on qb_0, qb_1 and
# qb_2 and every one of them fails, while the marker that actually does the work on anc
# is the one _count_measurements_and_treat_alloc derives from the qb_dealloc.


class _Disentangler(Operation):
    """A custom operation that indicates where the circuit can be split into separate branches for simulation."""

    def __init__(self, warning: bool = False):
        """Initialize a disentangling operation."""
        super().__init__("disentangle", num_qubits=1)
        self.definition = QuantumCircuit(1)
        self.permeability = {0: False}
        self.warning = warning


# A disentangler is a pure marker: its only state is the `warning` flag, nothing
# mutates it in place, and Instruction.copy() routes through Operation.copy() (a
# shallow copy), so copies never alias these. That makes it safe to insert one
# shared instance per flag instead of constructing a fresh Operation -- and a
# fresh QuantumCircuit(1) definition -- for every marker. The insertion sites
# below are per-qubit or per-measurement, so this is a hot path on wide circuits.
_DISENTANGLER = _Disentangler(warning=False)
_DISENTANGLER_WITH_WARNING = _Disentangler(warning=True)


def _insert_disentangling(qc: QuantumCircuit) -> QuantumCircuit:
    """Inserts disentangling operations into the circuit where appropriate."""
    # Give every qubit a terminal reset. A disentangler can only be justified backwards
    # from a point where the qubit's value stops mattering, and a measurement or reset is
    # such a point; appending a reset per qubit means every qubit has at least one, so
    # none is skipped just because the circuit happened to end without one. Note that
    # this appends to the circuit that was passed in.
    for qb in qc.qubits:
        qc.reset(qb)

    # The search runs backwards through the circuit, so the instruction list is reversed
    # once up front and all index arithmetic below is in reversed order.
    reversed_data = list(qc.data)[::-1]
    disentangling_counter = 0
    i = 0

    # Outer loop: find the next anchor, i.e. a measurement or reset. Disentanglers
    # inserted by the inner loop are named "disentangle" and are therefore not picked up
    # as anchors themselves, even though the list grows while we walk it.
    while i < len(reversed_data):
        if reversed_data[i].op.name not in ["measure", "reset"]:
            i += 1
            continue

        qubit = reversed_data[i].qubits[0]
        j = int(i)

        # Inner loop: walk from the anchor towards the start of the circuit, looking at
        # the operations that act on this one qubit, and find out how far back the
        # disentangling can legitimately be moved.
        while j < len(reversed_data):
            instr = reversed_data[j]
            if qubit in instr.qubits:
                # Measurements, resets and disentanglers do not consume the qubit's
                # value in a way that would prevent splitting the state here, so they
                # are stepped over rather than treated as a barrier.
                if instr.op.name in ["measure", "reset", "disentangle"]:
                    j += 1
                    continue

                qubit_index = instr.qubits.index(qubit)

                # A permeable operation leaves the qubit's computational basis state
                # alone up to a phase (see the block comment above), so the two branches
                # of the state cannot interfere across it and the disentangling may be
                # pulled to before this operation. Inserting at j + 1 in the reversed
                # list places the marker immediately ahead of this instruction in
                # circuit order; the extra j += 1 skips back over what was just
                # inserted. One marker is emitted per permeable operation on the way
                # back: the earliest one that succeeds does the work and the rest are
                # no-ops, which is cheaper than trying to determine the single optimal
                # position.
                if is_permeable(instr.op, [qubit_index]):
                    reversed_data.insert(j + 1, Instruction(_DISENTANGLER, [qubit]))
                    disentangling_counter += 1
                    j += 1
                else:
                    # A non-permeable operation genuinely depends on the qubit's value,
                    # so the state cannot be split any earlier than this.
                    break
            j += 1
        i += 1

    new_qc = qc.clearcopy()
    new_qc.data = reversed_data[::-1]
    return new_qc
