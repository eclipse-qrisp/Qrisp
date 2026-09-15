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

"""Preprocess quantum circuits for efficient statevector simulation.

Qrisp Simulator Circuit Preprocessing
=====================================

This module is the main entry point for the simulator preprocessing package.
It optimizes quantum circuits before they are dispatched to the Qrisp
statevector simulator backend. Because simulating large statevectors is
exponentially expensive in both time and memory, it applies several advanced
heuristic transformations to significantly reduce computational overhead.

The preprocessor acts as a compiler pass, modifying the circuit's structure
without altering its mathematical outcome. It combines the following
submodules into a single preprocessing pipeline:

1. :mod:`~qrisp.simulator.preprocessing.gate_grouping` -- Gate Grouping
2. :mod:`~qrisp.simulator.preprocessing.disentangling` -- State Disentangling
3. :mod:`~qrisp.simulator.preprocessing.measurement_handling` -- Measurement
    and Allocation Management
4. :mod:`~qrisp.simulator.preprocessing.circuit_reordering` -- Circuit
    Reordering

The :func:`_circuit_preprocessor` function evaluates the incoming circuit,
applies disentangling if the circuit is dangerously wide, groups the gates for
performance, and finally reorders the circuit to safely push measurements,
resets, and disentanglers to the end of execution blocks.
"""

from qrisp.circuit import QuantumCircuit
from qrisp.simulator.preprocessing.circuit_reordering import _reorder_circuit
from qrisp.simulator.preprocessing.disentangling import _insert_disentangling
from qrisp.simulator.preprocessing.gate_grouping import _group_qc

# Disentangling is only attempted above this width. Splitting the state into
# independent branches costs analysis time and pays off in proportion to how large the
# statevector would otherwise become, so it is not worth doing for narrow circuits. The
# specific value is empirical -- see the TO-DO in _circuit_preprocessor below.
_DISENTANGLING_QUBIT_THRESHOLD = 45


def _circuit_preprocessor(qc: QuantumCircuit) -> QuantumCircuit:
    """Preprocesses a quantum circuit by applying disentangling, grouping, and reordering operations."""
    # The three passes run in this order for a reason:
    #
    # 1. Disentangling inserts markers telling the simulator where the state may be
    #    split into independent branches. It goes first because the later passes treat
    #    those markers as ordinary non-unitary instructions, which is what lets the
    #    reordering pass pull them forward.
    # 2. Gate grouping merges runs of small gates into single medium-sized unitaries,
    #    which is where most of the simulation speed-up comes from. It also shortens the
    #    instruction list considerably, so running it before the reordering pass means
    #    the causal graph built there is correspondingly smaller.
    # 3. Reordering moves measurements, resets and disentanglers as early as it can, so
    #    that branches which turn out to have vanishing probability are abandoned before
    #    the remaining gates are applied to them. It runs last so that its ordering is
    #    final: the grouping pass is free to move gates around, and would otherwise be
    #    able to undo the ordering established here.
    if len(qc.data) == 0:
        return qc.copy()

    # TO-DO find reliable classifiaction when automatic disentangling works best
    if len(qc.qubits) > _DISENTANGLING_QUBIT_THRESHOLD:
        qc = _insert_disentangling(qc)

    qc = _group_qc(qc)
    return _reorder_circuit(qc, ["measure", "reset", "disentangle"])
