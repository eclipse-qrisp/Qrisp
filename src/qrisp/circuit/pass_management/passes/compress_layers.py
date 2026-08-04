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

from __future__ import annotations

from collections import defaultdict

from qrisp.circuit.instruction import Instruction
from qrisp.circuit.pass_management.circuit_pass import CircuitPass
from qrisp.circuit.quantum_circuit import QuantumCircuit


@CircuitPass
def compress_layers(qc: QuantumCircuit) -> QuantumCircuit:
    """Reorder circuit instructions into "as-soon-as-possible" (ASAP) layers.

    Each instruction is assigned the earliest layer where all of its qubits
    (and classical bits) are free.  Instructions that act on disjoint sets of
    qubits can therefore appear in the same layer, which compacts the
    timeline.  The relative order of instructions that share a qubit is
    preserved, so the circuit semantics are unchanged.

    This pass is especially useful before exporting a circuit to Stim: Stim
    draws gates strictly in the order they appear in the circuit data, so
    calling this pass first yields a visually compacted timeline diagram.

    Parameters
    ----------
    qc : QuantumCircuit
        The input quantum circuit.

    Returns
    -------
    QuantumCircuit
        A new circuit with instructions reordered into ASAP layers.

    Examples
    --------
    >>> from qrisp import QuantumCircuit, PassManager, compress_layers
    >>> qc = QuantumCircuit(3)
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> qc.h(2)       # independent of qb_0, qb_1 — can move up
    >>> qc.cx(1, 2)
    >>> print(qc)
    ...
    >>> pm = PassManager()
    >>> pm += compress_layers
    >>> compact_qc = pm.run(qc)
    >>> print(compact_qc)
    ...

    The ``h(2)`` gate will be moved before ``cx(0, 1)`` because it acts on
    a disjoint qubit set.

    """
    # ------------------------------------------------------------------
    # Map each qubit / clbit to its latest occupied time-step.
    # Start at -1 so the first instruction touching a resource can be
    # placed at step 0.
    # ------------------------------------------------------------------
    last_time = defaultdict(lambda: -1)  # type: dict[object, int]

    # ------------------------------------------------------------------
    # First pass: compute the ASAP layer for every instruction.
    # ------------------------------------------------------------------
    layers: list[int] = []

    for instr in qc.data:
        # Collect all resources (qubits + clbits) touched by this
        # instruction.
        resources: list[object] = []
        resources.extend(instr.qubits)
        resources.extend(instr.clbits)

        # Bookkeeping instructions (qb_alloc / qb_dealloc) do not
        # correspond to physical quantum operations and should not
        # influence the scheduling of real gates.  They are placed at
        # the current layer of their qubits but do not advance the
        # clock.
        is_bookkeeping = instr.op.name in ("qb_alloc", "qb_dealloc")

        if resources:
            t = max(last_time[r] for r in resources)
        else:
            # Barrier-like instructions that touch no qubits: place them
            # after everything seen so far.
            t = max(last_time.values(), default=-1)

        if not is_bookkeeping:
            t += 1

        layers.append(t)

        if not is_bookkeeping:
            for r in resources:
                last_time[r] = t

    # ------------------------------------------------------------------
    # Second pass: sort instructions by their layer, preserving the
    # original order for instructions in the same layer (stable sort).
    # ------------------------------------------------------------------
    paired = list(zip(layers, qc.data, strict=False))
    paired.sort(key=lambda item: item[0])  # stable → original order kept
    reordered_data = [instr for _, instr in paired]

    # ------------------------------------------------------------------
    # Build the output circuit.
    # ------------------------------------------------------------------
    new_qc = qc.clearcopy()

    for instr in reordered_data:
        new_qc.append(instr)

    return new_qc
