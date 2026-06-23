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

from qrisp.circuit.pass_management.circuit_pass import CircuitPass
from qrisp.circuit.quantum_circuit import QuantumCircuit


@CircuitPass
def visualize(qc: QuantumCircuit) -> QuantumCircuit:
    """Print the current circuit to ``stdout`` for debugging purposes.

    A no-op pass that prints the circuit as it is when encountered in a
    :class:`~qrisp.PassManager` pipeline.  Useful for inspecting the
    circuit between other transformation passes.

    The pass is always the identity — it never modifies the circuit.

    Examples
    --------
    We showcase how a Toffoli gate is decomposed step by step, inserting
    a visualization pass after each pass to show the progression.

    >>> from qrisp import PassManager, visualize, decompose, QuantumCircuit
    >>> qc = QuantumCircuit(3)
    >>> qc.ccx(0, 1, 2)
    >>> pm = PassManager()
    >>> pm += visualize
    >>> pm += decompose(1)
    >>> pm += visualize
    >>> pm += decompose(1)
    >>> pm += visualize
    >>> pm.run(qc)
    <BLANKLINE>
    qb_82: ──■──
             │
    qb_83: ──■──
           ┌─┴─┐
    qb_84: ┤ X ├
           └───┘
           ┌────────────────┐
    qb_82: ┤0               ├
           │                │
    qb_83: ┤1 gray multi cx ├
           │                │
    qb_84: ┤2               ├
           └────────────────┘
           ┌─────┐
    qb_82: ┤ Tdg ├───────■─────────■────■───────────────────────■──
           ├─────┤┌───┐  │  ┌───┐┌─┴─┐  │  ┌─────┐┌───┐ ┌───┐ ┌─┴─┐
    qb_83: ┤ Tdg ├┤ X ├──┼──┤ T ├┤ X ├──┼──┤ Tdg ├┤ X ├─┤ T ├─┤ X ├
           └┬───┬┘└─┬─┘┌─┴─┐├───┤└───┘┌─┴─┐└─────┘└─┬─┘┌┴───┴┐├───┤
    qb_84: ─┤ H ├───■──┤ X ├┤ T ├─────┤ X ├─────────■──┤ Tdg ├┤ H ├
            └───┘      └───┘└───┘     └───┘            └─────┘└───┘

    """
    print(qc)
    return qc
