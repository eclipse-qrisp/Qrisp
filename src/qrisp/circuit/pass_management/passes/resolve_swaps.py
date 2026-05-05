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
from qrisp.circuit.pass_management.circuit_pass import CircuitPass


@CircuitPass
def resolve_swaps(qc: QuantumCircuit) -> QuantumCircuit:
    """Remove SWAP gates by folding them into qubit index remapping.

    Every SWAP encountered updates an internal permutation table.  All
    subsequent instructions have their qubit operands remapped through
    this table, so the logical effect of the SWAP is preserved without
    any physical gates.

    This pass should be applied **before** routing.  Instead of decomposing
    each SWAP into three CX gates, it tracks a running qubit permutation and
    remaps all subsequent gate operands accordingly.  After processing, the
    circuit contains no SWAP gates and produces the same unitary up to a final
    qubit permutation (which the router will handle).

    Parameters
    ----------
    qc : QuantumCircuit
        Input circuit, potentially containing ``swap`` instructions.

    Returns
    -------
    QuantumCircuit
        A new circuit with all SWAP gates removed and operands remapped.

    Example
    -------
    >>> from qrisp import PassManager, resolve_swaps
    >>> pm = PassManager()
    >>> pm.add_pass(resolve_swaps)
    >>> routable_qc = pm.run(qc)
    """
    n = qc.num_qubits()
    qubits = qc.qubits

    # perm[i] = current physical qubit index that logical qubit i maps to.
    # Starts as the identity.
    perm = list(range(n))

    qc_new = qc.clearcopy()

    for instr in qc.data:
        if instr.op.name == "swap":
            # Update the permutation: swap the two entries.
            a = qubits.index(instr.qubits[0])
            b = qubits.index(instr.qubits[1])
            perm[a], perm[b] = perm[b], perm[a]
        else:
            # Remap qubit operands through the current permutation.
            mapped_qubits = [qc_new.qubits[perm[qubits.index(q)]] for q in instr.qubits]
            qc_new.append(instr.op, mapped_qubits, instr.clbits)

    return qc_new
