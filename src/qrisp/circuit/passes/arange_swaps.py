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


def arange_swaps(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Optimize SWAP gate arrangements for better compilation efficiency.

    This pass rearranges SWAP gates to enable better gate cancellation in later
    compilation steps. When a SWAP involves an unused qubit, the qubit order is
    reversed so the unused qubit comes first, allowing the first CX gate in the
    SWAP decomposition to be controlled on the unused qubit for later cancellation.

    Parameters
    ----------
    qc : QuantumCircuit
        Input quantum circuit.

    Returns
    -------
    QuantumCircuit
        Optimized circuit with rearranged SWAP gates.

    Optimization Rules
    ------------------
    - If SWAP(used_qubit, unused_qubit): reorder to SWAP(unused_qubit, used_qubit)
    - If both qubits unused: filter out the SWAP entirely
    - If both qubits used: preserve original order

    Example
    -------
    >>> from qrisp import PassManager, arange_swaps
    >>> pm = PassManager()
    >>> pm.add_pass(arange_swaps)
    >>> transpiled_qc = pm.run(qc)
    """
    qc_new = qc.clearcopy()

    used_qubits = set()
    for i in range(len(qc.data)):

        instr = qc.data[i]

        # Skip allocation instructions
        if "alloc" in instr.op.name:
            continue

        if instr.op.name == "swap":
            # Check if second qubit is unused
            if instr.qubits[1] not in used_qubits:
                # If both qubits are unused, skip this SWAP entirely
                if instr.qubits[0] not in used_qubits:
                    continue
                # If first qubit is used but second is unused, reverse order
                # This puts the unused qubit first for optimal CX decomposition
                instr = instr.copy()
                instr.qubits = instr.qubits[::-1]

        # Track which qubits have been used
        used_qubits = used_qubits.union(instr.qubits)

        qc_new.append(instr)

    return qc_new
