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
def arrange_swaps(qc: QuantumCircuit) -> QuantumCircuit:
    r"""
    Flip SWAP qubit order so that unused qubits come first.

    A SWAP gate decomposes into three CX gates.  When a SWAP involves a
    qubit that has not yet been touched by any prior operation, that qubit
    is in the \|0⟩ state, and the first CX in the decomposition is
    controlled on \|0⟩ — a no-op.

    This pass reorders the qubits of each SWAP so that any untouched qubit
    appears first.  This exposes the redundant CX to downstream passes such
    as :func:`~qrisp.cancel_zero_controls`, which can then remove it.

    After reordering, SWAPs that involve **two** untouched qubits are
    dropped entirely (a SWAP between \|0⟩ states does nothing).  SWAPs
    where both qubits have already been used are left in their original
    order.

    Parameters
    ----------
    qc : QuantumCircuit
        Input quantum circuit.

    Returns
    -------
    QuantumCircuit
        Circuit with SWAP qubit order flipped where beneficial.

    Examples
    --------
    A SWAP(0, 1) where qubit 1 hasn't been touched yet.

    Setup::

        >>> from qrisp import QuantumCircuit, PassManager, decompose
        >>> from qrisp import arrange_swaps, cancel_zero_controls
        >>>
        >>> qc = QuantumCircuit(2)
        >>> qc.x(0)
        >>> qc.swap(0, 1)          # qubit 1 is untouched

    Without ``arrange_swaps``, the SWAP decomposes into three CX gates.
    The first CX targets the unused qubit (qubit 1), but since that qubit
    is still in \|0⟩ the CX is a no-op — it remains in the circuit anyway::

        >>> pm_raw = PassManager()
        >>> pm_raw += decompose()
        >>> print(pm_raw.run(qc)) # doctest: +SKIP
              ┌───┐     ┌───┐     
        qb_0: ┤ X ├──■──┤ X ├──■──
              └───┘┌─┴─┐└─┬─┘┌─┴─┐
        qb_1: ─────┤ X ├──■──┤ X ├
                   └───┘     └───┘

    With ``arrange_swaps`` the qubit order is flipped so the unused qubit
    comes first.  ``cancel_zero_controls`` then removes the first CX
    (controlled on \|0⟩).  After decomposition, one CX is gone::

        >>> pm = PassManager()
        >>> pm += arrange_swaps
        >>> pm += decompose()
        >>> pm += cancel_zero_controls
        >>> print(pm.run(qc))
              ┌───┐     ┌───┐
        qb_0: ┤ X ├──■──┤ X ├
              └───┘┌─┴─┐└─┬─┘
        qb_1: ─────┤ X ├──■──
                   └───┘     

    A SWAP between **two** untouched qubits is dropped entirely (a SWAP
    between \|0⟩ states is the identity):

    >>> qc = QuantumCircuit(2)
    >>> qc.swap(0, 1)
    >>> result = pm.run(qc)
    >>> print(pm.run(qc))
    <BLANKLINE>
    qb_0: 
    <BLANKLINE>
    qb_1: 
        
    """
    qc_new = qc.clearcopy()

    used_qubits = set()
    for instr in qc.data:

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
        used_qubits.update(instr.qubits)

        qc_new.append(instr)

    return qc_new
