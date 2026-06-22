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
def remove_barriers(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of *qc* with all barrier instructions removed.

    Parameters
    ----------
    qc : QuantumCircuit
        The input circuit, potentially containing barrier instructions.

    Returns
    -------
    QuantumCircuit
        A new circuit identical to *qc* but without any barriers.

    Examples
    --------
    Remove a barrier from a circuit::

        >>> from qrisp import QuantumCircuit, PassManager
        >>> from qrisp import remove_barriers
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0)
        >>> qc.barrier()
        >>> qc.cx(0, 1)
        >>> print(qc)
                ┌───┐ ░
        qb_124: ┤ H ├─░───■──
                └───┘ ░ ┌─┴─┐
        qb_125: ──────░─┤ X ├
                      ░ └───┘

        >>> pm = PassManager()
        >>> pm += remove_barriers
        >>> clean_qc = pm.run(qc)
        >>> print(clean_qc)
                ┌───┐
        qb_124: ┤ H ├──■──
                └───┘┌─┴─┐
        qb_125: ─────┤ X ├
                     └───┘

    """
    qc_new = qc.clearcopy()
    for instr in qc.data:
        if instr.op.name == "barrier":
            continue
        qc_new.append(instr)
    return qc_new
