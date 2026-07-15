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

"""Reverse-order parallelization pass.

This pass reverses the instruction order, runs Qrisp's parallelization,
and restores the original order.  It is useful to group commuting gates
with later SWAP operations so cancellation passes can remove redundant
two-qubit gates.
"""

from __future__ import annotations

from qrisp.circuit.pass_management.circuit_pass import CircuitPass
from qrisp.circuit.quantum_circuit import QuantumCircuit


@CircuitPass
def reverse_parallelize(qc: QuantumCircuit) -> QuantumCircuit:
    """This pass leverages permeability commutations to move two qubit gates
    to a later point in the circuit. This is especially helpful when
    trying to cancel out SWAP gates with other two qubit interactions.

    Parameters
    ----------
    qc : QuantumCircuit
        Input quantum circuit.

    Returns
    -------
    QuantumCircuit
        Circuit after reverse-order parallelization.

    Examples
    --------
    We demonstrate how to move a CX gate towards a SWAP.

    >>> from qrisp import QuantumCircuit, PassManager
    >>> from qrisp import reverse_parallelize
    >>> qc = QuantumCircuit(2)
    >>> qc.cx(0, 1)
    >>> qc.z(0)
    >>> qc.swap(0, 1)
    >>> print(qc)
                 ┌───┐
    qb_130: ──■──┤ Z ├─X─
            ┌─┴─┐├───┤ │
    qb_131: ┤ X ├┤ X ├─X─
            └───┘└───┘

    >>> pm = PassManager()
    >>> pm += reverse_parallelize
    >>> optimized_qc = pm.run(qc)
    >>> print(optimized_qc)
           ┌───┐
    qb_66: ┤ Z ├──■───X─
           ├───┤┌─┴─┐ │
    qb_67: ┤ X ├┤ X ├─X─
           └───┘└───┘

    The CX gate can now be fused through the ``fuse_adjacents`` pass.

    """
    # Defer import: qrisp.permeability loads *after* qrisp.circuit, so
    # importing at module level would create a circular import.
    from qrisp.permeability import parallelize_qc

    # Make the one qubit gates slower than the two qubit ones
    # to make the parallelize pass execute the two qubit gates first.
    def depth_indicator(op):
        if op.num_qubits == 1:
            return 10
        return 1

    reversed_qc = qc.copy()
    reversed_qc.data.reverse()
    reversed_qc = parallelize_qc(reversed_qc, depth_indicator)
    reversed_qc.data.reverse()
    return reversed_qc
