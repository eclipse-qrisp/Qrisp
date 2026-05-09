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
def visualize(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Print the current circuit to ``stdout`` for debugging purposes.

    A no-op pass that prints the circuit as it is when encountered in a
    :class:`~qrisp.PassManager` pipeline.  Useful for inspecting the
    circuit between other transformation passes.

    The pass is always the identity — it never modifies the circuit.

    Example
    -------
    >>> from qrisp import PassManager, visualize, decompose
    >>> pm = PassManager()
    >>> pm += decompose()
    >>> pm += visualize
    >>> pm.run(qc)
    """
    print(qc)
    return qc
