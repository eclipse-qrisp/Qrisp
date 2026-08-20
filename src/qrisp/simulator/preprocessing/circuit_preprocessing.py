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

================================================================================
The Main Wrapper: circuit_preprocessor(qc)
================================================================================

The main entry point for the :mod:`~qrisp.simulator.preprocessing` package. It
evaluates the incoming circuit, applies disentangling (if the circuit is
dangerously wide, e.g., >45 qubits), groups the gates for performance, and
finally reorders the circuit to safely push measurements, resets, and
disentanglers to the end of execution blocks.
================================================================================
"""

from qrisp.circuit import QuantumCircuit
from qrisp.simulator.preprocessing.disentangling import insert_disentangling
from qrisp.simulator.preprocessing.gate_grouping import group_qc


def circuit_preprocessor(qc: QuantumCircuit) -> QuantumCircuit:
    """Preprocesses a quantum circuit by applying disentangling, grouping, and reordering operations."""
    from qrisp.simulator import reorder_circuit

    if len(qc.data) == 0:
        return qc.copy()

    # TO-DO find reliable classifiaction when automatic disentangling works best
    if len(qc.qubits) > 45:
        qc = insert_disentangling(qc)

    qc = group_qc(qc)
    return reorder_circuit(qc, ["measure", "reset", "disentangle"])
