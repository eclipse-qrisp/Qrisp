"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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

from qrisp import QuantumCircuit
from qrisp.interface import VirtualBackend

def run_on_stim(qc: QuantumCircuit, shots: int):
    """
    Simulate a Qrisp QuantumCircuit via Stim.

    Parameters
    ----------
    qc : QuantumCircuit
        The Qrisp quantum circuit to simulate.
    shots : int
        Number of measurement shots to perform.

    Returns
    -------
    dict[str, int]
        Dictionary mapping bitstrings to counts.
    """
    stim_circuit, measurement_map = qc.to_stim(return_measurement_map=True)

    sampler = stim_circuit.compile_sampler()
    shot_array = sampler.sample(shots=shots)

    # Reorder columns to match Qrisp clbit order
    permutation = [measurement_map[cb] for cb in qc.clbits]
    permuted_shot_array = shot_array[:, permutation]

    # Convert each row to a bitstring and count occurrences
    counts = {}
    for row in permuted_shot_array:
        # Convert boolean array to bitstring (reversed for little-endian convention)
        bitstring = "".join(str(int(b)) for b in reversed(row))
        counts[bitstring] = counts.get(bitstring, 0) + 1

    return counts

StimBackend = VirtualBackend(run_on_stim)
