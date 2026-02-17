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
from qrisp.interface import BatchedBackend


def run_on_stim(qc: QuantumCircuit, shots: int | None):
    """
    Simulate a Qrisp QuantumCircuit via Stim.

    Parameters
    ----------
    qc : QuantumCircuit
        The Qrisp quantum circuit to simulate.
    shots : int, optional
        Number of measurement shots to perform. If None, defaults to 10000.

    Returns
    -------
    dict[str, int]
        Dictionary mapping bitstrings to counts.
    """

    if shots is None:
        shots = 10000

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


def run_on_stim_batch(batch):
    """
    Simulate a batch of Qrisp QuantumCircuits via Stim.

    Parameters
    ----------
    batch : list[tuple[QuantumCircuit, int]]
        List of (circuit, shots) pairs.

    Returns
    -------
    list[dict[str, int]]
        List of result dictionaries, one per circuit.
    """
    return [run_on_stim(qc, shots) for qc, shots in batch]


def StimBackend():
    """
    This function creates a :ref:`BatchedBackend` for simulating Qrisp QuantumCircuits
    using the Stim simulator.
    
    `Stim <https://github.com/quantumlib/Stim>`_ is a fast stabilizer circuit simulator
    designed for quantum error correction research. It efficiently simulates Clifford
    circuits and is particularly well-suited for simulating quantum error correction codes
    with thousands of qubits and millions of gates.

    Returns
    -------
    BatchedBackend
        A backend instance that dispatches circuit simulation to Stim.

    Examples
    --------

    Basic usage with a QuantumVariable:

    ::

        from qrisp import QuantumVariable
        from qrisp.interface import StimBackend

        qv = QuantumVariable(2)
        qv[:] = "10"
        res = qv.get_measurement(backend=StimBackend())
        print(res)
        # Yields: {'10': 1.0}

    """
    return BatchedBackend(run_on_stim_batch)
