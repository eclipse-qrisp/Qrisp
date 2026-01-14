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

import numpy as np
from jax.errors import TracerArrayConversionError


def prepare_qiskit(qv, target_array, reversed=False):
    
    try:
        target_array = np.array(target_array)
    except TracerArrayConversionError:
        raise ValueError("Tried to initialize dynamic jax array using state preparation method qiskit")

    from qiskit.circuit.library.data_preparation.state_preparation import (
        StatePreparation,
    )

    n = len(target_array)
    m = int(np.ceil(np.log2(n)))

    # Fill target_array with zeros
    if not (n & (1 << m)):
        target_array = np.concatenate((target_array, np.zeros((1 << m) - n)))

    target_array = target_array / np.vdot(target_array, target_array) ** 0.5

    qiskit_qc = StatePreparation(target_array).definition
    from qrisp import QuantumCircuit

    init_qc = QuantumCircuit.from_qiskit(qiskit_qc)

    # Find global phase correction
    from qrisp.simulator import statevector_sim

    init_qc.qubits.reverse()
    sim_array = statevector_sim(init_qc)
    init_qc.qubits.reverse()

    arg_max = np.argmax(np.abs(sim_array))

    gphase_dif = (np.angle(target_array[arg_max] / sim_array[arg_max])) % (2 * np.pi)

    init_qc.gphase(gphase_dif, 0)

    init_gate = init_qc.to_gate()

    init_gate.name = "state_init"

    if reversed:
        qv.qs.append(init_gate, [qv[m - 1 - i] for i in range(m)])
    else:
        qv.qs.append(init_gate, [qv[i] for i in range(m)])
