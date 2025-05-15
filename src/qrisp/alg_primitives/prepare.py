"""
\********************************************************************************
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
********************************************************************************/
"""

from qrisp import QuantumFloat
import numpy as np


def prepare(qv, target_array, reversed=False):
    r"""
    This method performs quantum state preparation. Given a vector $b=(b_0,\dotsc,b_{N-1})$, the function acts as

    .. math::

        \ket{0} \rightarrow \sum_{i=0}^{N-1}b_i\ket{i}

    Parameters
    ----------
    qv : QuantumVariable
        The quantum variable on which to apply state preparation.
    target_array : numpy.ndarray
        The vector $b$.
    reversed : boolean
        If set to ``True``, the endianness is reversed. The default is ``False``.

    Examples
    --------

    We create a :ref:`QuantumFloat` and prepare the state $\sum_{i=0}^3b_i\ket{i}$ for $b=(0,1,2,3)$.

    ::

        b = np.array([0,1,2,3])

        qf = QuantumFloat(2)
        prepare(qf, b)

        res_dict = qf.get_measurement()

        for k, v in res_dict.items():
            res_dict[k] = v**0.5

        for k, v in res_dict.items():
            res_dict[k] = v/res_dict[1.0]

        print(res_dict)
        # Yields: {3: 2.9999766670425863, 2: 1.999965000393743, 1: 1.0}

    """

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
