"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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

# Created by ann81984 at 07.05.2022
import time

from qrisp.core import QuantumSession
from qrisp.arithmetic import QuantumFloat
from numpy.linalg import norm
from qiskit import execute, Aer
from qrisp.interface import convert_circuit


def test_unitary_calculation():
    n = 2

    qs = QuantumSession()

    x = QuantumFloat(n, 0, qs, signed=True)
    y = QuantumFloat(n, 0, qs, signed=False)

    s = x * y
    # qs = s.qs
    qc = qs.compile(1)

    start = time.time()
    test_unitary_1 = qc.get_unitary()
    end = time.time()

    print("Qrisp calculation time = %s" % (end - start))

    ###################
    qc.qubits = qc.qubits[::-1]
    qiskit_qc = convert_circuit(qc, target_api="qiskit")

    backend = Aer.get_backend("unitary_simulator")

    start = time.time()
    job = execute(qiskit_qc, backend)
    result = job.result()
    test_unitary_2 = result.get_unitary(qiskit_qc).data
    end = time.time()

    print("Qiskit calculation time = %s" % (end - start))

    # get the unitary matrix from the result object
    print("Unitaries matching:", bool(norm(test_unitary_1 - test_unitary_2) < 1e-3))
    assert bool(norm(test_unitary_1 - test_unitary_2) < 1e-3)
