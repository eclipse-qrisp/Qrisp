"""
\********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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

from qrisp import QuantumVariable, x, QPE
from qrisp.operators.pauli.pauli import X,Y,Z
import numpy as np

def test_pauli_hamiltonian_simulation():

    # Hydrogen (reduced 2 qubit Hamiltonian)
    H = -1.05237325 + 0.39793742*Z(0) -0.39793742*Z(1) -0.0112801*Z(0)*Z(1) + 0.1809312*X(0)*X(1)
    E0 = H.ground_state_energy()
    assert abs(E0-(-1.857275029288228))<1e-2

    U = H.trotterization()

    # Find minimum eigenvalue of H with Hamiltonian simulation and QPE

    # ansatz state
    qv = QuantumVariable(2)
    x(qv[0])
    E1 = H.get_measurement(qv)
    assert abs(E1-E0)<5e-2

    qpe_res = QPE(qv,U,precision=10,kwargs={"steps":3},iter_spec=True)

    results = qpe_res.get_measurement()    
    sorted_results= dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    
    phi = list(sorted_results.items())[0][0]
    E_qpe = 2*np.pi*(phi-1) # Results are modulo 2*pi, therefore subtract 2*pi 
    assert abs(E_qpe-E0)<5e-3
