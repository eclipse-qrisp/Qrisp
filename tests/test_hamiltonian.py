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

def test_pauli_hamiltonian():

    from qrisp import QuantumVariable, QuantumArray, h
    from qrisp.operators.qubit import X,Y,Z
    import numpy as np
            
    qv = QuantumVariable(2)
    h(qv)
    H = Z(0)*Z(1)
    res = H.get_measurement(qv)
    assert np.abs(res-0.0) < 2e-2

    qtype = QuantumVariable(2)
    q_array = QuantumArray(qtype, shape=(2))
    h(q_array)
    H = Z(0)*Z(1) + X(2)*X(3)
    res = H.get_measurement(q_array)
    assert np.abs(res-1.0) < 2e-2

def test_bound_pauli_hamiltonian():

    from qrisp import QuantumVariable, QuantumArray, h, x
    from qrisp.operators.qubit import X,Y,Z
    import numpy as np

    qv1 = QuantumVariable(2)
    qv2 = QuantumVariable(2)

    h(qv1[0])
    x(qv2[0])

    H = X(qv1[0])*Z(qv2[0])
    res = H.get_measurement([qv1,qv2])
    assert np.abs(res-(-1.0)) < 2e-2

def test_trotterization():

    from qrisp import QuantumVariable, x, QPE
    from qrisp.operators.qubit.pauli import X,Y,Z
    import numpy as np

    # Hydrogen https://arxiv.org/abs/1704.05018
    G = 0.011280*Z(0)*Z(1) + 0.397936*Z(0) + 0.397936*Z(1) + 0.180931*X(0)*X(1)
    E0 = G.ground_state_energy()
    assert np.abs(E0-(-0.804899065613056)) < 2e-2

    U = G.trotterization()

    qv = QuantumVariable(2)
    x(qv) 
    E1 = G.get_measurement(qv)
    assert np.abs(E1-(-0.78)) < 2e-2

    # Find minimum Eigenvalue with Hamiltonian simulation + QPE
    qv = QuantumVariable(2)
    x(qv) # Initial state close to exact solution
    res = QPE(qv,U,precision=10,kwargs={"steps":3},iter_spec=True)
    meas = res.get_measurement()
    sorted_meas = dict(sorted(meas.items(), key=lambda item: item[1], reverse=True))
    phi = list(sorted_meas.items())[0][0]
    E2 = 2*np.pi*(phi-1)
    assert np.abs(E0-E2) < 2e-2