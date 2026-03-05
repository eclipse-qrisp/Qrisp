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

from qrisp.operators import *
from qrisp import QuantumFloat
from qrisp.jasp import terminal_sampling

def test_jasp_hamiltonian_simulation():
    
    def test_hamiltonian(H):
        
        def main():
            qv = QuantumFloat(H.find_minimal_qubit_amount())
            qv[:] = 5
            U = H.trotterization()
            U(qv, 1.5, steps = 1)
            return qv
        
        jasp_res = terminal_sampling(main)()
        qrisp_res = main().get_measurement()
        
        for k in jasp_res.keys():
            assert abs(jasp_res[k] - qrisp_res[int(k)]) < 1E-3
            
    H = Y(0)*X(1)*Z(2)
    test_hamiltonian(H)
    H = Y(0)*X(1)*Z(2) + Z(0)*Z(1)*Z(2)
    test_hamiltonian(H)
    H = A(0)*A(1)*A(2)*C(3)
    test_hamiltonian(H)
    H = A(0)*A(1)*A(2)*C(3) + Z(0)*Z(1)*A(3)        
    test_hamiltonian(H)
    H = a(0)*c(2)*a(3)*a(4) + c(0)*a(1)*a(3)*c(2)
    test_hamiltonian(H)



from qrisp import *
from qrisp.operators import FermionicOperator
import numpy as np

def test_jasp_hamiltonian_simulation_H2():
    
    try:
        from pyscf import gto
    except ImportError:
        return

    # Finding the gound state energy of the Hydrogen molecule with QPE
    @terminal_sampling
    def main():

        mol = gto.M(
        atom = '''H 0 0 0; H 0 0 0.74''',
        basis = 'sto-3g')

        H = FermionicOperator.from_pyscf(mol).to_qubit_operator()

        U = H.trotterization(forward_evolution=False, method='commuting')

        qv = QuantumFloat(H.find_minimal_qubit_amount())
        [x(qv[i]) for i in range(2)] # Prepare Hartree-Fock state, H2 molecule has 2 electrons

        qpe_res = QPE(qv,U,precision=6,kwargs={"steps":3})
        return qpe_res

    meas_res = main()
    
    phi = list(meas_res.keys())[0]
    E = 2*np.pi*(phi-1)
    success_probability = meas_res[phi]

    assert np.abs(E-(-1.865)) < 1e-3
    assert success_probability > 0.9