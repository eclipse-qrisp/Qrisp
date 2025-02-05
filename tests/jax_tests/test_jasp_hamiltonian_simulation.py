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
