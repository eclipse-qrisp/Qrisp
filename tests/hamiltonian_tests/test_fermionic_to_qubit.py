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

from qrisp.operators.fermionic import a, c
from qrisp.operators.qubit import X,Y,Z
from qrisp.vqe.problems.electronic_structure import *

def test_fermionic_to_qubit():

    try:
        from pyscf import gto
    except:
        return    

    # Check if transformation works for both, reduced and non-reduced FermionicOperators

    H = c(0)*c(1)*a(3)*a(2) + c(2)*c(3)*a(1)*a(0)

    G1 = H.to_pauli_hamiltonian()

    H.reduce()

    G2 = H.to_pauli_hamiltonian()

    assert str(G1-G2)=='0'

    K = (1/8)*(X(0)*X(1)*X(2)*X(3) - X(0)*X(1)*Y(2)*Y(3) \
                + X(0)*Y(1)*X(2)*Y(3) + X(0)*Y(1)*Y(2)*X(3) \
                + Y(0)*X(1)*X(2)*Y(3) + Y(0)*X(1)*Y(2)*X(3) \
                - Y(0)*Y(1)*X(2)*X(3) + Y(0)*Y(1)*Y(2)*Y(3))
    
    assert str(G1-K)=='0'


def test_hamiltonian_H2():
    
    try:
        from pyscf import gto
    except:
        return

    K = -0.812170607248714 -0.0453026155037992*X(0)*X(1)*Y(2)*Y(3) +0.0453026155037992*X(0)*Y(1)*Y(2)*X(3) +0.0453026155037992*Y(0)*X(1)*X(2)*Y(3) -0.0453026155037992*Y(0)*Y(1)*X(2)*X(3) \
        +0.171412826447769*Z(0) +0.168688981703612*Z(0)*Z(1) +0.120625234833904*Z(0)*Z(2) +0.165927850337703*Z(0)*Z(3) +0.171412826447769*Z(1) \
        +0.165927850337703*Z(1)*Z(2) +0.120625234833904*Z(1)*Z(3) -0.223431536908133*Z(2) +0.174412876122615*Z(2)*Z(3) -0.223431536908133*Z(3)

    mol = gto.M(
        atom = '''H 0 0 0; H 0 0 0.74''',
        basis = 'sto-3g')
    
    H = create_electronic_hamiltonian(mol).to_pauli_hamiltonian()

    G = K-H
    G.apply_threshold(1e-4)
    assert str(G)=='0'