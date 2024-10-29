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
from qrisp.operators.pauli import X,Y,Z

def test_fermionic_to_pauli():

    # Check if transformation works for both, reduced and non-reduced FermionicHamiltonians

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