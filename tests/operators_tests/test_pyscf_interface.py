"""
********************************************************************************
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
********************************************************************************
"""

from qrisp import QuantumVariable
from qrisp.operators import FermionicOperator

def test_pyscf_interface():

    try:
        from pyscf import gto
    except:
        return
    
    mol = gto.M(atom = '''H 0 0 0; H 0 0 0.74''', basis = 'sto-3g')
    H = FermionicOperator.from_pyscf(mol)
    U = H.trotterization()
    
    electron_state = QuantumVariable(4)
    electron_state[:] = "0011"
    U(electron_state, t = 100, steps = 20)
    
    mol = gto.M(atom = '''Li     0.0000     0.0000     0.0000; H      1.595     0.0000     0.0000''', basis = 'sto-3g') # DOES NOT WORK

    H_ferm = FermionicOperator.from_pyscf(mol) # DOES NOT WORK
    
    qv = QuantumVariable(12)
    H_ferm.get_measurement(qv)
