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

from qrisp.vqe.problems.electronic_structure import *
from qrisp import QuantumVariable
import numpy as np

#
# H2 molecule
#

def test_vqe_electronic_structure_H2():  
    
    try:
        from pyscf import gto
    except:
        return

    mol = gto.M(
        atom = '''H 0 0 0; H 0 0 0.74''',
        basis = 'sto-3g')
    
    H = create_electronic_hamiltonian(mol).to_qubit_operator()
    assert np.abs(H.ground_state_energy()-(-1.85238817356958))

    vqe = electronic_structure_problem(mol)
    
    results = []
    for i in range(5):
        res = vqe.run(QuantumVariable(4),
                depth=1,
                max_iter=50)
        results.append(res)
    
    assert np.abs(min(results)-(-1.85238817356958)) < 1e-1

#
# BeH2 molecule, active space
#
"""
def test_vqe_electronic_structure_BeH2():

    mol = gto.M(
        atom = f'''Be 0 0 0; H 0 0 3.0; H 0 0 -3.0''',
        basis = 'sto-3g')
    
    H = create_electronic_hamiltonian(mol,active_orb=6,active_elec=4).to_qubit_operator()
    assert np.abs(H.ground_state_energy()-(-16.73195995959339))

    # runs for >1 minute
    vqe = electronic_structure_problem(mol,active_orb=6,active_elec=4)
    
    results = []
    for i in range(5):
        res = vqe.run(QuantumVariable(6),
                depth=1,
                max_iter=50)
        results.append(res)
    
    print(min(results))
    assert np.abs(min(results)-(-16.73195995959339)) < 1e-1
"""