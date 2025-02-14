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

def test_amplitude_amplification():
    from qrisp import QuantumBool, ry, z, amplitude_amplification
    import numpy as np

    def state_function(qb):
        ry(np.pi/8,qb)

    def oracle_function(qb):   
        z(qb)

    qb = QuantumBool()
    state_function(qb)
    assert np.round(qb.get_measurement()[True],2) == 0.04

    amplitude_amplification([qb], state_function, oracle_function, iter=1)
    assert np.round(qb.get_measurement()[True],2) == 0.31

    amplitude_amplification([qb], state_function, oracle_function, iter=1)
    assert np.round(qb.get_measurement()[True],2) == 0.69

    amplitude_amplification([qb], state_function, oracle_function, iter=1)
    assert np.round(qb.get_measurement()[True],2) == 0.96

