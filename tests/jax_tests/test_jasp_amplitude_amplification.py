"""
********************************************************************************
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
********************************************************************************
"""


def test_jasp_amplitude_amplification():
    import numpy as np

    from qrisp import QuantumFloat, amplitude_amplification, ry, z
    from qrisp.jasp import terminal_sampling

    def state_function(qb):
        ry(np.pi / 8, qb)

    def oracle_function(qb):
        z(qb)

    @terminal_sampling
    def main(i):
        qb = QuantumFloat(1)
        state_function(qb)
        amplitude_amplification([qb], state_function, oracle_function, iter=i)
        return qb

    assert np.round(main(0)[1], 2) == 0.04
    assert np.round(main(1)[1], 2) == 0.31
    assert np.round(main(2)[1], 2) == 0.69
    assert np.round(main(3)[1], 2) == 0.96
