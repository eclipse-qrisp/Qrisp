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

import numpy as np

from qrisp import LCU, QuantumVariable, prepare, terminal_sampling
from qrisp.operators import X, Y, Z


def test_operators_LCU():

    @terminal_sampling
    def main():

        H = 2 * X(0) * X(1) - Z(0) * Z(1)

        unitaries, coeffs = H.unitaries()

        def operand_prep():
            return QuantumVariable(2)

        def state_prep(case):
            prepare(case, np.sqrt(coeffs))

        qv = LCU(operand_prep, state_prep, unitaries)
        return qv

    res_dict = main()

    for k, v in res_dict.items():
        res_dict[k] = v**0.5
    for k, v in res_dict.items():
        res_dict[k] = v / res_dict[0]

    # print(res_dict)
    # {3: 2.0000002235174685, 0: 1.0}

    assert (np.abs(res_dict[0] - 1)) < 1e-4
    assert (np.abs(res_dict[3] - 2)) < 1e-4
