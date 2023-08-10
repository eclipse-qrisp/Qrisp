"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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

# Created by ann81984 at 26.07.2022
from qrisp import QuantumVariable, QuantumFloat
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm


def test_state_preparation():
    qf = QuantumFloat(4, -2, signed=True)

    state_dic = {
        2.75: 1 / 4**0.5,
        -1.5: -1 / 4**0.5,
        2: 1 / 4**0.5,
        3: 1j / 4**0.5,
    }

    qf.init_state(state_dic)

    debugger = qf.qs.statevector("function")

    print(type(debugger))

    print("Amplitude of state 2.75: ", debugger({qf: 2.75}))
    print("Amplitude of state -1.5: ", debugger({qf: -1.5}))
    print("Amplitude of state 3: ", debugger({qf: 3}))

    assert np.abs(debugger({qf: 2.75}) - 0.5) < 1e-5
    assert np.abs(debugger({qf: -1.5}) + 0.5) < 1e-5
    assert np.abs(debugger({qf: 3}) - 0.5j) < 1e-5
