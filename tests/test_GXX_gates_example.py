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

# Created by ann81984 at 05.05.2022
import pytest
import numpy as np

from qrisp import QuantumSession, QuantumVariable
from qrisp.misc.GMS_tools import (
    gms_multi_cx_fan_out,
    gms_multi_cx_fan_in,
    gms_multi_cp_gate,
    gms_multi_cp_gate_mono_phase,
    GXX_wrapper,
)
from qrisp import x


def test_GXX_gates_example():
    n = 5
    qs = QuantumSession()
    qv = QuantumVariable(n, qs)

    x(qv[-1])
    qs.append(
        gms_multi_cx_fan_out(
            n - 1, use_uniform=True, phase_tolerant=False, basis="GXX"
        ),
        qv.reg,
    )

    tmp = qv.get_measurement()

    print(tmp)
    assert qv.get_measurement() == {"11111": 1.0}

    ###################
    n = 3

    theta = [np.pi / (i + 1) for i in range(n - 1)]
    qs_0 = QuantumSession()
    qv = QuantumVariable(n, qs_0)
    qs_0.append(gms_multi_cp_gate(n - 1, theta), qv.reg)

    qs_1 = QuantumSession()
    qv = QuantumVariable(n, qs_1)

    for i in range(n - 1):
        qs_1.cp(theta[i], i, n - 1)

    unitary = qs_0.get_unitary()
    print(qs_0.get_unitary(decimals=4))
    tmparray = qs_0.get_unitary(decimals=4)
    assert type(qs_0.get_unitary(decimals=4)).__module__ == "numpy"
    while i < len(tmparray):
        assert all(tmparray[i]) in [
            1.0 + 0.0j,
            -0.0 - 0.0j,
            -0.0 + 0.0j,
            0.0 + 0.0j,
            1.0 - 0.0j,
            0.0 - 0.0j,
            0.0 + 1.0j,
            -1.0 + 0.0j,
            -0.0 - 1.0j,
        ]
        assert not bool(all(tmparray[i]) == 7153)
        i += 1

    print("Is equal:", qs_0.compare_unitary(qs_1))
    assert qs_0.compare_unitary(qs_1)

    print(
        "Is diagonal:", np.linalg.norm(unitary - np.diag(np.diagonal(unitary))) < 1e-3
    )
    assert np.linalg.norm(unitary - np.diag(np.diagonal(unitary))) < 1e-3

    print(np.round(np.angle(np.diagonal(unitary)) % (2 * np.pi) / (np.pi), 3))
