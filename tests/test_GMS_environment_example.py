"""
/*********************************************************************
* Copyright (c) 2023 the Qrisp Authors
*
* This program and the accompanying materials are made
* available under the terms of the Eclipse Public License 2.0
* which is available at https://www.eclipse.org/legal/epl-2.0/
*
* SPDX-License-Identifier: EPL-2.0
**********************************************************************/
"""

# Created by ann81984 at 05.05.2022
from qrisp import QuantumVariable, x, h, merge
from qrisp.environments.GMS_environment import GMSEnvironment
import numpy as np


def test_GMS_environment_example():
    qv1 = QuantumVariable(1)
    qv2 = QuantumVariable(1)

    merge([qv1, qv2])

    qs = qv1.qs
    test = GMSEnvironment()

    h(qv1)
    x(qv2)
    with test:
        qs.cp(np.pi / 2, 0, 1)
        qs.p(np.pi / 2, 0)
        qs.p(np.pi / 2, 1)

    h(qv1)

    ###################
    from qrisp.arithmetic import QuantumFloat

    qf = QuantumFloat(3)

    qf.encode(1)

    from qrisp import QFT

    QFT(qf, use_gms=True)

    sv = qf.qs.statevector("array")
