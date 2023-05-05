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

# Created by ann81984 at 29.04.2022
import numpy as np
import random
from qrisp import QuantumSession, QuantumVariable, p, cp
from qrisp.misc.GMS_tools import GXX_converter


def test_GMS_converter_example():
    n = 8

    qv = QuantumVariable(n)

    # Generate random phase-only circuit
    for i in range(n):
        for j in range(n):
            if random.randint(0, 1):
                if i == j:
                    p(np.pi / 2 ** random.randint(0, 3), qv[i])
                else:
                    cp(np.pi / 2 ** random.randint(0, 3), qv[i], qv[j])

    print(qv.qs)

    # Apply converter
    qc = GXX_converter(qv.qs)

    print("Is equal:", qv.qs.compare_unitary(qc, precision=4))

    unitary = qc.get_unitary()

    print(
        "Is diagonal:", np.linalg.norm(unitary - np.diag(np.diagonal(unitary))) < 1e-3
    )
    # print(np.round(np.angle(np.diagonal(qv.qs.get_unitary()))%(2*np.pi)/(2*np.pi), 3))
    assert qv.qs.compare_unitary(qc, precision=4)
    assert np.linalg.norm(unitary - np.diag(np.diagonal(unitary))) < 1e-3
