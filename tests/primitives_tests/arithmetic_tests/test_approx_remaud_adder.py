"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

import random

from qrisp import *


def remaud_success_probability(method="khattar", mcx_kwargs=None):
    n = 4

    a = QuantumFloat(n)
    b = QuantumFloat(n)
    b_in = QuantumFloat(n)
    z = QuantumFloat(1)

    h(a)
    h(b)
    cx(b, b_in)

    remaud_adder(a, b, z, method=method, mcx_kwargs=mcx_kwargs)

    measurement = multi_measurement([a, b_in, b, z])

    success_probability = 0.0
    for (a_value, b_initial, b_result, carry_result), probability in measurement.items():
        if (
            b_result == (a_value + b_initial) % (2**n)
            and carry_result == (a_value + b_initial) // (2**n)
        ):
            success_probability += probability

    return success_probability


def test_remaud_adder_with_approx_mcx():
    assert remaud_success_probability() > 1 - 1e-12

    approx_probabilities = []
    for seed in range(2):
        approx_probabilities.append(
            remaud_success_probability(
                method="approx",
                mcx_kwargs={"epsilon": 2**-8, "seed": random.Random(seed)},
            )
        )

    assert sum(approx_probabilities) / len(approx_probabilities) > 0.99
