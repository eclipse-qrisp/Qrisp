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

import numpy as np
from qrisp import (
    QuantumBool,
    QuantumArray,
    ry,
    z,
    amplitude_amplification,
)
from qrisp.jasp import terminal_sampling


def test_jasp_amplitude_amplification_progression():
    """Tests the mathematical correctness of amplitude amplification over multiple iterations."""

    def state_function(qb):
        ry(np.pi / 8, qb)

    def oracle_function(qb):
        z(qb)

    @terminal_sampling
    def main_jasp(i):
        qb = QuantumBool()
        state_function(qb)
        amplitude_amplification([qb], state_function, oracle_function, iter=i)
        return qb

    assert np.isclose(main_jasp(0)[True], 0.04, atol=1e-2)
    assert np.isclose(main_jasp(1)[True], 0.31, atol=1e-2)
    assert np.isclose(main_jasp(2)[True], 0.69, atol=1e-2)
    assert np.isclose(main_jasp(3)[True], 0.96, atol=1e-2)


def test_jasp_amplitude_amplification_quantum_array():
    """Tests that amplitude amplification correctly handles QuantumArray inputs."""

    def state_function(qa):
        ry(np.pi / 8, qa[0])

    def oracle_function(qa):
        z(qa[0])

    @terminal_sampling
    def main_jasp():
        qa = QuantumArray(QuantumBool(), shape=(2,))
        state_function(qa)
        amplitude_amplification(qa, state_function, oracle_function, iter=1)
        return qa[0], qa[1]

    mes_res = main_jasp()
    assert np.isclose(mes_res[(True, False)], 0.31, atol=1e-2)


def test_jasp_amplitude_amplification_multiple_variables():
    """Tests that amplitude amplification correctly handles a list of separate variables."""

    def state_function(qb0, qb1):
        ry(np.pi / 8, qb0)

    def oracle_function(qb0, qb1):
        z(qb0)

    @terminal_sampling
    def main_jasp():
        qb0 = QuantumBool()
        qb1 = QuantumBool()
        state_function(qb0, qb1)
        amplitude_amplification([qb0, qb1], state_function, oracle_function, iter=1)
        return qb0

    mes_res = main_jasp()
    assert np.isclose(mes_res[True], 0.31, atol=1e-2)


def test_jasp_amplitude_amplification_oblivious():
    """Tests oblivious amplitude amplification using reflection_indices."""

    def state_function(qa):
        ry(np.pi / 8, qa[0])
        # Act on the second qubit, which we will ignore in reflection
        ry(np.pi / 4, qa[1])

    def oracle_function(qa):
        z(qa[0])

    @terminal_sampling
    def main_jasp():
        qa = QuantumArray(QuantumBool(), shape=(2,))
        state_function(qa)
        amplitude_amplification(
            qa, state_function, oracle_function, iter=1, reflection_indices=[0]
        )
        return qa[0]

    mes_res = main_jasp()
    assert np.isclose(mes_res[True], 0.31, atol=1e-2)
