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
    OutcomeArray,
    ry,
    z,
    amplitude_amplification,
)


def test_amplitude_amplification_progression():
    """Tests the mathematical correctness of amplitude amplification over multiple iterations."""

    def state_function(qb):
        ry(np.pi / 8, qb)

    def oracle_function(qb):
        z(qb)

    qb = QuantumBool()
    state_function(qb)
    assert np.isclose(qb.get_measurement()[True], 0.04, atol=1e-2)

    amplitude_amplification([qb], state_function, oracle_function, iter=1)
    assert np.isclose(qb.get_measurement()[True], 0.31, atol=1e-2)

    amplitude_amplification([qb], state_function, oracle_function, iter=1)
    assert np.isclose(qb.get_measurement()[True], 0.69, atol=1e-2)

    amplitude_amplification([qb], state_function, oracle_function, iter=1)
    assert np.isclose(qb.get_measurement()[True], 0.96, atol=1e-2)


def test_amplitude_amplification_quantum_array():
    """Tests that amplitude amplification correctly handles QuantumArray inputs."""

    def state_function(qa):
        ry(np.pi / 8, qa[0])

    def oracle_function(qa):
        z(qa[0])

    qa = QuantumArray(QuantumBool(), shape=(2,))

    state_function(qa)
    amplitude_amplification(qa, state_function, oracle_function, iter=1)

    mes_res = qa.get_measurement()
    target_outcome = OutcomeArray([True, False])
    assert any(k == target_outcome for k in mes_res)
    target_prob = next((p for k, p in mes_res.items() if k == target_outcome), 0.0)
    assert np.isclose(target_prob, 0.31, atol=1e-2)


def test_amplitude_amplification_multiple_variables_list():
    """Tests that amplitude amplification correctly handles a list of separate variables."""

    def state_function(qb0, qb1):
        ry(np.pi / 8, qb0)

    def oracle_function(qb0, qb1):
        z(qb0)

    qb0 = QuantumBool()
    qb1 = QuantumBool()

    state_function(qb0, qb1)
    amplitude_amplification([qb0, qb1], state_function, oracle_function, iter=1)
    assert np.isclose(qb0.get_measurement()[True], 0.31, atol=1e-2)


def test_amplitude_amplification_multiple_variables_tuple():
    """Tests that amplitude amplification correctly handles a tuple of separate variables."""

    def state_function(qb0, qb1):
        ry(np.pi / 8, qb0)

    def oracle_function(qb0, qb1):
        z(qb0)

    qb0 = QuantumBool()
    qb1 = QuantumBool()

    state_function(qb0, qb1)
    amplitude_amplification((qb0, qb1), state_function, oracle_function, iter=1)
    assert np.isclose(qb0.get_measurement()[True], 0.31, atol=1e-2)


def test_amplitude_amplification_oblivious():
    """Tests oblivious amplitude amplification using reflection_indices."""

    def state_function(qa):
        ry(np.pi / 8, qa[0])
        # Act on the second qubit, which we will ignore in reflection
        ry(np.pi / 4, qa[1])

    def oracle_function(qa):
        z(qa[0])

    qa = QuantumArray(QuantumBool(), shape=(2,))

    # Perform reflection ONLY with respect to the first variable (index 0)
    state_function(qa)
    amplitude_amplification(
        qa, state_function, oracle_function, iter=1, reflection_indices=[0]
    )

    # Check that the amplification still succeeded on the targeted qubit
    assert np.isclose(qa[0].get_measurement()[True], 0.31, atol=1e-2)
