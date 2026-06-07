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
import pytest
from qrisp import (
    QuantumBool,
    QuantumFloat,
    QuantumArray,
    control,
    ry,
    z,
    h,
    QAE,
)


def test_QAE_single_variable():
    """Tests QAE with a single QuantumVariable."""

    def state_function(qb):
        ry(np.pi / 4, qb)

    def oracle_function(qb):
        z(qb)

    qb = QuantumBool()
    res = QAE(qb, state_function, oracle_function, precision=3)

    mes_res = res.get_measurement()

    assert np.isclose(mes_res.get(0.125, 0.0), 0.5)
    assert np.isclose(mes_res.get(0.875, 0.0), 0.5)


def test_QAE_single_variable_list():
    """Tests QAE with a single QuantumVariable in a list."""

    def state_function(qb):
        ry(np.pi / 4, qb)

    def oracle_function(qb):
        z(qb)

    qb = QuantumBool()
    res = QAE([qb], state_function, oracle_function, precision=3)

    mes_res = res.get_measurement()

    assert np.isclose(mes_res.get(0.125, 0.0), 0.5)
    assert np.isclose(mes_res.get(0.875, 0.0), 0.5)


def test_QAE_multiple_variables_list():
    """Tests QAE with multiple QuantumVariables in a list."""

    def state_function(qb0, qb1):
        ry(np.pi / 4, qb0)

    def oracle_function(qb0, qb1):
        z(qb0)

    qb0 = QuantumBool()
    qb1 = QuantumBool()
    res = QAE([qb0, qb1], state_function, oracle_function, precision=3)

    mes_res = res.get_measurement()

    assert np.isclose(mes_res.get(0.125, 0.0), 0.5)
    assert np.isclose(mes_res.get(0.875, 0.0), 0.5)


def test_QAE_multiple_variables_tuple():
    """Tests QAE with multiple QuantumVariables in a tuple."""

    def state_function(qb0, qb1):
        ry(np.pi / 4, qb0)

    def oracle_function(qb0, qb1):
        z(qb0)

    qb0 = QuantumBool()
    qb1 = QuantumBool()
    res = QAE((qb0, qb1), state_function, oracle_function, precision=3)

    mes_res = res.get_measurement()

    assert np.isclose(mes_res.get(0.125, 0.0), 0.5)
    assert np.isclose(mes_res.get(0.875, 0.0), 0.5)


def test_QAE_quantum_array():
    """Test that QAE correctly handles QuantumArray inputs."""

    def state_function(qa):
        ry(np.pi / 4, qa[0])

    def oracle_function(qa):
        z(qa[0])

    qa = QuantumArray(QuantumBool(), shape=(2,))
    res = QAE(qa, state_function, oracle_function, precision=3)

    mes_res = res.get_measurement()

    assert np.isclose(mes_res.get(0.125, 0.0), 0.5)
    assert np.isclose(mes_res.get(0.875, 0.0), 0.5)


def test_QAE_multiple_variables():
    """Tests that QAE correctly correctly handles lists of separate variables."""

    def state_function(qb0, qb1):
        ry(np.pi / 4, qb0)

    def oracle_function(qb0, qb1):
        z(qb0)

    qb0 = QuantumBool()
    qb1 = QuantumBool()

    res = QAE([qb0, qb1], state_function, oracle_function, precision=3)

    mes_res = res.get_measurement()

    assert np.isclose(mes_res.get(0.125, 0.0), 0.5)
    assert np.isclose(mes_res.get(0.875, 0.0), 0.5)


def test_QAE_missing_parameters_raises_error():
    """Tests that QAE raises an error if neither precision nor target is provided."""

    def state_function(qb):
        ry(np.pi / 4, qb)

    def oracle_function(qb):
        z(qb)

    qb = QuantumBool()

    with pytest.raises(ValueError, match="either 'precision' or 'target'"):
        QAE([qb], state_function, oracle_function, precision=None, target=None)


def test_QAE_numerical_integration():
    """Tests QAE on a more complex scenario: computing the integral of f(x) = (sin(x))^2."""

    def state_function(inp, tar):
        h(inp)  # Distribution

        N = 2**inp.size
        for k in range(inp.size):
            with control(inp[k]):
                ry(2 ** (k + 1) / N, tar)

    def oracle_function(inp, tar):
        z(tar)

    n = 6  # 2^n sampling points for integration
    inp = QuantumFloat(n, -n)
    tar = QuantumBool()
    input_list = [inp, tar]

    prec = 3  # precision
    res = QAE(input_list, state_function, oracle_function, precision=prec)

    mes_res = res.get_measurement()
    expected_results = {
        0.125: 0.31334,
        0.875: 0.31334,
        0.25: 0.12557,
        0.75: 0.12557,
        0.0: 0.05096,
        0.375: 0.02632,
        0.625: 0.02632,
        0.5: 0.01858,
    }

    # Verify each state matches the expected probability within a safe tolerance
    for state, expected_prob in expected_results.items():
        assert np.isclose(
            mes_res.get(state, 0.0), expected_prob, atol=1e-4
        ), f"Failed on state {state}: expected {expected_prob}, got {mes_res.get(state, 0.0)}"
