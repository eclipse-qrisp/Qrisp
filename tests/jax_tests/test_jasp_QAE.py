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
    QuantumFloat,
    QuantumArray,
    control,
    z,
    h,
    ry,
    QAE,
)
from qrisp.jasp import terminal_sampling, jrange


def test_jasp_QAE_single_variable():
    """Tests QAE with a single QuantumVariable."""

    def state_function(qb):
        ry(np.pi / 4, qb)

    def oracle_function(qb):
        z(qb)

    @terminal_sampling
    def main_jasp():
        qb = QuantumBool()
        res = QAE([qb], state_function, oracle_function, precision=3)
        return res

    mes_res = main_jasp()

    assert np.isclose(mes_res.get(0.125, 0.0), 0.5)
    assert np.isclose(mes_res.get(0.875, 0.0), 0.5)


def test_jasp_QAE_quantum_array():
    """Test that QAE correctly handles QuantumArray inputs."""

    def state_function(qa):
        ry(np.pi / 4, qa[0])

    def oracle_function(qa):
        z(qa[0])

    @terminal_sampling
    def main_jasp():
        qa = QuantumArray(QuantumBool(), shape=(2,))
        res = QAE(qa, state_function, oracle_function, precision=3)
        return res

    mes_res = main_jasp()

    assert np.isclose(mes_res.get(0.125, 0.0), 0.5)
    assert np.isclose(mes_res.get(0.875, 0.0), 0.5)


def test_jasp_QAE_multiple_variables():
    """Tests that QAE correctly handles lists of separate variables."""

    def state_function(qb0, qb1):
        ry(np.pi / 4, qb0)

    def oracle_function(qb0, qb1):
        z(qb0)

    @terminal_sampling
    def main_jasp():
        qb0 = QuantumBool()
        qb1 = QuantumBool()
        res = QAE([qb0, qb1], state_function, oracle_function, precision=3)
        return res

    mes_res = main_jasp()

    assert np.isclose(mes_res.get(0.125, 0.0), 0.5)
    assert np.isclose(mes_res.get(0.875, 0.0), 0.5)


def test_jasp_QAE_integration():
    """Tests QAE on a more complex scenario: computing the integral of f(x) = (sin(x))^2."""

    def state_function(inp, tar):
        h(inp)  # Distribution

        N = 2**inp.size
        for k in jrange(inp.size):
            with control(inp[k]):
                ry(2 ** (k + 1) / N, tar)

    def oracle_function(inp, tar):
        z(tar)

    @terminal_sampling
    def main_jasp():
        n = 6  # 2^n sampling points for integration
        inp = QuantumFloat(n, -n)
        tar = QuantumFloat(1)
        input_list = [inp, tar]

        prec = 6  # precision
        res = QAE(input_list, state_function, oracle_function, precision=prec)
        return res

    meas_res = main_jasp()

    # Get the most probable state and calculate the amplitude
    theta = np.pi * max(meas_res, key=meas_res.get)
    a = np.sin(theta) ** 2

    # Verify the integral matches the analytical expectation within tolerance
    assert np.abs(a - 0.26430) < 1e-4
