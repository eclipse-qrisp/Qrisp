"""********************************************************************************
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

import pytest

import cudaq
import numpy as np

from qrisp import (
    QuantumVariable,
    QuantumBool,
    QuantumFloat,
    h,
    mcx,
    x,
    z,
    p,
    ry,
    measure,
)
from qrisp.alg_primitives import amplitude_amplification, q_switch, QPE, QFT
from qrisp.jasp.cudaq_interface import cudaq_kernel
from qrisp.operators import X, Y, Z

# ---------------------------------------------------------------------------
# Test mcx
# ---------------------------------------------------------------------------

methods = ["balauca", "khattar"]


@pytest.mark.parametrize("method", methods)
def test_mcx(method):
    """Multi-controlled X gate (mcx) with variable number of controls."""

    @cudaq_kernel
    def circuit():
        """Test mcx with 2 controls and 1 target."""
        qv = QuantumVariable(3)
        x(qv[0])
        x(qv[1])
        mcx(qv[:2], qv[2], method=method)
        return measure(qv)

    results = cudaq.run(circuit, shots_count=10)
    assert results == 10 * [7], f"Expected target qubit flipped to 1 when both controls are 1 (7), got {results}"

    @cudaq_kernel
    def circuit():
        """Test mcx with 9 controls and 1 target."""
        qv = QuantumVariable(10)
        x(qv[:9])
        mcx(qv[:9], qv[9], method=method)
        return measure(qv)

    results = cudaq.run(circuit, shots_count=10)
    assert results == 10 * [1023], f"Expected target qubit flipped to 1 when all 9 controls are 1 (1023), got {results}"


# ---------------------------------------------------------------------------
# Test q_switch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["tree", "sequential"])
def test_q_switch(method):
    """Test q_switch with multiple branches and a quantum float index."""

    @cudaq_kernel
    def main():

        def f0(x):
            x += 1

        def f1(x):
            x += 2

        def f2(x):
            pass

        def f3(x):
            h(x[1])

        branches = [f0, f1, f2, f3]

        operand = QuantumFloat(4)
        operand[:] = 1
        index = QuantumFloat(2)
        h(index)

        q_switch(index, branches, operand, method=method)
        return measure(operand)

    results = cudaq.run(main, shots_count=10)


# ---------------------------------------------------------------------------
# Test Primitives
# ---------------------------------------------------------------------------


def test_amplitude_amplification():
    """Test amplitude amplification algorithm with a simple oracle and state function."""

    def state_function(qb):
        ry(np.pi / 8, qb)

    def oracle_function(qb):
        z(qb)

    @cudaq_kernel
    def main():
        qb = QuantumBool()
        state_function(qb)
        amplitude_amplification([qb], state_function, oracle_function, iter=3)
        return measure(qb[0])

    results = cudaq.run(main, shots_count=10)
    assert np.mean(results) >= 0.8, f"Expected amplitude amplification to yield mostly 1s, got {results}"


def test_quantum_fourier_transform():
    """Test that the quantum Fourier transform produces the expected output state."""

    @cudaq_kernel
    def main():
        qv = QuantumVariable(5)
        h(qv)
        QFT(qv)
        return measure(qv)

    results = cudaq.run(main, shots_count=10)
    assert results == 10 * [0]


def test_quantum_phase_estimation():
    """Test that the quantum phase estimation produces the expected output state."""

    def U(qv):
        x = 0.5
        y = 0.125

        p(x * 2 * np.pi, qv[0])
        p(y * 2 * np.pi, qv[1])

    @cudaq_kernel
    def main():
        qv = QuantumVariable(2)
        h(qv)
        res = QPE(qv, U, precision=3)
        return measure(res)

    results = cudaq.run(main, shots_count=10)
    for r in results:
        assert r in {0, 0.125, 0.5, 0.625}


def test_trotterization():
    """Test that a simple Trotterized Hamiltonian evolution produces valid results."""

    @cudaq_kernel
    def main():
        qv = QuantumVariable(2)
        H = X(0) * X(1) + Y(0) * Y(1) + Z(0) * Z(1) + X(0) + X(1)
        U = H.trotterization()
        U(qv, t=1.0, steps=10)
        return measure(qv)

    results = cudaq.run(main, shots_count=10)
