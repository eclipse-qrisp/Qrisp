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

"""
Quantum Inplace Adder Test Suite
================================

This module contains parameterized pytest suites to verify inplace quantum addition 
functions in Qrisp. 

An inplace addition function maps (a, b) to (a, a+b), where 'a' is a QuantumVariable, 
list of Qubits, or an integer, and 'b' is either a QuantumVariable or a list of Qubits.

Requirements Verified
-------------------
Any adder tested in this suite must satisfy the following conditions:
1. Arithmetic Accuracy: Computes the sum modulo 2^i, where i is the size of the target variable.
2. Operand Flexibility: Supports Quantum-Quantum addition (a.size <= b.size) and Classical-Quantum addition.
3. In-Place Preservation: The addend (first operand) must remain unchanged in its computational basis state.
4. Phase Preservation: The operation must be cleanly uncomputed without leaving behind faulty phase shifts.
5. Controllability: Must be fully compatible with Qrisp's `control` environment (acting as the identity when control is |0>).
6. Compilation Compatibility: Must function correctly and satisfy all requirements in both static and dynamic (Jasp) compilation modes.
"""

import pytest
import numpy as np
from qrisp import (
    QuantumBool,
    QuantumFloat,
    control,
    cx,
    h,
    multi_measurement,
    terminal_sampling,
)
from qrisp import cuccaro_adder

# ==========================================
# 1. Test Parameter Generation
# ==========================================
# We generate the exact tuples of (i, j) to avoid nested for-loops in the tests.

# Quantum-Quantum sizes: i from 1 to 6, j from 1 to i
QQ_SIZES = [(i, j) for i in range(1, 7) for j in range(1, i + 1)]

# Classical-Quantum sizes: i from 1 to 5, j from 1 to (2**i - 1)
CQ_SIZES = [(i, j) for i in range(1, 6) for j in range(1, 2**i)]

MODES = ["static", "dynamic"]


# ==========================================
# 2. Dependency Injection (The Fixture)
# ==========================================
@pytest.fixture(
    params=[
        cuccaro_adder,
    ],
    ids=[
        "CuccaroAdder",
    ],
)  # 'ids' makes the test output readable
def inpl_adder(request):
    """
    This fixture injects the adder function into the tests.
    Pytest will run the entire suite for every function listed in 'params'.
    """
    return request.param


# ==========================================
# 3. Validation Helpers
# ==========================================
def assert_valid_phase(q_var, size_calc: float):
    """Verifies the adder does not produce faulty phase shifts in static mode."""
    statevector_arr = q_var.qs.compile().statevector_array()
    threshold = 1 / (2 ** (size_calc / 2 + 1))
    valid_states = statevector_arr[np.abs(statevector_arr) > threshold]
    angles = np.angle(valid_states)
    assert np.sum(np.abs(angles)) < 0.1, "Faulty phase shift detected."


# ==========================================
# 4. The Test Cases
# ==========================================


@pytest.mark.parametrize("i, j", QQ_SIZES)
@pytest.mark.parametrize("mode", MODES)
def test_quantum_quantum_addition(inpl_adder, i, j, mode):
    """Requirement 1, 2, 3, 4, 6: Uncontrolled Q-Q Addition."""

    def circuit(i_val, j_val, adder):
        a = QuantumFloat(j_val)
        b = QuantumFloat(i_val)
        c = QuantumFloat(i_val)
        h(a)
        h(b)
        cx(b, c)
        adder(a, c)
        return a, b, c

    if mode == "static":
        a, b, c = circuit(i, j, inpl_adder)
        assert_valid_phase(a, a.size + b.size)
        mes_res = multi_measurement([a, b, c])
    else:
        mes_res = terminal_sampling(circuit)(i, j, inpl_adder)

    for res_a, res_b, res_c in mes_res.keys():
        assert (res_a + res_b) % (2**i) == res_c, f"{res_a} + {res_b} != {res_c}"


@pytest.mark.parametrize("i, j", CQ_SIZES)
@pytest.mark.parametrize("mode", MODES)
def test_classical_quantum_addition(inpl_adder, i, j, mode):
    """Requirement 1, 2, 3, 4, 6: Uncontrolled C-Q Addition."""

    def circuit(i_val, j_val, adder):
        a = QuantumFloat(i_val)
        b = QuantumFloat(i_val)
        h(a)
        cx(a, b)
        adder(j_val, a)
        return a, b

    if mode == "static":
        a, b = circuit(i, j, inpl_adder)
        assert_valid_phase(a, a.size)
        mes_res = multi_measurement([a, b])
    else:
        mes_res = terminal_sampling(circuit)(i, j, inpl_adder)

    for res_a, res_b in mes_res.keys():
        assert (res_b + j) % (2**i) == res_a, f"{res_b} + {j} != {res_a}"


@pytest.mark.parametrize("i, j", QQ_SIZES)
@pytest.mark.parametrize("mode", MODES)
def test_controlled_quantum_quantum_addition(inpl_adder, i, j, mode):
    """Requirement 5: Controlled Q-Q Addition."""

    def circuit(i_val, j_val, adder):
        a = QuantumFloat(j_val)
        b = QuantumFloat(i_val)
        c = QuantumFloat(i_val)
        qbl = QuantumBool()
        h(qbl)
        h(a)
        h(b)
        cx(b, c)
        with control(qbl):
            adder(a, c)
        return a, b, c, qbl

    if mode == "static":
        a, b, c, qbl = circuit(i, j, inpl_adder)
        assert_valid_phase(a, a.size + b.size)
        mes_res = multi_measurement([a, b, c, qbl])
    else:
        mes_res = terminal_sampling(circuit)(i, j, inpl_adder)

    for res_a, res_b, res_c, res_qbl in mes_res.keys():
        if res_qbl:
            assert (res_a + res_b) % (2**i) == res_c
        else:
            assert res_c == res_b, "Adder modified target while control was |0>"


@pytest.mark.parametrize("i, j", CQ_SIZES)
@pytest.mark.parametrize("mode", MODES)
def test_controlled_classical_quantum_addition(inpl_adder, i, j, mode):
    """Requirement 5: Controlled C-Q Addition."""

    def circuit(i_val, j_val, adder):
        a = QuantumFloat(i_val)
        b = QuantumFloat(i_val)
        qbl = QuantumBool()
        h(qbl)
        h(a)
        cx(a, b)
        with control(qbl):
            adder(j_val, a)
        return a, b, qbl

    if mode == "static":
        a, b, qbl = circuit(i, j, inpl_adder)
        assert_valid_phase(a, a.size)
        mes_res = multi_measurement([a, b, qbl])
    else:
        mes_res = terminal_sampling(circuit)(i, j, inpl_adder)

    for res_a, res_b, res_qbl in mes_res.keys():
        if res_qbl:
            assert (res_b + j) % (2**i) == res_a
        else:
            assert res_b == res_a, "Adder modified target while control was |0>"
