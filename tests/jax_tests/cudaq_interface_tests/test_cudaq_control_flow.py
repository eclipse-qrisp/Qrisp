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

import cudaq
import jax

from qrisp import (
    QuantumVariable,
    QuantumBool,
    h,
    conjugate,
    control,
    invert,
    measure,
    rx,
    x,
    z,
)
from qrisp.jasp import (
    jrange,
    q_while_loop,
    q_cond,
    q_fori_loop,
)
from qrisp.jasp.cudaq_interface import cudaq_kernel


# ---------------------------------------------------------------------------
# Test invert and conjugate
# ---------------------------------------------------------------------------


def test_invert():

    @cudaq_kernel
    def circuit():

        def inner(qv):
            rx(0.5, qv[0])
            z(qv[0])
            h(qv[0])

        qv = QuantumVariable(1)

        inner(qv)

        with invert():
            inner(qv)

        return measure(qv)

    results = cudaq.run(circuit, shots_count=10)
    assert results == 10 * [0]


def test_conjugate():

    @cudaq_kernel
    def circuit():
        qv = QuantumVariable(1)
        with conjugate(h)(qv[0]):
            z(qv[0])
        return measure(qv[0])

    results = cudaq.run(circuit, shots_count=10)
    assert results == 10 * [1]


# ---------------------------------------------------------------------------
# Test control
# ---------------------------------------------------------------------------


def test_classcial_control():
    """Control on a measurement result."""

    @cudaq_kernel
    def circuit():
        qv = QuantumVariable(2)
        h(qv[0])
        c = measure(qv[0])
        with control(c):
            x(qv[1])
        return measure(qv)

    results = cudaq.run(circuit, shots_count=10)
    for r in results:
        assert r in {0, 3}


def test_quantum_control():
    """Control on a qubit value (not measurement result)."""

    @cudaq_kernel
    def circuit():
        qv = QuantumVariable(2)
        x(qv[0])
        with control(qv[0]):
            x(qv[1])
        return measure(qv[1])

    results = cudaq.run(circuit, shots_count=10)
    assert results == 10 * [1]


# ---------------------------------------------------------------------------
# Test q_cond
# ---------------------------------------------------------------------------


def test_q_cond():
    """Conditional on a measurement result."""

    @cudaq_kernel
    def circuit():

        def false_fun(qbl):
            x(qbl[0])
            return qbl

        def true_fun(qbl):
            return qbl

        qbl = QuantumBool()
        h(qbl[0])
        pred = measure(qbl[0])

        qbl = q_cond(pred, true_fun, false_fun, qbl)

        return measure(qbl[0])

    results = cudaq.run(circuit, shots_count=10)
    assert results == 10 * [1]


# ---------------------------------------------------------------------------
# Test q_while_loop
# ---------------------------------------------------------------------------


def test_q_while_loop():
    """While loop with loop-carried quantum variable."""

    @cudaq_kernel
    def circuit():
        qv = QuantumVariable(10)

        def cond_fun(val):
            i, qv = val
            return i < 10

        def body_fun(val):
            i, qv = val
            x(qv[i])
            return i + 1, qv

        q_while_loop(cond_fun, body_fun, (0, qv))
        return measure(qv)

    results = cudaq.run(circuit, shots_count=10)
    assert results == 10 * [1023]


def test_q_while_loop_acc():
    """While loop-carried quantum variable and accumulator."""

    @cudaq_kernel
    def circuit():
        qv = QuantumVariable(10)

        def cond_fun(val):
            i, acc, qv = val
            return i < 5

        def body_fun(val):
            i, acc, qv = val
            x(qv[i])
            acc += measure(qv[i])
            return i + 1, acc, qv

        i, acc, qv = q_while_loop(cond_fun, body_fun, (0, 0, qv))
        return acc

    results = cudaq.run(circuit, shots_count=10)
    assert results == 10 * [5]


# ---------------------------------------------------------------------------
# Test q_fori_loop
# ---------------------------------------------------------------------------


def test_q_fori_loop():
    """Fori loop with loop-carried quantum variable and accumulator."""

    @cudaq_kernel
    def circuit():
        qv = QuantumVariable(10)

        def body_fun(i, val):
            acc, qv = val
            x(qv[i])
            acc += measure(qv[i])
            return acc, qv

        acc, qv = q_fori_loop(0, 5, body_fun, (0, qv))

        return acc

    results = cudaq.run(circuit, shots_count=10)
    assert results == 10 * [5]


# ---------------------------------------------------------------------------
# Test nested control flow
# ---------------------------------------------------------------------------


def test_nested_q_fori_loop_control():
    """Nested fori loop with control inside."""

    @cudaq_kernel
    def circuit():
        qv = QuantumVariable(10)

        def body_fun(i, val):
            acc, qv = val
            x(qv[i])
            c = measure(qv[i])
            with control(c):
                x(qv[i])
            acc += measure(qv[i])
            return acc, qv

        acc, qv = q_fori_loop(0, 5, body_fun, (0, qv))

        return acc

    results = cudaq.run(circuit, shots_count=10)
    assert results == 10 * [0]


# ---------------------------------------------------------------------------
# Test classical control flow
# ---------------------------------------------------------------------------


def test_jax_fori_loop():
    """Test that a JAX fori_loop is correctly lowered to MLIR."""

    @cudaq_kernel
    def circuit():
        result = jax.lax.fori_loop(0, 10, lambda i, x: x + i, 0)
        return result

    results = cudaq.run(circuit, shots_count=10)
    assert results == 10 * [45], f"Expected sum of 0..9 to be 45, got {results}"


# ---------------------------------------------------------------------------
# Test jrange loop
# ---------------------------------------------------------------------------


def test_jrange_loop():
    """Test that a jrange loop is correctly lowered to MLIR."""

    @cudaq_kernel
    def circuit():
        qv = QuantumVariable(3)
        for i in jrange(3):
            x(qv[i])
        return measure(qv)

    results = cudaq.run(circuit, shots_count=10)
    assert results == 10 * [7], f"Expected measurement result to be 7, got {results}"
