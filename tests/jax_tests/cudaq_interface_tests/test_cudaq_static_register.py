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

from qrisp import QuantumBool, QuantumFloat, QuantumVariable, control, cx, h, invert, measure, qache, x
from qrisp.jasp import jaspr_to_static_register_jaspr, jrange, make_jaspr, q_while_loop
from qrisp.jasp.cudaq_interface import cudaq_kernel
from qrisp.jasp.cudaq_interface.cudaq_ingestion.xdsl_ingestion import _cudaq_kernel_from_xdsl_module
from qrisp.jasp.cudaq_interface.quake_lowering.jaspr_to_quake import _jaspr_to_quake_mlir


def test_cudaq_static_register():
    """Test static register lowering of a simple quantum program."""

    def main():
        a = QuantumVariable(10)
        x(a[0])
        x(a[0])
        a.delete()

        b = QuantumVariable(10)
        x(b[0])
        x(b[0])
        b.delete()

        c = QuantumVariable(10)
        x(c)

        return measure(c)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 15)
    xdsl_module = _jaspr_to_quake_mlir(static_reg_jaspr)

    assert jaspr() == 1023
    assert static_reg_jaspr() == 1023

    kernel = _cudaq_kernel_from_xdsl_module(xdsl_module)
    assert kernel() == 1023


def test_cudaq_static_register_quantum_float_addition():
    """Test static register lowering of a simple QuantumFloat addition program with dynamic loop."""

    def main():
        a = QuantumFloat(8)

        for _ in jrange(10):
            a += 10

        return measure(a)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 15)
    xdsl_module = _jaspr_to_quake_mlir(static_reg_jaspr)

    assert jaspr() == 100
    assert static_reg_jaspr() == 100

    kernel = _cudaq_kernel_from_xdsl_module(xdsl_module)
    assert kernel() == 100


def test_cudaq_static_register_basic_quantum_float():
    """Test lowering for a simple QuantumFloat with selected bit flips."""

    def main():
        a = QuantumFloat(4)
        x(a[0])
        x(a[2])
        return measure(a)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 10)
    xdsl_module = _jaspr_to_quake_mlir(static_reg_jaspr)

    assert jaspr() == 5
    assert static_reg_jaspr() == 5

    kernel = _cudaq_kernel_from_xdsl_module(xdsl_module)
    assert kernel() == 5


def test_cudaq_static_register_qubit_reuse():
    """Test lowering when qubits are allocated, used, and recycled."""

    def main():
        a = QuantumVariable(4)
        x(a[0])
        x(a[0])
        a.delete()

        b = QuantumVariable(4)
        x(b[2])
        x(b[2])
        b.delete()

        c = QuantumVariable(4)
        x(c[3])

        return measure(c)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 4)
    xdsl_module = _jaspr_to_quake_mlir(static_reg_jaspr)

    assert jaspr() == 8
    assert static_reg_jaspr() == 8

    kernel = _cudaq_kernel_from_xdsl_module(xdsl_module)
    assert kernel() == 8


def test_cudaq_static_register_classical_control():
    """Test lowering for classical control based on a measured value."""

    def main():
        a = QuantumFloat(3)
        x(a[0])
        val = measure(a[0])
        with control(val):
            x(a[1])
        return measure(a)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 10)
    xdsl_module = _jaspr_to_quake_mlir(static_reg_jaspr)

    assert jaspr() == 3
    assert static_reg_jaspr() == 3

    kernel = _cudaq_kernel_from_xdsl_module(xdsl_module)
    assert kernel() == 3


@pytest.mark.parametrize("register_size", [6, 8, 12, 20])
def test_cudaq_static_register_size_independence(register_size):
    """Test that varying the static register size does not change the result."""

    def main():
        a = QuantumFloat(4)
        x(a[0])
        x(a[2])
        return measure(a)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, register_size)
    xdsl_module = _jaspr_to_quake_mlir(static_reg_jaspr)

    assert jaspr() == 5
    assert static_reg_jaspr() == 5

    kernel = _cudaq_kernel_from_xdsl_module(xdsl_module)
    assert kernel() == 5


def test_cudaq_static_register_classical_control_not_triggered():
    """Test that an untriggered control branch leaves the state unchanged."""

    def main():
        a = QuantumFloat(3)
        val = measure(a[0])
        with control(val):
            x(a[1])
        return measure(a)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 10)
    xdsl_module = _jaspr_to_quake_mlir(static_reg_jaspr)

    assert jaspr() == 0
    assert static_reg_jaspr() == 0

    kernel = _cudaq_kernel_from_xdsl_module(xdsl_module)
    assert kernel() == 0


def test_cudaq_static_register_invert():
    """Test that an invert environment cancels an applied gate."""

    def main():
        a = QuantumFloat(3)
        x(a[0])
        with invert():
            x(a[0])
        return measure(a)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 6)
    xdsl_module = _jaspr_to_quake_mlir(static_reg_jaspr)

    assert jaspr() == 0
    assert static_reg_jaspr() == 0

    kernel = _cudaq_kernel_from_xdsl_module(xdsl_module)
    assert kernel() == 0


def test_cudaq_static_register_tight_qubit_reuse():
    """Test lowering when qubits are allocated, reused, and then recycled tightly."""

    def main():
        a = QuantumVariable(4)
        x(a[0])
        x(a[0])
        a.delete()

        b = QuantumVariable(4)
        x(b[2])
        x(b[2])
        b.delete()

        c = QuantumVariable(4)
        x(c[3])

        return measure(c)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 4)
    xdsl_module = _jaspr_to_quake_mlir(static_reg_jaspr)

    assert jaspr() == 8
    assert static_reg_jaspr() == 8

    kernel = _cudaq_kernel_from_xdsl_module(xdsl_module)
    assert kernel() == 8


def test_cudaq_static_register_extend_append():
    """Test lowering for QuantumFloat.extend() using append semantics."""

    def main():
        qf = QuantumFloat(3)
        x(qf)
        qf.extend(1)
        return measure(qf)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 10)
    xdsl_module = _jaspr_to_quake_mlir(static_reg_jaspr)

    assert jaspr() == 7
    assert static_reg_jaspr() == 7

    kernel = _cudaq_kernel_from_xdsl_module(xdsl_module)
    assert kernel() == 7


def test_cudaq_static_register_extend_prepend():
    """Test lowering for QuantumFloat.extend() using prepend semantics."""

    def main():
        qf = QuantumFloat(3)
        x(qf)
        qf.extend(1, position=0)
        return measure(qf)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 10)
    xdsl_module = _jaspr_to_quake_mlir(static_reg_jaspr)

    assert jaspr() == 14
    assert static_reg_jaspr() == 14

    kernel = _cudaq_kernel_from_xdsl_module(xdsl_module)
    assert kernel() == 14


def test_cudaq_static_register_slicing():
    """Test lowering for slicing and applying gates to a sub-register."""

    def main():
        qv = QuantumFloat(10)
        sliced = qv[2:7]
        x(sliced[0])
        x(sliced[4])
        return measure(qv)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 12)
    xdsl_module = _jaspr_to_quake_mlir(static_reg_jaspr)

    expected = (1 << 2) + (1 << 6)
    assert jaspr() == expected
    assert static_reg_jaspr() == expected

    kernel = _cudaq_kernel_from_xdsl_module(xdsl_module)
    assert kernel() == expected


def test_cudaq_static_register_while_loop():
    """Test lowering for q_while_loop carrying a QuantumFloat through the loop."""

    def main():
        qf = QuantumFloat(6)

        def body_fun(val):
            i, acc, qf = val
            x(qf[i])
            acc += measure(qf[i])
            return i + 1, acc, qf

        def cond_fun(val):
            return val[0] < 5

        i, acc, qf = q_while_loop(cond_fun, body_fun, (0, 0, qf))
        return acc, measure(qf)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 10)
    xdsl_module = _jaspr_to_quake_mlir(static_reg_jaspr)

    assert jaspr() == (5, 31)
    assert static_reg_jaspr() == (5, 31)

    kernel = _cudaq_kernel_from_xdsl_module(xdsl_module)
    assert kernel() == (5, 31)


def test_cudaq_static_register_qache():
    """Test lowering when qache subroutines are reused across different QuantumVariable subtypes."""

    @qache
    def flip_and_measure(qv):
        x(qv[0])
        return measure(qv[0])

    def main():
        a = QuantumVariable(2)
        b = QuantumFloat(2)
        return flip_and_measure(a), flip_and_measure(b)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 10)
    xdsl_module = _jaspr_to_quake_mlir(static_reg_jaspr)

    assert jaspr() == (1, 1)
    assert static_reg_jaspr() == (1, 1)

    kernel = _cudaq_kernel_from_xdsl_module(xdsl_module)
    assert kernel() == (1, 1)


def test_cudaq_static_register_quantum_bool():
    """Test lowering for independently allocated QuantumBool values."""

    def main():
        a = QuantumBool()
        a[:] = True
        b = QuantumBool()
        b[:] = False
        return measure(a), measure(b)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 4)
    xdsl_module = _jaspr_to_quake_mlir(static_reg_jaspr)

    assert jaspr() == (True, False)
    assert static_reg_jaspr() == (True, False)

    kernel = _cudaq_kernel_from_xdsl_module(xdsl_module)
    assert kernel() == (True, False)


def test_cudaq_static_register_bell_state_correlation():
    """Test lowering for a Bell-state circuit with correlated outcomes."""

    def main():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return measure(qv)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 10)
    xdsl_module = _jaspr_to_quake_mlir(static_reg_jaspr)

    for _ in range(20):
        assert jaspr() in (0, 3)
        assert static_reg_jaspr() in (0, 3)

    kernel = _cudaq_kernel_from_xdsl_module(xdsl_module)
    for _ in range(20):
        assert kernel() in (0, 3)
