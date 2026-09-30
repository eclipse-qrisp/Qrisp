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

import numpy as np
import pytest

from qrisp import (
    QuantumArray,
    QuantumBool,
    QuantumFloat,
    QuantumVariable,
    control,
    cx,
    h,
    invert,
    measure,
    qache,
    x,
)
from qrisp.jasp import jaspr_to_static_register_jaspr, jrange, make_jaspr, q_while_loop

# ---------------------------------------------------------------------------
# These tests compare the execution of an original jaspr against
# its static-register counterpart produced by ``jaspr_to_static_register_jaspr``.
# All test circuits are deterministic (no superposition-based randomness)
# unless explicitly noted, so that the result of both the original and the static-register
# jaspr can be compared against a known ground-truth value.
# ---------------------------------------------------------------------------


def test_static_register_basic_quantum_float():
    """Basic single-register allocation, X gates, measurement."""

    def main():
        a = QuantumFloat(4)
        x(a[0])
        x(a[2])
        return measure(a)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 10)

    assert jaspr() == 5
    assert static_reg_jaspr() == 5


@pytest.mark.parametrize("register_size", [6, 8, 12, 20])
def test_static_register_size_independence(register_size):
    """The register size should not influence the numerical result."""

    def main():
        a = QuantumFloat(4)
        x(a[0])
        x(a[2])
        return measure(a)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, register_size)

    assert jaspr() == 5
    assert static_reg_jaspr() == 5


def test_static_register_tight_qubit_reuse():
    """Register size exactly matches the max number of simultaneously-alive
    qubits, forcing every allocation to reuse freed indices from the
    previous one."""

    def main():
        # Note: qubits must be uncomputed back to |0> before ``.delete()``,
        # otherwise the deletion is invalid regardless of the interpreter.
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

    assert jaspr() == 8
    assert static_reg_jaspr() == 8


@pytest.mark.parametrize("register_size", [10, 12, 20])
def test_static_register_qubit_reuse(register_size):
    """Sequential allocate/use/delete cycles must correctly recycle free
    indices, independent of how much slack the register has."""

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
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, register_size)

    assert jaspr() == 1023
    assert static_reg_jaspr() == 1023


def test_static_register_dynamic_loop_arithmetic():
    """QuantumFloat addition inside a dynamically-bounded ``jrange`` loop."""

    def main():
        a = QuantumFloat(8)

        for _ in jrange(10):
            a += 10

        return measure(a)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 15)

    assert jaspr() == 100
    assert static_reg_jaspr() == 100


def test_static_register_extend_append():
    """``QuantumFloat.extend`` (default = append at the end) exercises the
    ScalarList ``append``/``extend`` path."""

    def main():
        qf = QuantumFloat(3)
        x(qf)
        qf.extend(1)
        return measure(qf)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 10)

    assert jaspr() == 7
    assert static_reg_jaspr() == 7


def test_static_register_extend_prepend():
    """``QuantumFloat.extend(position=0)`` exercises the ScalarList
    ``prepend`` path, which changes logical qubit ordering."""

    def main():
        qf = QuantumFloat(3)
        x(qf)
        qf.extend(1, position=0)
        return measure(qf)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 10)

    assert jaspr() == 14
    assert static_reg_jaspr() == 14


def test_static_register_slicing():
    """Indexing a slice of a QubitArray and applying gates to it."""

    def main():
        qv = QuantumFloat(10)
        sliced = qv[2:7]
        x(sliced[0])  # flips qv[2]
        x(sliced[4])  # flips qv[6]
        return measure(qv)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 12)

    expected = (1 << 2) + (1 << 6)
    assert jaspr() == expected
    assert static_reg_jaspr() == expected


def test_static_register_classical_control():
    """A control environment gated on a runtime (measurement-derived)
    classical value exercises the ``cond`` handling path."""

    def main():
        a = QuantumFloat(3)
        x(a[0])
        val = measure(a[0])
        with control(val):
            x(a[1])
        return measure(a)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 10)

    assert jaspr() == 3
    assert static_reg_jaspr() == 3


def test_static_register_classical_control_not_triggered():
    """Same as above, but the control condition is False, so the guarded
    gate must not be applied."""

    def main():
        a = QuantumFloat(3)
        val = measure(a[0])
        with control(val):
            x(a[1])
        return measure(a)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 10)

    assert jaspr() == 0
    assert static_reg_jaspr() == 0


def test_static_register_while_loop():
    """``q_while_loop`` with a QuantumFloat threaded through the loop carry."""

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

    assert jaspr() == (5, 31)
    assert static_reg_jaspr() == (5, 31)


def test_static_register_qache():
    """Cached (``qache``) sub-routines are re-interpreted correctly across
    multiple call sites with different QuantumVariable subtypes."""

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

    assert jaspr() == (1, 1)
    assert static_reg_jaspr() == (1, 1)


def test_static_register_invert():
    """An ``invert()`` environment cancelling out a previously applied gate."""

    def main():
        a = QuantumFloat(3)
        x(a[0])
        with invert():
            x(a[0])
        return measure(a)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 6)

    assert jaspr() == 0
    assert static_reg_jaspr() == 0


def test_static_register_quantum_bool():
    """Two independently-allocated QuantumBool values."""

    def main():
        a = QuantumBool()
        a[:] = True
        b = QuantumBool()
        b[:] = False
        return measure(a), measure(b)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 4)

    assert jaspr() == (True, False)
    assert static_reg_jaspr() == (True, False)


def test_static_register_quantum_array_assignment():
    """QuantumArray value assignment and measurement (multiple fused
    QubitArrays)."""

    def main():
        qf = QuantumFloat(3)
        qa = QuantumArray(qtype=qf, shape=(3,))
        qa[:] = np.array([1, 2, 3])
        return measure(qa)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 15)

    expected = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(jaspr(), expected)
    assert np.array_equal(static_reg_jaspr(), expected)


def test_static_register_quantum_array_flatten():
    """Flattening a QuantumArray and re-fusing the individual QubitArrays'
    registers into one via ``sum(..., [])``."""

    def main():
        qarg = QuantumArray(QuantumFloat(3), shape=(3,))
        flattened_qarg = qarg.flatten()
        reg = sum([qv.reg for qv in flattened_qarg], [])
        return measure(reg)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 15)

    assert jaspr() == 0
    assert static_reg_jaspr() == 0


def test_static_register_bell_state_correlation():
    """Test that a Bell state produces correlated measurement results, and that the
    static-register interpreter produces the same correlation as the original"""

    def main():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return measure(qv)

    jaspr = make_jaspr(main)()
    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, 10)

    for _ in range(20):
        assert jaspr() in (0, 3)
        assert static_reg_jaspr() in (0, 3)
