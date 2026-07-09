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

from qrisp import (
    QuantumBool,
    QuantumFloat,
    QuantumVariable,
    control,
    measure,
    thapliyal_adder,
    x,
)
from qrisp.circuit import Qubit
from qrisp.jasp import jaspify

# ---------------------------------------------------------------------------
# Static smoke tests — just a few representative cases with small registers
# to catch gross regressions without the full statevector-simulation cost.
# Exhaustive coverage lives in the jaspify tests below.
# ---------------------------------------------------------------------------


def test_thapliyal_adder_static_smoke_quantum_a():
    """Quantum a + quantum b, equal size, no optional args."""
    a = QuantumFloat(3)
    b = QuantumFloat(3)
    a[:] = 5
    b[:] = 3
    thapliyal_adder(a, b)
    assert b.get_measurement() == {0: 1.0}  # (5+3) % 8


def test_thapliyal_adder_static_smoke_classical_a():
    """Classical a + quantum b."""
    b = QuantumFloat(3)
    b[:] = 3
    thapliyal_adder(5, b)
    assert b.get_measurement() == {0: 1.0}


def test_thapliyal_adder_static_smoke_cin():
    """c_in with classical a."""
    b = QuantumFloat(3)
    b[:] = 2
    c_in = QuantumBool()
    x(c_in[0])
    thapliyal_adder(3, b, c_in=c_in)
    assert b.get_measurement() == {6: 1.0}  # 2 + 3 + 1


def test_thapliyal_adder_static_smoke_cin_qubit():
    """c_in of type Qubit."""
    b = QuantumFloat(3)
    b[:] = 2
    qv = QuantumVariable(1)
    c_in = qv[0]
    assert isinstance(c_in, Qubit)
    x(c_in)
    thapliyal_adder(3, b, c_in=c_in)
    assert b.get_measurement() == {6: 1.0}  # 2 + 3 + 1


def test_thapliyal_adder_static_smoke_c_in_type_error():
    """TypeError when c_in is neither QuantumBool nor Qubit."""
    b = QuantumFloat(4)
    b[:] = 3
    with pytest.raises(TypeError, match="c_in must be of type QuantumBool or Qubit"):
        thapliyal_adder(1, b, c_in=QuantumFloat(2))
    with pytest.raises(TypeError, match="c_in must be of type QuantumBool or Qubit"):
        thapliyal_adder(1, b, c_in="invalid")
    with pytest.raises(TypeError, match="c_in must be of type QuantumBool or Qubit"):
        thapliyal_adder(1, b, c_in=42)


def test_thapliyal_adder_static_smoke_cout_overflow():
    """c_out captures overflow."""
    b = QuantumFloat(3)
    b[:] = 6
    c_out = QuantumBool()
    thapliyal_adder(3, b, c_out=c_out)
    assert b.get_measurement() == {1: 1.0}  # (6+3) % 8
    assert c_out.get_measurement() == {True: 1.0}


def test_thapliyal_adder_static_smoke_ctrl():
    """Controlled addition (ctrl kwarg)."""
    a = QuantumFloat(3)
    b = QuantumFloat(3)
    a[:] = 3
    b[:] = 5
    ctrl = QuantumBool()
    x(ctrl[0])
    thapliyal_adder(a, b, ctrl=ctrl)
    assert b.get_measurement() == {0: 1.0}  # (5+3)%8


def test_thapliyal_adder_static_smoke_cin_and_cout():
    """c_in + c_out together."""
    b = QuantumFloat(3)
    b[:] = 6
    c_in = QuantumBool()
    x(c_in[0])
    c_out = QuantumBool()
    thapliyal_adder(3, b, c_in=c_in, c_out=c_out)
    assert b.get_measurement() == {2: 1.0}  # (6 + 3 + 1) % 8
    assert c_out.get_measurement() == {True: 1.0}


def test_thapliyal_adder_static_smoke_cin_qubit_and_cout():
    """c_in of type Qubit with c_out together."""
    b = QuantumFloat(3)
    b[:] = 6
    qv = QuantumVariable(1)
    c_in = qv[0]
    assert isinstance(c_in, Qubit)
    x(c_in)
    c_out = QuantumBool()
    thapliyal_adder(3, b, c_in=c_in, c_out=c_out)
    assert b.get_measurement() == {2: 1.0}  # (6 + 3 + 1) % 8
    assert c_out.get_measurement() == {True: 1.0}


def test_thapliyal_adder_static_smoke_cout_and_ctrl():
    """c_out + ctrl (ctrl=on)."""
    a = QuantumFloat(3)
    b = QuantumFloat(3)
    a[:] = 6
    b[:] = 6
    c_out = QuantumBool()
    ctrl = QuantumBool()
    x(ctrl[0])
    thapliyal_adder(a, b, c_out=c_out, ctrl=ctrl)
    assert b.get_measurement() == {4: 1.0}  # (6+6)%8
    assert c_out.get_measurement() == {True: 1.0}


def test_thapliyal_adder_static_smoke_unequal_sizes():
    """a smaller than b (extension) and a larger than b (truncation/modulo)."""
    a = QuantumFloat(2)
    b = QuantumFloat(4)
    a[:] = 3
    b[:] = 10
    thapliyal_adder(a, b)
    assert b.get_measurement() == {13: 1.0}  # (3+10) % 16

    a2 = QuantumFloat(4)
    b2 = QuantumFloat(2)
    a2[:] = 10
    b2[:] = 3
    thapliyal_adder(a2, b2)
    assert b2.get_measurement() == {1: 1.0}  # (10+3) % 4


def test_thapliyal_adder_static_smoke_inputs_unmodified():
    """Input QuantumFloat sizes are unchanged after addition."""
    a = QuantumFloat(5)
    b = QuantumFloat(7)
    orig_a, orig_b = a.size, b.size
    a[:] = 3
    b[:] = 4
    thapliyal_adder(a, b)
    assert a.size == orig_a
    assert b.size == orig_b


# ---------------------------------------------------------------------------
# Exhaustive tests via @jaspify.
#
# Unlike cuccaro_adder, thapliyal_adder's TR-gate step uses rx/p rotations
# (see THAPLIYAL_ADDER_REFACTOR_NOTES.md) that the fast classical
# @boolean_simulation evaluator can't process ("Classical function simulator
# can't process gate crx"), even though the whole circuit reduces to a boolean
# permutation on computational basis states. So these tests fall back to
# @jaspify (full statevector simulation of the traced program), which is
# considerably slower — register sizes here are kept small to keep runtime
# reasonable.
#
# Each helper factory captures configuration as CLOSURE variables (not function
# parameters) so JAX treats them as compile-time constants during @jit tracing.
# Loops live in plain Python outer functions to keep the JAX cache warm.
# ---------------------------------------------------------------------------


def _mk_add_basic():
    """No optional args."""

    @jaspify
    def add(N, L, j, k):
        A = QuantumFloat(N)
        B = QuantumFloat(L)
        A[:] = j
        B[:] = k
        thapliyal_adder(A, B)
        return measure(A), measure(B)

    return add


def _mk_add_cin(c_in_val):
    """c_in_val: 0 or 1."""

    @jaspify
    def add(N, L, j, k):
        A = QuantumFloat(N)
        B = QuantumFloat(L)
        A[:] = j
        B[:] = k
        c_in = QuantumBool()
        if c_in_val:
            c_in.flip()
        thapliyal_adder(A, B, c_in=c_in)
        return measure(B)

    return add


def _mk_add_cin_qubit(c_in_val):
    """c_in_val: 0 or 1; c_in is a bare Qubit."""

    @jaspify
    def add(N, L, j, k):
        A = QuantumFloat(N)
        B = QuantumFloat(L)
        A[:] = j
        B[:] = k
        qv = QuantumVariable(1)
        c_in = qv[0]
        if c_in_val:
            x(c_in)
        thapliyal_adder(A, B, c_in=c_in)
        return measure(B)

    return add


def _mk_add_cout(c_in_val):
    """c_in_val: 0 or 1; c_out always present.  Classical a."""

    @jaspify
    def add(L, j, k):
        B = QuantumFloat(L)
        B[:] = k
        c_in = QuantumBool()
        if c_in_val:
            c_in.flip()
        c_out = QuantumBool()
        thapliyal_adder(j, B, c_in=c_in, c_out=c_out)
        return measure(B), measure(c_out)

    return add


def _mk_add_cout_qubit(c_in_val):
    """c_in_val: 0 or 1; c_out always present; c_in is a bare Qubit.  Classical a."""

    @jaspify
    def add(L, j, k):
        B = QuantumFloat(L)
        B[:] = k
        qv = QuantumVariable(1)
        c_in = qv[0]
        if c_in_val:
            x(c_in)
        c_out = QuantumBool()
        thapliyal_adder(j, B, c_in=c_in, c_out=c_out)
        return measure(B), measure(c_out)

    return add


def _mk_add_cout_qq(c_in_val):
    """c_in_val: 0 or 1; c_out always present.  Quantum a, equal sizes."""

    @jaspify
    def add(L, j, k):
        A = QuantumFloat(L)
        B = QuantumFloat(L)
        A[:] = j
        B[:] = k
        c_in = QuantumBool()
        if c_in_val:
            c_in.flip()
        c_out = QuantumBool()
        thapliyal_adder(A, B, c_in=c_in, c_out=c_out)
        return measure(A), measure(B), measure(c_out)

    return add


def _mk_add_ctrl(c_in_val, use_kwarg):
    """c_in_val: 0 or 1.  use_kwarg: bool — ctrl= vs with control()."""

    @jaspify
    def add(N, L, j, k):
        A = QuantumFloat(N)
        B = QuantumFloat(L)
        A[:] = j
        B[:] = k
        qbl = QuantumBool()
        qbl.flip()  # ctrl is always |1>
        c_in = QuantumBool()
        if c_in_val:
            c_in.flip()
        if use_kwarg:
            thapliyal_adder(A, B, c_in=c_in, ctrl=qbl)
        else:
            with control(qbl):
                thapliyal_adder(A, B, c_in=c_in)
        return measure(A), measure(B)

    return add


def _mk_add_ctrl_qubit(c_in_val, use_kwarg):
    """c_in_val: 0 or 1; c_in is a bare Qubit.  use_kwarg: bool — ctrl= vs with control()."""

    @jaspify
    def add(N, L, j, k):
        A = QuantumFloat(N)
        B = QuantumFloat(L)
        A[:] = j
        B[:] = k
        qbl = QuantumBool()
        qbl.flip()  # ctrl is always |1>
        qv = QuantumVariable(1)
        c_in = qv[0]
        if c_in_val:
            x(c_in)
        if use_kwarg:
            thapliyal_adder(A, B, c_in=c_in, ctrl=qbl)
        else:
            with control(qbl):
                thapliyal_adder(A, B, c_in=c_in)
        return measure(A), measure(B)

    return add


def _mk_add_cout_ctrl(c_in_val):
    """c_in_val: 0 or 1.  c_out and ctrl together, ctrl is always |1>."""

    @jaspify
    def add(L, j, k):
        A = QuantumFloat(L)
        B = QuantumFloat(L)
        A[:] = j
        B[:] = k
        c_in = QuantumBool()
        if c_in_val:
            c_in.flip()
        c_out = QuantumBool()
        ctrl = QuantumBool()
        ctrl.flip()
        thapliyal_adder(A, B, c_in=c_in, c_out=c_out, ctrl=ctrl)
        return measure(A), measure(B), measure(c_out)

    return add


# -- exhaustive test runners -------------------------------------------------
#
# Register sizes are kept small (2-3 bits) since each call runs a full
# statevector simulation of the traced program (see module docstring above).


def _run_basic_exhaustive():
    add = _mk_add_basic()
    for N in range(2, 4):
        for L in range(2, 4):
            for j in range(1 << N):
                for k in range(1 << L):
                    A, B = add(N, L, j, k)
                    assert A == j
                    assert B == (k + j) % (1 << L)


def test_thapliyal_adder_basic_dynamic():
    _run_basic_exhaustive()


def _run_cin_exhaustive():
    for c_in_val in (0, 1):
        add = _mk_add_cin(c_in_val)
        for N in range(2, 4):
            for L in range(2, 4):
                for j in range(1 << N):
                    for k in range(1 << L):
                        B = add(N, L, j, k)
                        assert B == (k + j + c_in_val) % (1 << L)


def test_thapliyal_adder_cin_dynamic():
    _run_cin_exhaustive()


def _run_cin_qubit_exhaustive():
    for c_in_val in (0, 1):
        add = _mk_add_cin_qubit(c_in_val)
        for N in range(2, 4):
            for L in range(2, 4):
                for j in range(1 << N):
                    for k in range(1 << L):
                        B = add(N, L, j, k)
                        assert B == (k + j + c_in_val) % (1 << L)


def test_thapliyal_adder_cin_qubit_dynamic():
    _run_cin_qubit_exhaustive()


def _run_cout_exhaustive():
    for c_in_val in (0, 1):
        add = _mk_add_cout(c_in_val)
        for L in range(2, 4):
            for j in range(1 << L):
                for k in range(1 << L):
                    total = k + j + c_in_val
                    B, cout = add(L, j, k)
                    assert B == total % (1 << L)
                    assert cout == (total >= (1 << L))


def test_thapliyal_adder_cout_dynamic():
    _run_cout_exhaustive()


def _run_cout_qubit_exhaustive():
    for c_in_val in (0, 1):
        add = _mk_add_cout_qubit(c_in_val)
        for L in range(2, 4):
            for j in range(1 << L):
                for k in range(1 << L):
                    total = k + j + c_in_val
                    B, cout = add(L, j, k)
                    assert B == total % (1 << L)
                    assert cout == (total >= (1 << L))


def test_thapliyal_adder_cout_qubit_dynamic():
    _run_cout_qubit_exhaustive()


def _run_cout_equal_sizes_exhaustive():
    for c_in_val in (0, 1):
        add = _mk_add_cout_qq(c_in_val)
        for L in range(2, 4):
            for j in range(1 << L):
                for k in range(1 << L):
                    total = k + j + c_in_val
                    A_res, B_res, cout = add(L, j, k)
                    assert A_res == j
                    assert B_res == total % (1 << L)
                    assert cout == (total >= (1 << L))


def test_thapliyal_adder_cout_equal_sizes_dynamic():
    _run_cout_equal_sizes_exhaustive()


def _run_ctrl_exhaustive():
    # ctrl retraces the controlled variant per distinct (N, L) shape (see
    # custom_control_environment.py), which is the most expensive part of this
    # test file under @jaspify. A single shape still exercises the ctrl code
    # path fully; broader (N, L) coverage is already exhausted by the
    # non-ctrl basic/cin/cout tests above.
    for c_in_val in (0, 1):
        for use_kwarg in (False, True):
            add = _mk_add_ctrl(c_in_val, use_kwarg)
            for N in (2,):
                for L in (2,):
                    for j in range(1 << N):
                        for k in range(1 << L):
                            A, B = add(N, L, j, k)
                            assert A == j
                            assert B == (k + j + c_in_val) % (1 << L)


def test_thapliyal_adder_ctrl_dynamic():
    _run_ctrl_exhaustive()


def _run_ctrl_qubit_exhaustive():
    for c_in_val in (0, 1):
        for use_kwarg in (False, True):
            add = _mk_add_ctrl_qubit(c_in_val, use_kwarg)
            for N in (2,):
                for L in (2,):
                    for j in range(1 << N):
                        for k in range(1 << L):
                            A, B = add(N, L, j, k)
                            assert A == j
                            assert B == (k + j + c_in_val) % (1 << L)


def test_thapliyal_adder_ctrl_qubit_dynamic():
    _run_ctrl_qubit_exhaustive()


def _run_cout_ctrl_exhaustive():
    for c_in_val in (0, 1):
        add = _mk_add_cout_ctrl(c_in_val)
        for L in (2,):
            for j in range(1 << L):
                for k in range(1 << L):
                    total = k + j + c_in_val
                    A_res, B_res, cout = add(L, j, k)
                    assert A_res == j
                    assert B_res == total % (1 << L)
                    assert cout == (total >= (1 << L))


def test_thapliyal_adder_cout_ctrl_dynamic():
    _run_cout_ctrl_exhaustive()
