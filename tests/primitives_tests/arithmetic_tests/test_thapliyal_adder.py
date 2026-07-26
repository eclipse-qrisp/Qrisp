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

from dataclasses import dataclass
from typing import Optional

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


@dataclass
class _SmokeCase:
    name: str
    a_size: int
    a_val: int
    a_quantum: bool
    b_size: int
    b_val: int
    expected_b: int
    c_in_kind: Optional[str] = None  # None, "bool", "qubit"
    c_in_val: int = 0
    use_c_out: bool = False
    expected_c_out: Optional[bool] = None
    use_ctrl: bool = False


SMOKE_CASES = [
    _SmokeCase("quantum_a", a_size=3, a_val=5, a_quantum=True, b_size=3, b_val=3, expected_b=0),  # (5+3)%8
    _SmokeCase("classical_a", a_size=3, a_val=5, a_quantum=False, b_size=3, b_val=3, expected_b=0),  # (5+3)%8
    _SmokeCase(
        "cin", a_size=3, a_val=3, a_quantum=False, b_size=3, b_val=2, c_in_kind="bool", c_in_val=1, expected_b=6
    ),  # 2+3+1
    _SmokeCase(
        "cin_qubit", a_size=3, a_val=3, a_quantum=False, b_size=3, b_val=2, c_in_kind="qubit", c_in_val=1, expected_b=6
    ),  # 2+3+1
    _SmokeCase(
        "cout_overflow",
        a_size=3,
        a_val=3,
        a_quantum=False,
        b_size=3,
        b_val=6,
        use_c_out=True,
        expected_b=1,
        expected_c_out=True,
    ),  # (6+3)%8, overflow
    _SmokeCase("ctrl", a_size=3, a_val=3, a_quantum=True, b_size=3, b_val=5, use_ctrl=True, expected_b=0),  # (5+3)%8
    _SmokeCase(
        "cin_and_cout",
        a_size=3,
        a_val=3,
        a_quantum=False,
        b_size=3,
        b_val=6,
        c_in_kind="bool",
        c_in_val=1,
        use_c_out=True,
        expected_b=2,
        expected_c_out=True,
    ),  # (6+3+1)%8, overflow
    _SmokeCase(
        "cin_qubit_and_cout",
        a_size=3,
        a_val=3,
        a_quantum=False,
        b_size=3,
        b_val=6,
        c_in_kind="qubit",
        c_in_val=1,
        use_c_out=True,
        expected_b=2,
        expected_c_out=True,
    ),  # (6+3+1)%8, overflow
    _SmokeCase(
        "cout_and_ctrl",
        a_size=3,
        a_val=6,
        a_quantum=True,
        b_size=3,
        b_val=6,
        use_c_out=True,
        use_ctrl=True,
        expected_b=4,
        expected_c_out=True,
    ),  # (6+6)%8, overflow
    _SmokeCase(
        "unequal_sizes_a_smaller", a_size=2, a_val=3, a_quantum=True, b_size=4, b_val=10, expected_b=13
    ),  # (3+10)%16, a extended
    _SmokeCase(
        "unequal_sizes_a_larger", a_size=4, a_val=10, a_quantum=True, b_size=2, b_val=3, expected_b=1
    ),  # (10+3)%4, a truncated
]


@pytest.mark.parametrize("case", SMOKE_CASES, ids=[c.name for c in SMOKE_CASES])
def test_thapliyal_adder_static_smoke(case):
    b = QuantumFloat(case.b_size)
    b[:] = case.b_val

    if case.a_quantum:
        a = QuantumFloat(case.a_size)
        a[:] = case.a_val
    else:
        a = case.a_val

    kwargs = {}

    if case.c_in_kind == "bool":
        c_in = QuantumBool()
        if case.c_in_val:
            x(c_in[0])
        kwargs["c_in"] = c_in
    elif case.c_in_kind == "qubit":
        qv = QuantumVariable(1)
        c_in = qv[0]
        assert isinstance(c_in, Qubit)
        if case.c_in_val:
            x(c_in)
        kwargs["c_in"] = c_in

    if case.use_c_out:
        c_out = QuantumBool()
        kwargs["c_out"] = c_out

    if case.use_ctrl:
        ctrl = QuantumBool()
        x(ctrl[0])
        kwargs["ctrl"] = ctrl

    thapliyal_adder(a, b, **kwargs)

    assert b.get_measurement() == {case.expected_b: 1.0}
    if case.expected_c_out is not None:
        assert c_out.get_measurement() == {case.expected_c_out: 1.0}


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


def test_thapliyal_adder_static_smoke_b_type_error():
    """ValueError when b is not a QuantumVariable (a is quantum)."""
    a = QuantumFloat(4)
    a[:] = 3
    with pytest.raises(ValueError, match="The second argument must be of type QuantumVariable"):
        thapliyal_adder(a, 5)
    with pytest.raises(ValueError, match="The second argument must be of type QuantumVariable"):
        thapliyal_adder(a, "invalid")


def test_thapliyal_adder_static_smoke_c_out_type_error():
    """TypeError when c_out is neither QuantumBool nor Qubit."""
    a = QuantumFloat(4)
    b = QuantumFloat(4)
    a[:] = 3
    b[:] = 4
    with pytest.raises(TypeError, match="c_out must be of type QuantumBool or Qubit"):
        thapliyal_adder(a, b, c_out=QuantumFloat(2))
    with pytest.raises(TypeError, match="c_out must be of type QuantumBool or Qubit"):
        thapliyal_adder(a, b, c_out="invalid")
    with pytest.raises(TypeError, match="c_out must be of type QuantumBool or Qubit"):
        thapliyal_adder(a, b, c_out=42)


def test_thapliyal_adder_static_smoke_c_out_qubit():
    """c_out as a bare Qubit (not QuantumBool) exercises the plain-Qubit assignment branch."""
    a = QuantumFloat(3)
    b = QuantumFloat(3)
    a[:] = 3
    b[:] = 6
    qv = QuantumVariable(1)
    c_out = qv[0]
    assert isinstance(c_out, Qubit)
    thapliyal_adder(a, b, c_out=c_out)
    assert b.get_measurement() == {1: 1.0}  # (6+3)%8, overflow
    assert qv.get_measurement() == {"1": 1.0}


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
# Unlike cuccaro_adder, thapliyal_adder's TR-gate step uses rx/p rotations that
# the fast classical @boolean_simulation evaluator can't process ("Classical function simulator
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


def _mk_add_ctrl(c_in_val, use_kwarg, ctrl_val):
    """c_in_val: 0 or 1.  use_kwarg: bool — ctrl= vs with control().  ctrl_val: 0 or
    1 — when 0 the addition must be a no-op."""

    @jaspify
    def add(N, L, j, k):
        A = QuantumFloat(N)
        B = QuantumFloat(L)
        A[:] = j
        B[:] = k
        qbl = QuantumBool()
        if ctrl_val:
            qbl.flip()
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


def _mk_add_ctrl_qubit(c_in_val, use_kwarg, ctrl_val):
    """c_in_val: 0 or 1; c_in is a bare Qubit.  use_kwarg: bool — ctrl= vs with
    control().  ctrl_val: 0 or 1 — when 0 the addition must be a no-op."""

    @jaspify
    def add(N, L, j, k):
        A = QuantumFloat(N)
        B = QuantumFloat(L)
        A[:] = j
        B[:] = k
        qbl = QuantumBool()
        if ctrl_val:
            qbl.flip()
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


def _mk_add_cout_ctrl(c_in_val, ctrl_val):
    """c_in_val: 0 or 1.  c_out and ctrl together.  ctrl_val: 0 or 1 — when 0 the
    addition must be a no-op and c_out must stay |0>."""

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
        if ctrl_val:
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
            for ctrl_val in (0, 1):
                add = _mk_add_ctrl(c_in_val, use_kwarg, ctrl_val)
                for N in (2,):
                    for L in (2,):
                        for j in range(1 << N):
                            for k in range(1 << L):
                                A, B = add(N, L, j, k)
                                assert A == j
                                # ctrl off -> addition is a no-op (B unchanged)
                                if ctrl_val:
                                    assert B == (k + j + c_in_val) % (1 << L)
                                else:
                                    assert B == k


def test_thapliyal_adder_ctrl_dynamic():
    _run_ctrl_exhaustive()


def _run_ctrl_qubit_exhaustive():
    for c_in_val in (0, 1):
        for use_kwarg in (False, True):
            for ctrl_val in (0, 1):
                add = _mk_add_ctrl_qubit(c_in_val, use_kwarg, ctrl_val)
                for N in (2,):
                    for L in (2,):
                        for j in range(1 << N):
                            for k in range(1 << L):
                                A, B = add(N, L, j, k)
                                assert A == j
                                # ctrl off -> addition is a no-op (B unchanged)
                                if ctrl_val:
                                    assert B == (k + j + c_in_val) % (1 << L)
                                else:
                                    assert B == k


def test_thapliyal_adder_ctrl_qubit_dynamic():
    _run_ctrl_qubit_exhaustive()


def _run_cout_ctrl_exhaustive():
    for c_in_val in (0, 1):
        for ctrl_val in (0, 1):
            add = _mk_add_cout_ctrl(c_in_val, ctrl_val)
            for L in (2,):
                for j in range(1 << L):
                    for k in range(1 << L):
                        total = k + j + c_in_val
                        A_res, B_res, cout = add(L, j, k)
                        assert A_res == j
                        if ctrl_val:
                            assert B_res == total % (1 << L)
                            assert cout == (total >= (1 << L))
                        else:
                            # ctrl off -> no-op: B unchanged and c_out stays |0>
                            assert B_res == k
                            assert not cout


def test_thapliyal_adder_cout_ctrl_dynamic():
    _run_cout_ctrl_exhaustive()
