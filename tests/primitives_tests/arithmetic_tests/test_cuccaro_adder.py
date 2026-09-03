# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Tests for the Cuccaro ripple-carry in-place adder."""

import re

import pytest

from qrisp import (
    QuantumBool,
    QuantumFloat,
    QuantumModulus,
    QuantumVariable,
    boolean_simulation,
    control,
    cuccaro_adder,
    measure,
    x,
)
from qrisp.circuit import Qubit
from qrisp.misc import int_encoder

# ---------------------------------------------------------------------------
# Static smoke tests — a few representative cases with small registers to catch
# gross regressions without the full statevector-simulation cost. Exhaustive
# coverage lives in the boolean_simulation tests further down.
# ---------------------------------------------------------------------------


def test_cuccaro_adder_static_quantum_a():
    """Quantum a + quantum b, equal size, no optional args."""
    a = QuantumFloat(3)
    b = QuantumFloat(3)
    a[:] = 5
    b[:] = 3
    cuccaro_adder(a, b)
    assert b.get_measurement() == {0: 1.0}  # (5 + 3) % 8


def test_cuccaro_adder_static_classical_a():
    """Classical a + quantum b."""
    b = QuantumFloat(3)
    b[:] = 3
    cuccaro_adder(5, b)
    assert b.get_measurement() == {0: 1.0}


def test_cuccaro_adder_static_cin():
    """c_in with classical a."""
    b = QuantumFloat(3)
    b[:] = 2
    c_in = QuantumBool()
    x(c_in[0])
    cuccaro_adder(3, b, c_in=c_in)
    assert b.get_measurement() == {6: 1.0}  # 2 + 3 + 1


def test_cuccaro_adder_static_cin_qubit():
    """c_in of type Qubit."""
    b = QuantumFloat(3)
    b[:] = 2
    qv = QuantumVariable(1)
    c_in = qv[0]
    assert isinstance(c_in, Qubit)
    x(c_in)
    cuccaro_adder(3, b, c_in=c_in)
    assert b.get_measurement() == {6: 1.0}


def test_cuccaro_adder_static_c_in_type_error():
    """TypeError when c_in is neither QuantumBool nor Qubit."""
    b = QuantumFloat(4)
    b[:] = 3
    for bad_c_in in (QuantumFloat(2), "invalid", 42):
        with pytest.raises(TypeError, match="c_in must be of type QuantumBool or Qubit"):
            cuccaro_adder(1, b, c_in=bad_c_in)


def test_cuccaro_adder_static_cout_overflow():
    """c_out captures overflow."""
    b = QuantumFloat(3)
    b[:] = 6
    c_out = QuantumBool()
    cuccaro_adder(3, b, c_out=c_out)
    assert b.get_measurement() == {1: 1.0}  # (6 + 3) % 8
    assert c_out.get_measurement() == {True: 1.0}


def test_cuccaro_adder_static_ctrl():
    """Controlled addition (ctrl kwarg)."""
    a = QuantumFloat(3)
    b = QuantumFloat(3)
    a[:] = 3
    b[:] = 5
    ctrl = QuantumBool()
    x(ctrl[0])
    cuccaro_adder(a, b, ctrl=ctrl)
    assert b.get_measurement() == {0: 1.0}  # (5 + 3) % 8


def test_cuccaro_adder_static_cin_cout():
    """c_in + c_out together."""
    b = QuantumFloat(3)
    b[:] = 6
    c_in = QuantumBool()
    x(c_in[0])
    c_out = QuantumBool()
    cuccaro_adder(3, b, c_in=c_in, c_out=c_out)
    assert b.get_measurement() == {2: 1.0}  # (6 + 3 + 1) % 8
    assert c_out.get_measurement() == {True: 1.0}


def test_cuccaro_adder_static_cin_qubit_cout():
    """c_in of type Qubit with c_out together."""
    b = QuantumFloat(3)
    b[:] = 6
    qv = QuantumVariable(1)
    c_in = qv[0]
    assert isinstance(c_in, Qubit)
    x(c_in)
    c_out = QuantumBool()
    cuccaro_adder(3, b, c_in=c_in, c_out=c_out)
    assert b.get_measurement() == {2: 1.0}  # (6 + 3 + 1) % 8
    assert c_out.get_measurement() == {True: 1.0}


def test_cuccaro_adder_static_cout_ctrl():
    """c_out + ctrl (ctrl=on) — exercises the MAJ-phase cx(a[-1], c_out) path."""
    a = QuantumFloat(3)
    b = QuantumFloat(3)
    a[:] = 6
    b[:] = 6
    c_out = QuantumBool()
    ctrl = QuantumBool()
    x(ctrl[0])
    cuccaro_adder(a, b, c_out=c_out, ctrl=ctrl)
    assert b.get_measurement() == {4: 1.0}  # (6 + 6) % 8
    assert c_out.get_measurement() == {True: 1.0}


def test_cuccaro_adder_static_inputs_unmodified():
    """Input QuantumFloat sizes are unchanged after addition."""
    a = QuantumFloat(5)
    b = QuantumFloat(7)
    orig_a, orig_b = a.size, b.size
    a[:] = 3
    b[:] = 4
    cuccaro_adder(a, b)
    assert a.size == orig_a
    assert b.size == orig_b


# -- list[Qubit] compatibility -----------------------------------------------


@pytest.mark.parametrize("a_spec", ["quantum", "classical", "list"])
@pytest.mark.parametrize("b_spec", ["variable", "list"])
def test_cuccaro_adder_static_list_combinations(a_spec, b_spec):
    """QuantumVariable and list[Qubit] inputs in all a/b combinations."""
    b = QuantumFloat(3)
    b[:] = 3
    if a_spec == "classical":
        a_arg = 5
    else:
        a = QuantumFloat(3)
        a[:] = 5
        a_arg = a[:] if a_spec == "list" else a
    b_arg = b[:] if b_spec == "list" else b

    cuccaro_adder(a_arg, b_arg)

    assert b.get_measurement() == {0: 1.0}  # (3 + 5) % 8
    if a_spec == "quantum":
        assert a.get_measurement() == {5: 1.0}


def test_cuccaro_adder_static_list_unequal_sizes():
    """list[Qubit] inputs of unequal size (truncation + extension ancillas)."""
    a = QuantumFloat(5)
    a[:] = 3
    b = QuantumFloat(3)
    b[:] = 7
    cuccaro_adder(a[:], b[:])
    assert b.get_measurement() == {2: 1.0}  # (7 + 3) % 8

    a = QuantumFloat(3)
    a[:] = 3
    b = QuantumFloat(5)
    b[:] = 7
    cuccaro_adder(a[:], b[:])
    assert b.get_measurement() == {10: 1.0}  # 7 + 3


@pytest.mark.parametrize("b_is_list", [False, True])
def test_cuccaro_adder_static_classical_a_larger_than_b(b_is_list):
    """Classical a wider than the target register wraps modulo 2**len(b)."""
    b = QuantumFloat(3)
    b[:] = 5
    cuccaro_adder(10, b[:] if b_is_list else b)
    assert b.get_measurement() == {7: 1.0}  # (5 + 10) % 8


# -- other quantum types ------------------------------------------------------


def _measure_int(qv):
    """Return the single-shot integer outcome of ``qv``.

    Works regardless of the decoder of the concrete quantum type (int, bool or
    little-endian bit string keys).
    """
    ((key, _),) = qv.get_measurement().items()
    if isinstance(key, bool):
        return int(key)
    if isinstance(key, str):
        return int(key[::-1], 2) if key else 0
    return key


def test_cuccaro_adder_static_quantum_variable():
    """Base QuantumVariable registers as a and b."""
    a_val, b_val = 5, 3
    a = QuantumVariable(3)
    b = QuantumVariable(3)
    int_encoder(a, a_val)
    int_encoder(b, b_val)
    cuccaro_adder(a, b)
    assert _measure_int(a) == a_val
    assert _measure_int(b) == 0  # (3 + 5) % 8

    b = QuantumVariable(3)
    int_encoder(b, b_val)
    cuccaro_adder(a_val, b)
    assert _measure_int(b) == 0  # (3 + 5) % 8


def test_cuccaro_adder_static_quantum_bool():
    """QuantumBool (single-qubit) registers as a and b."""
    a = QuantumBool()
    b = QuantumBool()
    a.flip()  # a = 1
    b.flip()  # b = 1
    cuccaro_adder(a, b)
    assert b.get_measurement() == {False: 1.0}  # (1 + 1) % 2

    b = QuantumBool()
    cuccaro_adder(1, b)
    assert b.get_measurement() == {True: 1.0}


def test_cuccaro_adder_static_quantum_modulus():
    """QuantumModulus registers as a and b (sum stays below the modulus)."""
    a = QuantumModulus(13)
    b = QuantumModulus(13)
    a[:] = 5
    b[:] = 3
    cuccaro_adder(a, b)
    assert a.get_measurement() == {5: 1.0}
    assert b.get_measurement() == {8: 1.0}

    b = QuantumModulus(13)
    b[:] = 3
    cuccaro_adder(5, b)
    assert b.get_measurement() == {8: 1.0}


# -- input validation and issue #839 regression -------------------------------


def test_cuccaro_adder_static_invalid_inputs_raise_value_error():
    """Non-quantum b and non-qubit lists are rejected with ValueError."""
    a = QuantumFloat(3)
    a[:] = 1
    b = QuantumFloat(3)
    b[:] = 1

    b_msg = "The second argument must be of type QuantumVariable, DynamicQubitArray or a non-empty list[Qubit]."
    a_msg = "If the first argument is a list, it must contain only Qubits."

    with pytest.raises(ValueError, match=re.escape(b_msg)):
        cuccaro_adder(a, 7)
    with pytest.raises(ValueError, match=re.escape(b_msg)):
        cuccaro_adder(a, [])
    with pytest.raises(ValueError, match=re.escape(a_msg)):
        cuccaro_adder([1, 0], b)


def test_cuccaro_adder_quantum_modulus_issue_839():
    """QuantumModulus with cuccaro_adder as inpl_adder (regression for #839).

    QuantumModulus hands the configured adder a raw ``list[Qubit]`` target (the
    auxiliary register of the Montgomery multiplication). Before the fix this
    raised ``AttributeError: 'list' object has no attribute 'duplicate'`` while
    the circuit was being constructed.
    """
    a = QuantumModulus(13, inpl_adder=cuccaro_adder)
    a[:] = 5
    a *= 10
    a.qs.compile()

    @boolean_simulation
    def montgomery_multiply(N, value, factor):
        qm = QuantumModulus(N, inpl_adder=cuccaro_adder)
        qm[:] = value
        qm *= factor
        return measure(qm)

    assert montgomery_multiply(13, 5, 10) == (5 * 10) % 13


# ---------------------------------------------------------------------------
# Exhaustive tests via @boolean_simulation.
#
# Configuration is captured as CLOSURE variables (not function parameters) so
# JAX treats them as compile-time constants during tracing. Sweeps run in plain
# Python outer functions to keep the JAX cache warm.
# ---------------------------------------------------------------------------


def _mk_add(inputs, c_in_kind=None, c_out=False, ctrl_kind=None, c_in_val=0):
    """Factory for a ``@boolean_simulation`` ``cuccaro_adder`` wrapper.

    - inputs: (a_kind, b_kind), with a_kind in {"quantum", "classical", "list"}
      and b_kind in {"variable", "list"}.
    - c_in_kind: None | "qbool" | "qubit" — optional carry-in.
    - c_out: add a carry-out register.
    - ctrl_kind: None | "kwarg" | "env" — optional control, via ``ctrl=`` or
      ``with control()``.
    - c_in_val: value (0/1) of the carry-in.
    """
    a_kind, b_kind = inputs

    def build_c_in():
        if c_in_kind == "qubit":
            c_in = QuantumVariable(1)[0]
            if c_in_val:
                x(c_in)
        else:
            c_in = QuantumBool()
            if c_in_val:
                c_in.flip()
        return c_in

    def apply(a_arg, b_arg, kwargs):
        if ctrl_kind is None:
            cuccaro_adder(a_arg, b_arg, **kwargs)
        else:
            qbl = QuantumBool()
            qbl.flip()  # ctrl is always |1>
            if ctrl_kind == "kwarg":
                cuccaro_adder(a_arg, b_arg, ctrl=qbl, **kwargs)
            else:
                with control(qbl):
                    cuccaro_adder(a_arg, b_arg, **kwargs)

    @boolean_simulation
    def add(N, L, j, k):
        B = QuantumFloat(L)
        B[:] = k
        b_arg = B[:] if b_kind == "list" else B

        if a_kind == "classical":
            a_arg = j
        else:
            A = QuantumFloat(N)
            A[:] = j
            a_arg = A[:] if a_kind == "list" else A

        kwargs = {}
        if c_in_kind is not None:
            kwargs["c_in"] = build_c_in()
        if c_out:
            c_out_qb = QuantumBool()
            kwargs["c_out"] = c_out_qb

        apply(a_arg, b_arg, kwargs)

        res = []
        if a_kind != "classical":
            res.append(measure(A))
        res.append(measure(B))
        if c_out:
            res.append(measure(c_out_qb))
        return tuple(res)

    return add


def _sweep(add, sizes_a, sizes_b, check):
    for N in sizes_a:
        for L in sizes_b:
            for j in range(1 << N):
                for k in range(1 << L):
                    check(add, N, L, j, k)


def _sweep_equal(add, sizes, check):
    for N in sizes:
        for j in range(1 << N):
            for k in range(1 << N):
                check(add, N, N, j, k)


def _check_qq(c_in_val=0):
    """Check factory: quantum a, A unchanged, B += a + c_in (mod 2**L)."""

    def check(add, N, L, j, k):
        A, B = add(N, L, j, k)
        assert A == j
        assert B == (k + j + c_in_val) % (1 << L)

    return check


def _check_cout_qq(c_in_val=0):
    """Check factory: quantum a with carry-out, overflow captured in c_out."""

    def check(add, N, L, j, k):
        A, B, cout = add(N, L, j, k)
        total = k + j + c_in_val
        assert A == j
        assert B == total % (1 << L)
        assert cout == (total >= (1 << L))

    return check


def _check_cq(c_in_val=0):
    """Check factory: classical a, B += a + c_in (mod 2**L)."""

    def check(add, N, L, j, k):
        (B,) = add(N, L, j, k)
        assert B == (k + j + c_in_val) % (1 << L)

    return check


def _check_cout_cq(c_in_val=0):
    """Check factory: classical a with carry-out, overflow captured in c_out."""

    def check(add, N, L, j, k):
        B, cout = add(N, L, j, k)
        total = k + j + c_in_val
        assert B == total % (1 << L)
        assert cout == (total >= (1 << L))

    return check


def test_cuccaro_adder_dynamic_basic():
    """Exhaustive quantum-quantum addition over small register sizes."""
    _sweep(_mk_add(("quantum", "variable")), range(2, 6), range(2, 6), _check_qq())


def test_cuccaro_adder_dynamic_cin():
    """Exhaustive addition with a QuantumBool carry-in."""
    for c_in_val in (0, 1):
        add = _mk_add(("quantum", "variable"), c_in_kind="qbool", c_in_val=c_in_val)
        _sweep(add, range(2, 6), range(2, 6), _check_qq(c_in_val))


def test_cuccaro_adder_dynamic_cin_qubit():
    """Exhaustive addition with a bare Qubit carry-in."""
    for c_in_val in (0, 1):
        add = _mk_add(("quantum", "variable"), c_in_kind="qubit", c_in_val=c_in_val)
        _sweep(add, range(2, 6), range(2, 6), _check_qq(c_in_val))


def test_cuccaro_adder_dynamic_cout():
    """Exhaustive classical-a addition capturing the carry-out overflow."""
    for c_in_val in (0, 1):
        add = _mk_add(("classical", "variable"), c_in_kind="qbool", c_out=True, c_in_val=c_in_val)
        _sweep_equal(add, range(2, 6), _check_cout_cq(c_in_val))


def test_cuccaro_adder_dynamic_cout_qubit():
    """Exhaustive classical-a addition with a bare Qubit carry-in and carry-out."""
    for c_in_val in (0, 1):
        add = _mk_add(("classical", "variable"), c_in_kind="qubit", c_out=True, c_in_val=c_in_val)
        _sweep_equal(add, range(2, 6), _check_cout_cq(c_in_val))


def test_cuccaro_adder_dynamic_cout_equal_sizes():
    """Exhaustive equal-size quantum addition with carry-out."""
    for c_in_val in (0, 1):
        add = _mk_add(("quantum", "variable"), c_in_kind="qbool", c_out=True, c_in_val=c_in_val)
        _sweep_equal(add, range(2, 6), _check_cout_qq(c_in_val))


def test_cuccaro_adder_dynamic_ctrl():
    """Exhaustive controlled addition via ctrl kwarg and control environment."""
    for c_in_val in (0, 1):
        for ctrl_kind in ("kwarg", "env"):
            add = _mk_add(("quantum", "variable"), c_in_kind="qbool", ctrl_kind=ctrl_kind, c_in_val=c_in_val)
            _sweep(add, range(2, 5), range(2, 5), _check_qq(c_in_val))


def test_cuccaro_adder_dynamic_ctrl_qubit():
    """Exhaustive controlled addition with a bare Qubit carry-in."""
    for c_in_val in (0, 1):
        for ctrl_kind in ("kwarg", "env"):
            add = _mk_add(("quantum", "variable"), c_in_kind="qubit", ctrl_kind=ctrl_kind, c_in_val=c_in_val)
            _sweep(add, range(2, 5), range(2, 5), _check_qq(c_in_val))


def test_cuccaro_adder_dynamic_cout_ctrl():
    """Exhaustive addition with carry-out and control combined."""
    for c_in_val in (0, 1):
        add = _mk_add(("quantum", "variable"), c_in_kind="qbool", c_out=True, ctrl_kind="kwarg", c_in_val=c_in_val)
        _sweep_equal(add, range(2, 5), _check_cout_qq(c_in_val))


def test_cuccaro_adder_dynamic_list_target():
    """Exhaustive classical-quantum and quantum-quantum addition on list[Qubit]."""
    add_cq = _mk_add(("classical", "list"))
    _sweep_equal(add_cq, range(2, 5), _check_cq())

    add_qq = _mk_add(("list", "list"))
    _sweep(add_qq, range(2, 5), range(2, 5), _check_qq())


def test_cuccaro_adder_dynamic_classical_a_wider_than_b():
    """Classical a wider than the target wraps modulo 2**len(b) in dynamic mode."""
    add = _mk_add(("classical", "variable"))
    check = _check_cq()
    for N in range(2, 6):
        for j in range(1 << (N + 3)):
            for k in range(1 << N):
                check(add, N, N, j, k)


# -- other quantum types (dynamic) -------------------------------------------


QTYPE_SPECS = {
    "qvariable": {
        "make": QuantumVariable,
        "encode": int_encoder,
        "sizes": [2, 3, 4],
        "values": lambda n: range(1 << n),
        "expected": lambda j, k, n: (k + j) % (1 << n),
    },
    "qbool": {
        "make": lambda n: QuantumBool(),
        "encode": int_encoder,
        "sizes": [1],
        "values": lambda n: range(2),
        "expected": lambda j, k, n: (k + j) % 2,
    },
    "qmodulus": {
        "make": QuantumModulus,
        "encode": lambda qv, v: qv.__setitem__(slice(None), v),
        "sizes": [13],
        "values": range,
        "expected": lambda j, k, n: ((k + j) % (1 << n.bit_length())) % n,
    },
}


def _mk_type_add(make, encode, a_kind):
    """Factory for a ``@boolean_simulation`` adder on non-QuantumFloat types."""

    @boolean_simulation
    def add(n_a, n_b, j, k):
        if a_kind == "quantum":
            A = make(n_a)
            encode(A, j)
            a_arg = A
        else:
            a_arg = j
        B = make(n_b)
        encode(B, k)
        cuccaro_adder(a_arg, B)
        if a_kind == "quantum":
            return measure(A), measure(B)
        return measure(B)

    return add


@pytest.mark.parametrize("qtype", ["qvariable", "qbool", "qmodulus"])
def test_cuccaro_adder_dynamic_quantum_types(qtype):
    """Exhaustive addition on QuantumVariable, QuantumBool and QuantumModulus."""
    spec = QTYPE_SPECS[qtype]
    add_qq = _mk_type_add(spec["make"], spec["encode"], "quantum")
    add_cq = _mk_type_add(spec["make"], spec["encode"], "classical")

    for n_a in spec["sizes"]:
        for n_b in spec["sizes"]:
            for j in spec["values"](n_a):
                for k in spec["values"](n_b):
                    A, B = add_qq(n_a, n_b, j, k)
                    assert A == spec["expected"](j, 0, n_a)  # A is unchanged
                    assert B == spec["expected"](j, k, n_b)

    for n in spec["sizes"]:
        for j in spec["values"](n):
            for k in spec["values"](n):
                assert add_cq(n, n, j, k) == spec["expected"](j, k, n)
