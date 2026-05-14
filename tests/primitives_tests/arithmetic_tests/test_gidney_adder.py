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
    QuantumVariable,
    boolean_simulation,
    control,
    cuccaro_adder,
    gidney_adder,
    measure,
    x,
)
@pytest.mark.parametrize(
    "a_spec, a_val, b_bits, b_val, expected_a, expected_b",
    [
        # Quantum-quantum (various sizes)
        (("qf", 13), 20, 14, 14, {20: 1.0}, {34: 1.0}),
        (("qf", 10), 15, 5, 7, {15: 1.0}, {(7 + 15) % 32: 1.0}),
        # Classical int
        (("classical", None), 20, 15, 14, 20, {34: 1.0}),
        # Edge cases: zero, single qubit
        (("qf", 10), 0, 10, 42, {0: 1.0}, {42: 1.0}),
        (("classical", None), 0, 10, 42, 0, {42: 1.0}),
        (("qf", 1), 0, 1, 1, {0: 1.0}, {1: 1.0}),
        # Various sizes
        (("qf", 5), 7, 10, 15, {7: 1.0}, {22: 1.0}),
        (("qf", 8), 100, 8, 50, {100: 1.0}, {150: 1.0}),
        (("qf", 16), 255, 4, 3, {255: 1.0}, {(3 + 255) % 16: 1.0}),
        # String and special inputs
        (("classical", None), "10", 8, 5, "10", {6: 1.0}),
        (("classical", None), "", 8, 42, "", {42: 1.0}),
        (("classical", None), np.int64(5), 8, 10, np.int64(5), {15: 1.0}),
        # Classical modulo reduction
        (("classical", None), 17, 4, 0, 17, {1: 1.0}),
        (("classical", None), 255, 4, 7, 255, {6: 1.0}),
        # Negative classical a
        (("classical", None), -3, 4, 5, -3, {2: 1.0}),
        (("classical", None), -1, 8, 0, -1, {255: 1.0}),
        # List-of-qubits input
        (("qf_list", 4), 5, 4, 3, None, {8: 1.0}),
        (("qf_list", 3), 7, 8, 100, None, {107: 1.0}),
    ],
)
def test_gidney_adder_basic(a_spec, a_val, b_bits, b_val, expected_a, expected_b):
    """Verify b += a for various input types and sizes."""
    kind, n = a_spec
    if kind == "qf":
        a = QuantumFloat(n)
        a[:] = a_val
    elif kind == "qf_list":
        a = QuantumFloat(n)
        a[:] = a_val
        a = a[:]
    else:
        a = a_val

    b = QuantumFloat(b_bits)
    b[:] = b_val
    gidney_adder(a, b)

    assert b.get_measurement() == expected_b
    if expected_a is not None and isinstance(a, QuantumFloat):
        assert a.get_measurement() == expected_a


@pytest.mark.parametrize(
    "a_spec, a_val, b_bits, b_val, expected_b, expected_cout",
    [
        # Quantum-quantum (no overflow)
        (("qf", 13), 20, 14, 14, {34: 1.0}, {0: 1.0}),
        (("qf", 13), 20, 13, 14, {34: 1.0}, {0: 1.0}),
        # Classical int (no overflow, overflow)
        (("classical", None), 20, 15, 14, {34: 1.0}, {0: 1.0}),
        (("classical", None), 10, 8, 20, {30: 1.0}, {0: 1.0}),
        (("qf", 1), 1, 1, 1, {0: 1.0}, {1: 1.0}),
        # Classical overflow
        (("classical", None), 255, 8, 128, {127: 1.0}, {1: 1.0}),
        (("classical", None), 200, 8, 100, {44: 1.0}, {1: 1.0}),
        (("classical", None), 100, 8, 200, {44: 1.0}, {1: 1.0}),
    ],
)
def test_gidney_adder_with_carry_out(
    a_spec, a_val, b_bits, b_val, expected_b, expected_cout
):
    """Verify b += a with carry-out for overflow detection."""
    kind, n = a_spec
    if kind == "qf":
        a = QuantumFloat(n)
        a[:] = a_val
    elif kind == "qf_list":
        qf = QuantumFloat(n)
        qf[:] = a_val
        a = qf[:]
    else:
        a = a_val

    b = QuantumFloat(b_bits)
    b[:] = b_val
    c_out = QuantumBool()

    gidney_adder(a, b, c_out=c_out)

    assert b.get_measurement() == expected_b
    assert c_out.get_measurement() == expected_cout


@pytest.mark.parametrize(
    "a_spec, a_val, b_val, c_in_val, expected_b",
    [
        # Classical a, c_in=0 (implicit: 5+0+0=5)
        # Classical a, c_in=1
        (("classical", None), 5, 0, 1, {6: 1.0}),
        (("classical", None), 20, 0, 1, {21: 1.0}),
        # Quantum a, c_in effects
        (("qf", 8), 50, 100, 1, {151: 1.0}),
        (("qf", 8), 50, 100, 0, {150: 1.0}),
        (("qf", 8), 10, 0, 1, {11: 1.0}),
    ],
)
def test_gidney_adder_with_carry_in(a_spec, a_val, b_val, c_in_val, expected_b):
    """Verify b += a + c_in."""
    kind, n = a_spec
    if kind == "qf":
        a = QuantumFloat(n)
        a[:] = a_val
    elif kind == "qf_list":
        qf = QuantumFloat(n)
        qf[:] = a_val
        a = qf[:]
    else:
        a = a_val

    b = QuantumFloat(8 if a_spec[1] is None else max(a_spec[1], 8))
    b[:] = b_val

    c_in = QuantumBool()
    if c_in_val:
        x(c_in[0])

    gidney_adder(a, b, c_in=c_in)

    assert b.get_measurement() == expected_b


@pytest.mark.parametrize(
    "a_spec, a_val, b_val, expected_b, expected_cout",
    [
        # Quantum a, various results
        (("qf", 8), 100, 150, {251: 1.0}, {0: 1.0}),
        (("qf", 8), 200, 55, {0: 1.0}, {1: 1.0}),
        # Classical a, c_in edge cases
        (("classical", None), 0, 255, {0: 1.0}, {1: 1.0}),
        (("classical", None), 127, 128, {0: 1.0}, {1: 1.0}),
        (("classical", None), 1, 1, {3: 1.0}, {0: 1.0}),
    ],
)
def test_gidney_adder_with_carry_in_and_carry_out(
    a_spec, a_val, b_val, expected_b, expected_cout
):
    """Verify b += a + c_in with carry-out."""
    kind, n = a_spec
    if kind == "qf":
        a = QuantumFloat(n)
        a[:] = a_val
    elif kind == "qf_list":
        qf = QuantumFloat(n)
        qf[:] = a_val
        a = qf[:]
    else:
        a = a_val

    b = QuantumFloat(8)
    b[:] = b_val

    c_in = QuantumBool()
    x(c_in[0])
    c_out = QuantumBool()

    gidney_adder(a, b, c_in=c_in, c_out=c_out)

    assert b.get_measurement() == expected_b
    assert c_out.get_measurement() == expected_cout


@pytest.mark.parametrize(
    "a_spec, a_val, b_bits, b_val, ctrl_on, expected_b",
    [
        # Quantum a, ctrl on
        (("qf", 10), 3, 11, 5, True, {8: 1.0}),
        # Quantum a, ctrl off
        (("qf", 10), 3, 11, 5, False, {5: 1.0}),
        # Classical a, ctrl on
        (("classical", None), 5, 8, 10, True, {15: 1.0}),
        # Classical a, ctrl off
        (("classical", None), 5, 8, 10, False, {10: 1.0}),
        # Classical zero with ctrl on
        (("classical", None), 0, 8, 42, True, {42: 1.0}),
        # Single-qubit classical a with ctrl on
        (("classical", None), 1, 1, 0, True, {1: 1.0}),
        # Quantum a + c_in, ctrl off (nothing should happen)
        (("qf", 8), 50, 8, 100, False, {100: 1.0}),
    ],
    ids=[
        "quantum_a_ctrl_on",
        "quantum_a_ctrl_off",
        "classical_a_ctrl_on",
        "classical_a_ctrl_off",
        "classical_zero_ctrl_on",
        "single_qubit_classical_ctrl_on",
        "quantum_a_cin_ctrl_off",
    ],
)
def test_gidney_adder_with_control(
    a_spec, a_val, b_bits, b_val, ctrl_on, expected_b
):
    """Verify controlled addition only fires when ctrl=|1⟩."""
    kind, n = a_spec
    if kind == "qf":
        a = QuantumFloat(n)
        a[:] = a_val
    elif kind == "qf_list":
        qf = QuantumFloat(n)
        qf[:] = a_val
        a = qf[:]
    else:
        a = a_val

    b = QuantumFloat(b_bits)
    b[:] = b_val
    ctrl_qb = QuantumBool()
    if ctrl_on:
        x(ctrl_qb[0])

    # For the c_in + ctrl_off case, also set up c_in
    use_c_in = (a_spec[0] == "qf" and not ctrl_on and a_val == 50)
    c_in = None
    if use_c_in:
        c_in = QuantumBool()
        x(c_in[0])

    with control(ctrl_qb):
        gidney_adder(a, b, c_in=c_in)

    assert b.get_measurement() == expected_b


def test_gidney_adder_carry_in_carry_out_ctrl_quantum_a():
    """All optional params active with quantum a — exercises the lsb_ctrl_anc allocation."""
    a = QuantumFloat(8)
    b = QuantumFloat(8)
    a[:] = 50
    b[:] = 100
    c_in = QuantumBool()
    x(c_in[0])
    c_out = QuantumBool()
    ctrl_qb = QuantumBool()
    x(ctrl_qb[0])

    with control(ctrl_qb):
        gidney_adder(a, b, c_in=c_in, c_out=c_out)

    assert b.get_measurement() == {151: 1.0}
    assert c_out.get_measurement() == {0: 1.0}


@pytest.mark.parametrize(
    "a_val, b_val, ctrl_on, expected_b, expected_cout",
    [
        (55, 200, True, {0: 1.0}, {1: 1.0}),    # ctrl on: 200+55+1=256→0, cout=1
        (55, 200, False, {200: 1.0}, {0: 1.0}),  # ctrl off: unchanged
    ],
)
def test_gidney_adder_classical_a_cin_cout_ctrl(
    a_val, b_val, ctrl_on, expected_b, expected_cout
):
    """All optional params (c_in, c_out, ctrl) active with classical a."""
    b = QuantumFloat(8)
    b[:] = b_val
    c_in = QuantumBool()
    x(c_in[0])
    c_out = QuantumBool()
    ctrl_qb = QuantumBool()
    if ctrl_on:
        x(ctrl_qb[0])

    with control(ctrl_qb):
        gidney_adder(a_val, b, c_in=c_in, c_out=c_out)

    assert b.get_measurement() == expected_b
    assert c_out.get_measurement() == expected_cout


@pytest.mark.parametrize(
    "a_spec, a_val, b_bits, b_val",
    [
        (("qf", 6), 5, 10, 7),
        (("classical", None), 17, 10, 42),
    ],
    ids=["quantum_a", "classical_a"],
)
def test_gidney_adder_ancilla_cleanup(a_spec, a_val, b_bits, b_val):
    """Verify that internal ancillas (gidney_anc*, gidney_a_ext*) are deallocated."""
    kind, n = a_spec
    if kind == "qf":
        a = QuantumFloat(n)
        a[:] = a_val
    elif kind == "qf_list":
        qf = QuantumFloat(n)
        qf[:] = a_val
        a = qf[:]
    else:
        a = a_val

    b = QuantumFloat(b_bits)
    b[:] = b_val

    gidney_adder(a, b)

    remaining = [v.name for v in b.qs.qv_list if "gidney" in v.name]
    assert remaining == [], f"Leaked ancilla variables: {remaining}"


def test_gidney_adder_raw_qubit_carry_in():
    """Pass a raw Qubit instead of QuantumBool as c_in."""
    c_in_var = QuantumVariable(1)
    x(c_in_var[0])

    b = QuantumFloat(8)
    b[:] = 10

    gidney_adder(5, b, c_in=c_in_var[0])

    assert b.get_measurement() == {16: 1.0}


def test_gidney_adder_raw_qubit_carry_out():
    """Pass a raw Qubit instead of QuantumBool as c_out."""
    c_out_var = QuantumVariable(1)

    b = QuantumFloat(8)
    b[:] = 200

    gidney_adder(100, b, c_out=c_out_var[0])

    assert b.get_measurement() == {44: 1.0}
    assert c_out_var.get_measurement() == {"1": 1.0}


def test_gidney_adder_inputs_unmodified_size():
    """Verify the size of inputs are unmodified (in-place) when they are initially of unequal size."""
    a = QuantumFloat(10)
    b = QuantumFloat(12)
    original_size_a = a.size
    original_size_b = b.size
    a[:] = 5
    b[:] = 7

    gidney_adder(a, b)

    assert a.size == original_size_a
    assert b.size == original_size_b


def test_gidney_adder_t_depth_significantly_lower_than_cuccaro():
    """Verify that the gidney adder achieves significant T-depth reduction across inputs of
    different sizes."""
    for qf_size in [8, 12, 16]:
        a_cuccaro = QuantumFloat(qf_size)
        b_cuccaro = QuantumFloat(qf_size)
        a_cuccaro[:] = 5
        b_cuccaro[:] = 10
        cuccaro_adder(a_cuccaro, b_cuccaro)
        cuccaro_t_depth = b_cuccaro.qs.compile().t_depth()

        a_gidney = QuantumFloat(qf_size)
        b_gidney = QuantumFloat(qf_size)
        a_gidney[:] = 5
        b_gidney[:] = 10
        gidney_adder(a_gidney, b_gidney)
        gidney_t_depth = b_gidney.qs.compile().t_depth()

        assert gidney_t_depth < cuccaro_t_depth


def test_gidney_adder_t_count():
    """Verify that the Gidney adder for two n-qubit inputs uses 4*(n-1) T gates.

    This follows from Fig. 1 of the Gidney paper (https://arxiv.org/abs/1709.06648):
    each of the (n-1) forward Toffolis decomposes into 4 T gates via the logical-AND
    construction, while the (n-1) inverse Toffolis use measurement-based uncomputation
    with 0 T gates, giving a total T count of 4*(n-1).
    """
    for n in [3, 5, 8, 12]:
        a = QuantumFloat(n)
        b = QuantumFloat(n)
        a[:] = 2
        b[:] = 3

        gidney_adder(a, b)

        qc = b.qs.compile(compile_mcm=True).transpile()
        ops = qc.count_ops()
        t_count = ops.get("t", 0) + ops.get("t_dg", 0)

        assert t_count == 4 * (n - 1)


def test_gidney_adder_jasp_mode():
    """Verify the function works as expected in dynamic (jasp) mode.
    This test covers both cases where inputs are of equal and unequal size."""

    @boolean_simulation
    def main(N, L, j, k):
        A = QuantumFloat(N)
        B = QuantumFloat(L)
        A[:] = j
        B[:] = k
        gidney_adder(A, B)
        return measure(A), measure(B)

    for N in range(2, 5):
        j_max = 1 << N
        for L in range(2, 5):
            k_max = 1 << L
            mod = 1 << L
            for j in range(j_max):
                for k in range(k_max):
                    A, B = main(N, L, j, k)
                    assert A == j
                    assert B == (k + j) % mod


def test_gidney_adder_jasp_mode_with_control():
    """Verify the gidney adder is triggered when the control qubit is in the |1⟩ state
    in dynamic (jasp) mode."""

    @boolean_simulation
    def main(N, L, j, k):
        A = QuantumFloat(N)
        B = QuantumFloat(L)
        A[:] = j
        B[:] = k
        qbl = QuantumBool()
        qbl.flip()

        with control(qbl):
            gidney_adder(A, B)
        return measure(A), measure(B)

    for N in range(2, 5):
        j_max = 1 << N
        for L in range(2, 5):
            k_max = 1 << L
            mod = 1 << L
            for j in range(j_max):
                for k in range(k_max):
                    A, B = main(N, L, j, k)
                    assert A == j
                    assert B == (k + j) % mod


def test_gidney_adder_jasp_mode_with_carry_in():
    """Carry-in in dynamic mode with quantum a."""

    @boolean_simulation
    def main(N, j, k):
        A = QuantumFloat(N)
        B = QuantumFloat(N)
        A[:] = j
        B[:] = k
        c_in = QuantumBool()
        c_in.flip()
        gidney_adder(A, B, c_in=c_in)
        return measure(A), measure(B)

    for N in range(2, 5):
        vals_max = 1 << N
        mod = 1 << N
        for j in range(vals_max):
            for k in range(vals_max):
                A, B = main(N, j, k)
                assert A == j
                assert B == (k + j + 1) % mod


def test_gidney_adder_jasp_mode_ctrl_off():
    """In dynamic mode, ctrl=|0⟩ means b is untouched."""

    @boolean_simulation
    def main(N, j, k):
        A = QuantumFloat(N)
        B = QuantumFloat(N)
        A[:] = j
        B[:] = k
        qbl = QuantumBool()

        with control(qbl):
            gidney_adder(A, B)
        return measure(A), measure(B)

    for N in range(2, 4):
        vals_max = 1 << N
        for j in range(vals_max):
            for k in range(vals_max):
                A, B = main(N, j, k)
                assert A == j
                assert B == k