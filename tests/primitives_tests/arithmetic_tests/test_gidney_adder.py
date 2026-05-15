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
        (("qf", 13), 20, 14, 14, {20: 1.0}, {34: 1.0}),  # quantum a and wider b without wrap
        (("qf", 10), 15, 5, 7, {15: 1.0}, {(7 + 15) % 32: 1.0}),  # quantum a larger than b to force modulo on b
        (("classical", None), 20, 15, 14, 20, {34: 1.0}),  # classical integer a path
        (("qf", 10), 0, 10, 42, {0: 1.0}, {42: 1.0}),  # additive identity with quantum zero
        (("classical", None), 0, 10, 42, 0, {42: 1.0}),  # additive identity with classical zero
        (("qf", 1), 0, 1, 1, {0: 1.0}, {1: 1.0}),  # single-qubit register boundary
        (("qf", 5), 7, 10, 15, {7: 1.0}, {22: 1.0}),  # mixed widths where no overflow occurs
        (("qf", 8), 100, 8, 50, {100: 1.0}, {150: 1.0}),  # equal-width quantum addition baseline
        (("qf", 16), 255, 4, 3, {255: 1.0}, {(3 + 255) % 16: 1.0}),  # strong wrap with much smaller b register
        (("classical", None), "10", 8, 5, "10", {6: 1.0}),  # string input accepted through int conversion
        (("classical", None), "", 8, 42, "", {42: 1.0}),  # empty string maps to zero contribution
        (("classical", None), np.int64(5), 8, 10, np.int64(5), {15: 1.0}),  # numpy integer compatibility
        (("classical", None), 17, 4, 0, 17, {1: 1.0}),  # classical overflow reduced modulo 2^b_bits
        (("classical", None), 255, 4, 7, 255, {6: 1.0}),  # large classical a reduced before addition
        (("classical", None), -3, 4, 5, -3, {2: 1.0}),  # negative classical input wraps modulo 2^b_bits
        (("classical", None), -1, 8, 0, -1, {255: 1.0}),  # -1 becomes all-ones pattern in target width
        (("qf_list", 4), 5, 4, 3, None, {8: 1.0}),  # list-of-qubits input path at equal width
        (("qf_list", 3), 7, 8, 100, None, {107: 1.0}),  # list-of-qubits input with wider target register
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
        (("qf", 13), 20, 14, 14, {34: 1.0}, {0: 1.0}),  # carry-out remains zero when sum fits
        (("qf", 13), 20, 13, 14, {34: 1.0}, {0: 1.0}),  # same arithmetic at equal widths
        (("classical", None), 20, 15, 14, {34: 1.0}, {0: 1.0}),  # classical a without overflow
        (("classical", None), 10, 8, 20, {30: 1.0}, {0: 1.0}),  # classical a on standard 8-bit target
        (("qf", 1), 1, 1, 1, {0: 1.0}, {1: 1.0}),  # minimal-width overflow sets carry-out
        (("classical", None), 255, 8, 128, {127: 1.0}, {1: 1.0}),  # high-value overflow case
        (("classical", None), 200, 8, 100, {44: 1.0}, {1: 1.0}),  # nontrivial overflow with wrap and carry
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
        (("classical", None), 5, 0, 1, {6: 1.0}),  # classical path with carry-in increment
        (("classical", None), 20, 0, 1, {21: 1.0}),  # larger classical addend with carry-in
        (("qf", 8), 50, 100, 1, {151: 1.0}),  # quantum path where carry-in changes result
        (("qf", 8), 50, 100, 0, {150: 1.0}),  # paired case to isolate carry-in effect
        (("qf", 8), 10, 0, 1, {11: 1.0}),  # carry-in into zero target register
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
        (("qf", 8), 100, 150, {251: 1.0}, {0: 1.0}),  # c_in active but still no overflow
        (("qf", 8), 200, 55, {0: 1.0}, {1: 1.0}),  # c_in pushes sum exactly over modulo boundary
        (("classical", None), 0, 255, {0: 1.0}, {1: 1.0}),  # zero addend with carry-in-only overflow
        (("classical", None), 1, 1, {3: 1.0}, {0: 1.0}),  # small numbers where carry remains zero
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
        (("qf", 10), 3, 11, 5, True, {8: 1.0}),  # controlled quantum addition executes when ctrl is on
        (("qf", 10), 3, 11, 5, False, {5: 1.0}),  # matched ctrl-off case leaves b unchanged
        (("classical", None), 5, 8, 10, True, {15: 1.0}),  # controlled classical addition executes
        (("classical", None), 5, 8, 10, False, {10: 1.0}),  # classical path blocked by ctrl-off
        (("classical", None), 0, 8, 42, True, {42: 1.0}),  # no-op addend under active control
        (("classical", None), 1, 1, 0, True, {1: 1.0}),  # one-bit controlled boundary case
        (("qf", 8), 50, 8, 100, False, {100: 1.0}),  # ctrl-off with c_in setup still must do nothing
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
        (55, 200, True, {0: 1.0}, {1: 1.0}),  # ctrl-on path with exact overflow boundary
        (55, 200, False, {200: 1.0}, {0: 1.0}),  # matched ctrl-off case proves full gating
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
        (("qf", 6), 5, 10, 7),  # ancilla cleanup for quantum addend path
        (("classical", None), 17, 10, 42),  # ancilla cleanup for classical addend path
    ],
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


@pytest.mark.parametrize(
    "mode, validation_case, should_raise",
    [
        ("static", "invalid_a", True),
        ("static", "invalid_b", True),
        ("static", "valid_list_b", False),
        ("dynamic", "invalid_a", True),
        ("dynamic", "invalid_b", True),
        ("dynamic", "valid_list_b", False),
    ],
)
def test_gidney_adder_invalid_inputs_raise_value_error(mode, validation_case, should_raise):
    """Validation should reject invalid pairs and accept list-based quantum b targets."""
    expected_msg = "classical-quantum|quantum-quantum"
    if mode == "static":
        b = QuantumFloat(4)
        b[:] = 0
        a = QuantumFloat(4)
        a[:] = 1

        if should_raise:
            with pytest.raises(ValueError, match=expected_msg):
                if validation_case == "invalid_a":
                    gidney_adder(object(), b)
                else:
                    gidney_adder(a, 7)
            return

        b_list = QuantumFloat(3)
        b_list[:] = 2
        gidney_adder(1, b_list[:])
        assert b_list.get_measurement() == {3: 1.0}
        return

    @boolean_simulation
    def main_invalid_a(n_bits):
        b_dyn = QuantumFloat(n_bits)
        b_dyn[:] = 0
        # Use a traced non-scalar array so validation in gidney_adder (not JAX)
        # raises the expected ValueError for an invalid classical input.
        gidney_adder(np.array([1, 0], dtype=np.int64), b_dyn)
        return measure(b_dyn)

    @boolean_simulation
    def main_invalid_b(n_bits):
        a_dyn = QuantumFloat(n_bits)
        a_dyn[:] = 1
        gidney_adder(a_dyn, 7)
        return measure(a_dyn)

    @boolean_simulation
    def main_valid_list_b(n_bits, a_val, b_val):
        b_dyn = QuantumFloat(n_bits)
        b_dyn[:] = b_val
        gidney_adder(a_val, b_dyn[:])
        return measure(b_dyn)

    if should_raise:
        with pytest.raises(ValueError, match=expected_msg):
            if validation_case == "invalid_a":
                main_invalid_a(3)
            else:
                main_invalid_b(3)
        return

    assert main_valid_list_b(3, 1, 2) == 3


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


@pytest.mark.parametrize(
    "n, use_c_in, use_c_out, use_ctrl",
    [
        (3, True, False, False),  # c_in-only overhead without control term
        (3, False, True, False),  # c_out-only ladder extension
        (5, True, True, False),  # combined c_in and c_out without control
        (3, False, False, True),  # ctrl-only linear T contribution at small n
        (5, True, False, True),  # ctrl plus c_in extra forward logical-AND
        (5, False, True, True),  # ctrl plus c_out ladder extension
        (8, True, True, True),  # all options enabled at larger width
    ],
)
def test_gidney_adder_t_count_formula_with_optional_inputs(
    n, use_c_in, use_c_out, use_ctrl
):
    """Verify explicit T-count formula from 4*(n-1) plus optional-input terms.

    Base (no optional inputs):
        T(n) = 4*(n-1)

    Optional contributions:
        - c_out extends the ladder by one bit: +4
        - ctrl adds a linear term from controlled two-control mcx gates:
                            (n + 1) * tau
          where tau is backend-specific T-count of one controlled two-control mcx
        - c_in with ctrl adds one extra forward gidney logical-AND: +4
    """
    # Calibrate backend-specific controlled two-control mcx T-cost (tau)
    # from a ctrl-only reference instance:
    # T = 4*(n_ref-1) + (n_ref+1)*tau
    n_ref = 3
    a_ref = QuantumFloat(n_ref)
    b_ref = QuantumFloat(n_ref)
    a_ref[:] = 2
    b_ref[:] = 3
    ctrl_ref = QuantumBool()
    x(ctrl_ref[0])
    with control(ctrl_ref):
        gidney_adder(a_ref, b_ref)
    qc_ref = b_ref.qs.compile(compile_mcm=True).transpile()
    ops_ref = qc_ref.count_ops()
    t_count_ref = ops_ref.get("t", 0) + ops_ref.get("t_dg", 0)
    numerator = t_count_ref - 4 * (n_ref - 1)
    assert numerator % (n_ref + 1) == 0
    tau = numerator // (n_ref + 1)

    a = QuantumFloat(n)
    b = QuantumFloat(n)
    a[:] = 2
    b[:] = 3

    c_in = QuantumBool() if use_c_in else None
    c_out = QuantumBool() if use_c_out else None
    ctrl_qb = QuantumBool() if use_ctrl else None

    if c_in is not None:
        x(c_in[0])
    if ctrl_qb is not None:
        x(ctrl_qb[0])

    if ctrl_qb is not None:
        with control(ctrl_qb):
            gidney_adder(a, b, c_in=c_in, c_out=c_out)
    else:
        gidney_adder(a, b, c_in=c_in, c_out=c_out)

    qc = b.qs.compile(compile_mcm=True).transpile()
    ops = qc.count_ops()
    t_count = ops.get("t", 0) + ops.get("t_dg", 0)

    if not use_ctrl:
        expected_t_count = 4 * (n - 1 + int(use_c_out))
    else:
        expected_t_count = (
            4 * (n - 1 + int(use_c_out) + int(use_c_in))
            + (n + 1) * tau
        )

    assert t_count == expected_t_count


@pytest.mark.parametrize(
    "N, L",
    [
        (2, 2),  # equal widths small baseline
        (2, 5),  # narrow a into wider b
        (3, 4),  # near-equal widths with modulo on b
        (4, 3),  # wider a into narrower b
        (5, 5),  # larger equal-width dynamic case
    ],
)
def test_gidney_adder_jasp_mode(N, L):
    """Verify the function works as expected in dynamic (jasp) mode.
    This test covers both equal and unequal input sizes with parametrized widths."""

    @boolean_simulation
    def main(n_bits, l_bits, j, k):
        A = QuantumFloat(n_bits)
        B = QuantumFloat(l_bits)
        A[:] = j
        B[:] = k
        gidney_adder(A, B)
        return measure(A), measure(B)

    j_max = 1 << N
    k_max = 1 << L
    mod = 1 << L
    for j in range(j_max):
        for k in range(k_max):
            A, B = main(N, L, j, k)
            assert A == j
            assert B == (k + j) % mod


@pytest.mark.parametrize(
    "N, L",
    [
        (2, 2),  # equal widths under active control
        (2, 5),  # controlled execution with wider b
        (3, 4),  # controlled mixed-width dynamic path
        (4, 3),  # controlled modulo behavior with narrower b
        (5, 5),  # larger controlled equal-width case
    ],
)
def test_gidney_adder_jasp_mode_with_control(N, L):
    """Verify controlled execution in dynamic mode for parametrized input widths."""

    @boolean_simulation
    def main(n_bits, l_bits, j, k):
        A = QuantumFloat(n_bits)
        B = QuantumFloat(l_bits)
        A[:] = j
        B[:] = k
        qbl = QuantumBool()
        qbl.flip()

        with control(qbl):
            gidney_adder(A, B)
        return measure(A), measure(B)

    j_max = 1 << N
    k_max = 1 << L
    mod = 1 << L
    for j in range(j_max):
        for k in range(k_max):
            A, B = main(N, L, j, k)
            assert A == j
            assert B == (k + j) % mod


@pytest.mark.parametrize(
    "N",
    [2, 3, 4, 5],  # sweep register sizes for carry-in dynamic path
)
def test_gidney_adder_jasp_mode_with_carry_in(N):
    """Carry-in in dynamic mode with quantum a for multiple register sizes."""

    @boolean_simulation
    def main(n_bits, j, k):
        A = QuantumFloat(n_bits)
        B = QuantumFloat(n_bits)
        A[:] = j
        B[:] = k
        c_in = QuantumBool()
        c_in.flip()
        gidney_adder(A, B, c_in=c_in)
        return measure(A), measure(B)

    vals_max = 1 << N
    mod = 1 << N
    for j in range(vals_max):
        for k in range(vals_max):
            A, B = main(N, j, k)
            assert A == j
            assert B == (k + j + 1) % mod


@pytest.mark.parametrize(
    "N",
    [2, 3, 4, 5],  # sweep register sizes for ctrl-off no-op behavior
)
def test_gidney_adder_jasp_mode_ctrl_off(N):
    """In dynamic mode, ctrl=|0⟩ means b is untouched across multiple sizes."""

    @boolean_simulation
    def main(n_bits, j, k):
        A = QuantumFloat(n_bits)
        B = QuantumFloat(n_bits)
        A[:] = j
        B[:] = k
        qbl = QuantumBool()

        with control(qbl):
            gidney_adder(A, B)
        return measure(A), measure(B)

    vals_max = 1 << N
    for j in range(vals_max):
        for k in range(vals_max):
            A, B = main(N, j, k)
            assert A == j
            assert B == k