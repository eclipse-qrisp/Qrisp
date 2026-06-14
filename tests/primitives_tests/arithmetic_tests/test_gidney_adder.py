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
    BigInteger,
    QuantumBool,
    QuantumFloat,
    QuantumVariable,
    boolean_simulation,
    control,
    cuccaro_adder,
    gidney_adder,
    mcx,
    measure,
    x,
)


@pytest.mark.parametrize(
    "a_spec, a_val, b_bits, b_val, expected_a, expected_b",
    [
        (
            ("qf", 13),
            20,
            14,
            14,
            {20: 1.0},
            {34: 1.0},
        ),  # quantum a and wider b without wrap
        (
            ("qf", 10),
            15,
            5,
            7,
            {15: 1.0},
            {(7 + 15) % 32: 1.0},
        ),  # quantum a larger than b to force modulo on b
        (("classical", None), 20, 15, 14, 20, {34: 1.0}),  # classical integer a path
        (("qf", 1), 0, 1, 1, {0: 1.0}, {1: 1.0}),  # single-qubit register boundary
        (
            ("classical", None),
            "10",
            8,
            5,
            "10",
            {6: 1.0},
        ),  # string input accepted through int conversion
        (
            ("classical", None),
            np.int64(5),
            8,
            10,
            np.int64(5),
            {15: 1.0},
        ),  # numpy integer compatibility
        (
            ("classical", None),
            -3,
            4,
            5,
            -3,
            {2: 1.0},
        ),  # negative classical input wraps modulo 2^b_bits
        (
            ("qf_list", 3),
            7,
            8,
            100,
            None,
            {107: 1.0},
        ),  # list-of-qubits input with wider target register
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
    "a_spec, a_val, b_bits, b_val, c_in_val, use_c_out, expected_b, expected_cout",
    [
        (
            ("qf", 13),
            20,
            14,
            14,
            None,
            True,
            {34: 1.0},
            {0: 1.0},
        ),  # carry-out remains zero when sum fits
        (
            ("qf", 1),
            1,
            1,
            1,
            None,
            True,
            {0: 1.0},
            {1: 1.0},
        ),  # minimal-width overflow sets carry-out
        (
            ("classical", None),
            255,
            8,
            128,
            None,
            True,
            {127: 1.0},
            {1: 1.0},
        ),  # high-value overflow case
        (
            ("classical", None),
            10,
            8,
            20,
            None,
            True,
            {30: 1.0},
            {0: 1.0},
        ),  # classical a on standard 8-bit target
        (
            ("classical", None),
            5,
            8,
            0,
            1,
            False,
            {6: 1.0},
            None,
        ),  # classical path with carry-in increment
        (
            ("qf", 8),
            50,
            8,
            100,
            1,
            False,
            {151: 1.0},
            None,
        ),  # quantum path where carry-in changes result
        (
            ("qf", 8),
            50,
            8,
            100,
            0,
            False,
            {150: 1.0},
            None,
        ),  # paired case to isolate carry-in effect
        (
            ("qf", 8),
            200,
            8,
            55,
            1,
            True,
            {0: 1.0},
            {1: 1.0},
        ),  # c_in pushes sum exactly over modulo boundary
        (
            ("classical", None),
            0,
            8,
            255,
            1,
            True,
            {0: 1.0},
            {1: 1.0},
        ),  # zero addend with carry-in-only overflow
        (
            ("classical", None),
            1,
            8,
            1,
            1,
            True,
            {3: 1.0},
            {0: 1.0},
        ),  # small numbers where carry remains zero
    ],
)
def test_gidney_adder_with_carry_options(
    a_spec, a_val, b_bits, b_val, c_in_val, use_c_out, expected_b, expected_cout
):
    """Verify carry-in and carry-out combinations for b += a (+ c_in)."""
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

    kwargs = {}
    if c_in_val is not None:
        c_in = QuantumBool()
        if c_in_val:
            x(c_in[0])
        kwargs["c_in"] = c_in

    c_out = None
    if use_c_out:
        c_out = QuantumBool()
        kwargs["c_out"] = c_out

    gidney_adder(a, b, **kwargs)

    assert b.get_measurement() == expected_b
    if expected_cout is not None:
        assert c_out is not None
        assert c_out.get_measurement() == expected_cout
    # a is unchanged by the adder
    if isinstance(a, QuantumFloat):
        assert a.get_measurement() == {a_val: 1.0}


@pytest.mark.parametrize(
    "a_spec, a_val, b_bits, b_val, ctrl_on, expected_b",
    [
        (
            ("qf", 10),
            3,
            11,
            5,
            True,
            {8: 1.0},
        ),  # controlled quantum addition executes when ctrl is on
        (
            ("qf", 10),
            3,
            11,
            5,
            False,
            {5: 1.0},
        ),  # matched ctrl-off case leaves b unchanged
        (
            ("classical", None),
            5,
            8,
            10,
            True,
            {15: 1.0},
        ),  # controlled classical addition executes
        (
            ("classical", None),
            5,
            8,
            10,
            False,
            {10: 1.0},
        ),  # classical path blocked by ctrl-off
    ],
)
def test_gidney_adder_with_control(a_spec, a_val, b_bits, b_val, ctrl_on, expected_b):
    """Verify controlled addition only fires when :math:`ctrl=\\ket{1}`."""
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

    with control(ctrl_qb):
        gidney_adder(a, b)

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
        (
            55,
            200,
            True,
            {0: 1.0},
            {1: 1.0},
        ),  # ctrl-on path with exact overflow boundary
        (
            55,
            200,
            False,
            {200: 1.0},
            {0: 1.0},
        ),  # matched ctrl-off case proves full gating
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


@pytest.mark.parametrize("a_str", ["10201", "abc"])
def test_gidney_adder_invalid_binary_string_raises_value_error(a_str):
    """Invalid binary strings should raise from the string-to-int conversion branch."""
    b = QuantumFloat(8)
    b[:] = 3

    with pytest.raises(ValueError, match="base 2"):
        gidney_adder(a_str, b)


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
        a_var = a
    elif kind == "qf_list":
        qf = QuantumFloat(n)
        qf[:] = a_val
        a = qf[:]
        a_var = qf
    else:
        a = a_val
        a_var = None

    b = QuantumFloat(b_bits)
    b[:] = b_val

    user_vars = [v for v in [a_var, b] if v is not None]
    user_ids = {id(v) for v in user_vars}
    gidney_adder(a, b)
    # Any variable in the session whose identity doesn't match a user-created
    # variable is a leaked ancilla — regardless of naming conventions.
    leaked = [v for v in b.qs.qv_list if id(v) not in user_ids]
    assert leaked == []


@pytest.mark.parametrize("carry_kind", ["in", "out"])
def test_gidney_adder_raw_qubit_carry_wires(carry_kind):
    """Pass raw Qubits instead of QuantumBool for carry wires."""
    carry_var = QuantumVariable(1)
    b = QuantumFloat(8)

    if carry_kind == "in":
        x(carry_var[0])
        b[:] = 10
        gidney_adder(5, b, c_in=carry_var[0])
        assert b.get_measurement() == {16: 1.0}
    else:
        b[:] = 200
        gidney_adder(100, b, c_out=carry_var[0])
        assert b.get_measurement() == {44: 1.0}
        assert carry_var.get_measurement() == {"1": 1.0}


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
    "kind, a_val, b_val, use_cin, use_cout, expected_b, expected_cout",
    [
        ("classical", 0, 0, False, False, 0, None),
        ("classical", 1, 0, False, False, 1, None),
        ("classical", 1, 1, False, False, 0, None),
        ("classical", 0, 0, False, True, 0, 0),
        ("classical", 1, 1, False, True, 0, 1),
        ("classical", 0, 0, True, False, 1, None),
        ("classical", 1, 0, True, False, 0, None),
        ("classical", 1, 1, True, False, 1, None),
        ("classical", 0, 0, True, True, 1, 0),
        ("classical", 1, 1, True, True, 1, 1),
        ("quantum", 0, 0, False, False, 0, None),
        ("quantum", 1, 0, False, False, 1, None),
        ("quantum", 1, 1, False, False, 0, None),
        ("quantum", 0, 0, False, True, 0, 0),
        ("quantum", 1, 1, False, True, 0, 1),
        ("quantum", 0, 0, True, False, 1, None),
        ("quantum", 1, 0, True, False, 0, None),
        ("quantum", 1, 1, True, False, 1, None),
        ("quantum", 0, 0, True, True, 1, 0),
        ("quantum", 1, 1, True, True, 1, 1),
    ],
)
def test_gidney_adder_single_qubit(
    kind, a_val, b_val, use_cin, use_cout, expected_b, expected_cout
):
    """n == 1 edge case: carry chain is skipped via control(n > 1)."""
    if kind == "classical":
        a = a_val
    else:
        a = QuantumFloat(1)
        a[:] = a_val

    b = QuantumFloat(1)
    b[:] = b_val

    kwargs = {}
    if use_cin:
        c_in = QuantumBool()
        x(c_in[0])
        kwargs["c_in"] = c_in
    if use_cout:
        c_out = QuantumBool()
        kwargs["c_out"] = c_out

    gidney_adder(a, b, **kwargs)

    assert b.get_measurement() == {expected_b: 1.0}
    if expected_cout is not None:
        assert c_out.get_measurement() == {expected_cout: 1.0}


@pytest.mark.parametrize(
    "n_bits, a_val, b_val, spec, expected",
    [
        (1, 0, 0, None, 0),  # n=1 edge case (item 8)
        (1, 1, 0, None, 1),
        (1, 1, 1, None, 0),
        (4, -3, 10, None, 7),  # negative / large a
        (4, -1, 1, None, 0),
        (4, 20, 5, None, 9),
        (4, 3, 5, "numpy", 8),  # numpy int a
        (4, 5, 10, "dqa", 15),  # DynamicQubitArray as b
    ],
)
def test_gidney_adder_classical_a_dynamic(n_bits, a_val, b_val, spec, expected):
    """Classical a + quantum b in dynamic mode (varying input types)."""

    @boolean_simulation
    def main(bits, av, bv):
        b_qf = QuantumFloat(bits)
        b_qf[:] = bv
        b_arg = b_qf.reg if spec == "dqa" else b_qf
        gidney_adder(av, b_arg)
        return measure(b_qf)

    a = np.int64(a_val) if spec == "numpy" else a_val
    assert main(n_bits, a, b_val) == expected


def test_gidney_adder_single_qubit_no_ancilla_leak():
    """n == 1: QuantumVariable(0) should not leak."""
    b = QuantumFloat(1)
    b[:] = 0
    initial_ids = {id(b)}
    gidney_adder(1, b)
    leaked = [v for v in b.qs.qv_list if id(v) not in initial_ids]
    assert leaked == []


@pytest.mark.parametrize(
    "desc, a_spec, a_val, b_bits, b_val, carry_args, expected",
    [
        # Quantum a + c_in + c_out without ctrl (item 6)
        (
            "quantum a + c_in + c_out, no ctrl",
            ("qf", 4),
            3,
            4,
            5,
            {"c_in": 1, "c_out": True},
            {"b": {9: 1.0}, "cout": {0: 1.0}},
        ),
        # n == 1 with control (item 8)
        ("n=1 quantum a + ctrl", ("qf", 1), 1, 1, 1, {"ctrl": True}, {"b": {0: 1.0}}),
        (
            "n=1 classical a + ctrl",
            ("classical", None),
            1,
            1,
            1,
            {"ctrl": True},
            {"b": {0: 1.0}},
        ),
        # Empty quantum a list — identity operation (item 2)
        ("empty quantum a list", ("empty_list", None), None, 4, 7, {}, {"b": {7: 1.0}}),
        # Classical a wider than b (item 7)
        (
            "classical a=257 in 8-bit b",
            ("classical", None),
            257,
            8,
            0,
            {},
            {"b": {1: 1.0}},
        ),
        # String a + c_out (item 4)
        (
            "string a + c_out",
            ("classical", None),
            "10",
            8,
            5,
            {"c_out": True},
            {"b": {6: 1.0}, "cout": {0: 1.0}},
        ),
    ],
)
def test_gidney_adder_additional_edge_cases(
    desc, a_spec, a_val, b_bits, b_val, carry_args, expected
):
    """Cover remaining static-mode gaps identified during code review."""
    kind, n = a_spec
    if kind == "qf":
        a = QuantumFloat(n)
        a[:] = a_val
    elif kind == "empty_list":
        a = []
    else:
        a = a_val

    b = QuantumFloat(b_bits)
    b[:] = b_val

    kwargs = {}
    if "c_in" in carry_args:
        c_in = QuantumBool()
        if carry_args["c_in"]:
            x(c_in[0])
        kwargs["c_in"] = c_in
    if "c_out" in carry_args and carry_args["c_out"]:
        c_out = QuantumBool()
        kwargs["c_out"] = c_out

    if "ctrl" in carry_args and carry_args["ctrl"]:
        ctrl = QuantumBool()
        x(ctrl[0])
        with control(ctrl):
            gidney_adder(a, b, **kwargs)
    else:
        gidney_adder(a, b, **kwargs)

    assert b.get_measurement() == expected["b"]
    if "cout" in expected:
        assert c_out.get_measurement() == expected["cout"]


@pytest.mark.parametrize(
    "mode, validation_case, should_raise",
    [
        ("static", "invalid_a", True),
        ("static", "invalid_a_list", True),
        ("static", "invalid_b", True),
        ("static", "invalid_b_empty_list", True),
        ("static", "valid_list_b", False),
        ("dynamic", "invalid_a", True),
        ("dynamic", "invalid_a_list", True),
        ("dynamic", "invalid_b", True),
        ("dynamic", "invalid_b_empty_list", True),
        ("dynamic", "valid_list_b", False),
    ],
)
def test_gidney_adder_invalid_inputs_raise_value_error(
    mode, validation_case, should_raise
):
    """Validation should reject invalid pairs and accept list-based quantum b targets."""
    expected_msg = "gidney_adder expects inputs"
    if mode == "static":
        b = QuantumFloat(4)
        b[:] = 0
        a = QuantumFloat(4)
        a[:] = 1

        if should_raise:
            with pytest.raises(ValueError, match=expected_msg):
                if validation_case == "invalid_a":
                    gidney_adder(object(), b)
                elif validation_case == "invalid_a_list":
                    gidney_adder([1, 0], b)
                elif validation_case == "invalid_b":
                    gidney_adder(a, 7)
                else:
                    gidney_adder(a, [])
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
    def main_invalid_a_list(n_bits):
        b_dyn = QuantumFloat(n_bits)
        b_dyn[:] = 0
        gidney_adder([1, 0], b_dyn)
        return measure(b_dyn)

    @boolean_simulation
    def main_invalid_b(n_bits):
        a_dyn = QuantumFloat(n_bits)
        a_dyn[:] = 1
        gidney_adder(a_dyn, 7)
        return measure(a_dyn)

    @boolean_simulation
    def main_invalid_b_empty_list(n_bits):
        a_dyn = QuantumFloat(n_bits)
        a_dyn[:] = 1
        gidney_adder(a_dyn, [])
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
            elif validation_case == "invalid_a_list":
                main_invalid_a_list(3)
            elif validation_case == "invalid_b":
                main_invalid_b(3)
            else:
                main_invalid_b_empty_list(3)
        return

    assert main_valid_list_b(3, 1, 2) == 3


def test_gidney_adder_t_depth_significantly_lower_than_cuccaro():
    """Compare gidney and cuccaro adders for T-depth, measurement depth, and T-state scaling."""

    def measurement_depth_indicator(op):
        return int(op.name == "measure")

    ctrl_c = QuantumVariable(2)
    tgt_c = QuantumVariable(1)
    mcx(ctrl_c, tgt_c[0], method="gray")
    tau_cuccaro_qc = tgt_c.qs.compile(compile_mcm=True).transpile()
    tau_cuccaro_ops = tau_cuccaro_qc.count_ops()
    tau_cuccaro = tau_cuccaro_ops.get("t", 0) + tau_cuccaro_ops.get("t_dg", 0)

    ctrl_g = QuantumVariable(2)
    tgt_g = QuantumVariable(1)
    mcx(ctrl_g, tgt_g[0], method="gidney")
    tau_gidney_qc = tgt_g.qs.compile(compile_mcm=True).transpile()
    tau_gidney_ops = tau_gidney_qc.count_ops()
    tau_gidney = tau_gidney_ops.get("t", 0) + tau_gidney_ops.get("t_dg", 0)

    assert tau_cuccaro > 0
    assert tau_gidney > 0

    t_count_gap_offsets = []

    for qf_size in [8, 12, 16]:
        a_cuccaro = QuantumFloat(qf_size)
        b_cuccaro = QuantumFloat(qf_size)
        a_cuccaro[:] = 5
        b_cuccaro[:] = 10
        cuccaro_adder(a_cuccaro, b_cuccaro)
        cuccaro_compiled = b_cuccaro.qs.compile()
        cuccaro_t_depth = cuccaro_compiled.t_depth()
        cuccaro_measurement_depth = cuccaro_compiled.depth(
            depth_indicator=measurement_depth_indicator
        )
        cuccaro_transpiled = b_cuccaro.qs.compile(compile_mcm=True).transpile()
        cuccaro_ops = cuccaro_transpiled.count_ops()
        cuccaro_raw_t = cuccaro_ops.get("t", 0) + cuccaro_ops.get("t_dg", 0)

        a_gidney = QuantumFloat(qf_size)
        b_gidney = QuantumFloat(qf_size)
        a_gidney[:] = 5
        b_gidney[:] = 10
        gidney_adder(a_gidney, b_gidney)
        gidney_compiled = b_gidney.qs.compile()
        gidney_t_depth = gidney_compiled.t_depth()
        gidney_measurement_depth = gidney_compiled.depth(
            depth_indicator=measurement_depth_indicator
        )
        gidney_transpiled = b_gidney.qs.compile(compile_mcm=True).transpile()
        gidney_ops = gidney_transpiled.count_ops()
        gidney_raw_t = gidney_ops.get("t", 0) + gidney_ops.get("t_dg", 0)

        # T-depth claim: Gidney achieves lower T-depth than Cuccaro.
        assert gidney_t_depth < cuccaro_t_depth

        # As described on page 1 of the main paper, both adders have the same
        # measurement depth behavior for n-bit inputs.
        assert gidney_measurement_depth == cuccaro_measurement_depth

        # T-state scaling claim: normalize backend-dependent raw T-counts and
        # compare against the paper's 8n (Cuccaro) and 4n (Gidney) logical laws.
        assert cuccaro_raw_t % tau_cuccaro == 0
        assert gidney_raw_t % tau_gidney == 0

        cuccaro_t_count = (cuccaro_raw_t // tau_cuccaro) * 4
        gidney_t_count = (gidney_raw_t // tau_gidney) * 4

        t_count_gap_offsets.append((cuccaro_t_count - gidney_t_count) - 4 * qf_size)

    # Paper-derived gap claim:
    # (8n + O(1)) - (4n + O(1)) = 4n + O(1).
    # We test this by checking that (T_cuccaro - T_gidney - 4n) stays bounded.
    assert max(t_count_gap_offsets) - min(t_count_gap_offsets) <= 4


@pytest.mark.parametrize(
    "n, use_c_in, use_c_out, use_ctrl",
    [
        (3, True, False, False),  # c_in-only overhead without control term
        (3, False, True, False),  # c_out-only ladder extension
        (3, False, False, True),  # ctrl-only linear T contribution at small n
        (8, False, False, True),  # ctrl-only larger n for 8n+O(1) scaling check
        (5, True, True, True),  # all options enabled together
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

    # defined to take into consideration how a backend transpiles the controlled
    # two-control mcx, which is the dominant contribution to the ctrl term's T-count
    # used to verify the T count as described in abstract and in the formula comments above.
    tau = numerator // (n_ref + 1)
    ctrl_only_const = t_count_ref - 8 * n_ref

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
        expected_t_count = 4 * (n - 1 + int(use_c_out) + int(use_c_in)) + (n + 1) * tau

    assert t_count == expected_t_count

    # Verify the abstract's 8n + O(1) scaling for ctrl-only (ideal tau=4).
    if use_ctrl and (not use_c_in) and (not use_c_out):
        if tau == 4:
            assert t_count == 8 * n + ctrl_only_const


@pytest.mark.parametrize(
    "N, L, control_state, use_carry_in",
    [
        # no control cases
        (2, 2, "no_control", False),  # equal widths baseline
        (4, 3, "no_control", False),  # wider a into narrower b
        (5, 5, "no_control", False),  # larger equal-width
        # control on cases
        (2, 2, "control_on", False),  # equal widths under active control
        (4, 3, "control_on", False),  # controlled modulo behavior
        (5, 5, "control_on", False),  # larger controlled equal-width
        # control on with carry_in cases (equal widths N=L)
        (2, 2, "control_on", True),  # small register with carry_in
        (5, 5, "control_on", True),  # larger register with carry_in
        # control off cases (equal widths N=L)
        (2, 2, "control_off", False),  # ctrl-off no-op small
        (5, 5, "control_off", False),  # ctrl-off no-op larger
    ],
)
def test_gidney_adder_jasp_mode_variants(N, L, control_state, use_carry_in):
    """Verify gidney_adder in dynamic mode with optional control and carry_in."""

    @boolean_simulation
    def main(n_bits, l_bits, j, k):
        A = QuantumFloat(n_bits)
        B = QuantumFloat(l_bits)
        A[:] = j
        B[:] = k

        kwargs = {}
        if use_carry_in:
            c_in = QuantumBool()
            c_in.flip()
            kwargs["c_in"] = c_in

        if control_state == "no_control":
            gidney_adder(A, B, **kwargs)
        else:
            qbl = QuantumBool()
            if control_state == "control_on":
                qbl.flip()
            with control(qbl):
                gidney_adder(A, B, **kwargs)

        return measure(A), measure(B)

    j_max = 1 << N
    k_max = 1 << L
    mod = 1 << L
    for j in range(j_max):
        for k in range(k_max):
            A, B = main(N, L, j, k)
            assert A == j

            if control_state == "control_off":
                expected_B = k
            elif use_carry_in:
                expected_B = (k + j + 1) % mod
            else:
                expected_B = (k + j) % mod

            assert B == expected_B


@pytest.mark.parametrize(
    "n_bits, a_val, b_val",
    [
        (4, 5, 3),
        (6, 17, 12),
    ],
)
def test_gidney_adder_jasp_mode_with_biginteger_classical_input(n_bits, a_val, b_val):
    """Cover traced-classical input path where ``a`` is a ``BigInteger`` with ``get_bit``."""

    @boolean_simulation
    def main(bits, a_num, b_num):
        b_dyn = QuantumFloat(bits)
        b_dyn[:] = b_num
        a_big = BigInteger.create(a_num, 1)
        gidney_adder(a_big, b_dyn)
        return measure(b_dyn)

    mod = 1 << n_bits
    assert main(n_bits, a_val, b_val) == (a_val + b_val) % mod


@pytest.mark.parametrize("N, L, a_val, b_val", [(5, 3, 17, 5)])
def test_gidney_adder_quantum_a_wider_than_b_dynamic(N, L, a_val, b_val):
    """Quantum a wider than b in dynamic/tracing mode.

    Exercises the truncation + extension ancilla path.
    """

    @boolean_simulation
    def main(n, bit_len, av, bv):
        a_qf = QuantumFloat(n)
        a_qf[:] = av
        b_qf = QuantumFloat(bit_len)
        b_qf[:] = bv
        gidney_adder(a_qf, b_qf)
        return measure(b_qf)

    expected = (a_val + b_val) % (1 << L)
    assert main(N, L, a_val, b_val) == expected


def test_gidney_adder_invalid_binary_string_dynamic():
    """Invalid binary string a raises ValueError in dynamic/tracing mode."""

    @boolean_simulation
    def main():
        b = QuantumFloat(4)
        b[:] = 0
        gidney_adder("abc", b)
        return measure(b)

    with pytest.raises(ValueError, match="base 2"):
        main()



@pytest.mark.parametrize("n_bits, a_val, b_val", [(4, 5, 10)])
def test_gidney_adder_classical_a_cin_dynamic(n_bits, a_val, b_val):
    """Classical a + c_in in dynamic mode."""

    @boolean_simulation
    def main(bits, av, bv):
        b_qf = QuantumFloat(bits)
        b_qf[:] = bv
        c_in = QuantumBool()
        c_in.flip()
        gidney_adder(av, b_qf, c_in=c_in)
        return measure(b_qf)

    assert main(n_bits, a_val, b_val) == (a_val + b_val + 1) % (1 << n_bits)


@pytest.mark.parametrize(
    "kind, n_bits, a_val, b_val",
    [
        ("classical", 4, 5, 10),
        ("classical", 4, 12, 7),
        ("classical", 1, 1, 1),
        ("quantum", 4, 5, 10),
        ("quantum", 4, 12, 7),
        ("quantum", 1, 1, 1),
    ],
)
def test_gidney_adder_cout_dynamic(kind, n_bits, a_val, b_val):
    """C_out in dynamic mode.

    Covers both classical and quantum a paths.
    """

    @boolean_simulation
    def main(bits, av, bv):
        if kind == "quantum":
            a_qf = QuantumFloat(bits)
            a_qf[:] = av
            av = a_qf
        b_qf = QuantumFloat(bits)
        b_qf[:] = bv
        c_out = QuantumBool()
        gidney_adder(av, b_qf, c_out=c_out)
        return measure(b_qf), measure(c_out)

    mod = 1 << n_bits
    total = a_val + b_val
    expected_b = total % mod
    expected_cout = 1 if total >= mod else 0
    b_res, cout_res = main(n_bits, a_val, b_val)
    assert b_res == expected_b
    assert cout_res == expected_cout


@pytest.mark.parametrize("n_bits, a_val, b_val", [(4, 3, 10)])
def test_gidney_adder_cin_cout_ctrl_dynamic(n_bits, a_val, b_val):
    """c_in + c_out + ctrl in dynamic mode.

    Also exercises the ``[c_in_qb] + b_qbs`` path (__radd__).
    """

    @boolean_simulation
    def main(bits, av, bv):
        a_qf = QuantumFloat(bits)
        a_qf[:] = av
        b_qf = QuantumFloat(bits)
        b_qf[:] = bv
        c_in = QuantumBool()
        c_in.flip()
        c_out = QuantumBool()
        ctrl = QuantumBool()
        ctrl.flip()
        with control(ctrl):
            gidney_adder(a_qf, b_qf, c_in=c_in, c_out=c_out)
        return measure(b_qf), measure(c_out)

    mod = 1 << n_bits
    total = a_val + b_val + 1
    expected_b = total % mod
    expected_cout = 1 if total >= mod else 0
    b_res, cout_res = main(n_bits, a_val, b_val)
    assert b_res == expected_b
    assert cout_res == expected_cout


@pytest.mark.parametrize(
    "a_str, b_bits, b_val",
    [
        ("1010", 4, 5),
        ("11", 2, 1),
    ],
)
def test_gidney_adder_string_a_dynamic(a_str, b_bits, b_val):
    """String a input in dynamic/tracing mode (string captured as closure)."""

    @boolean_simulation
    def main(bits, bv):
        s = a_str
        b_qf = QuantumFloat(bits)
        b_qf[:] = bv
        gidney_adder(s, b_qf)
        return measure(b_qf)

    a_int = int(a_str[::-1], 2)
    expected = (a_int + b_val) % (1 << b_bits)
    assert main(b_bits, b_val) == expected


@pytest.mark.parametrize(
    "n_bits, a_val, b_val",
    [
        (4, 5, 10),
        (8, 17, 42),
    ],
)
def test_gidney_adder_ancilla_cleanup_dynamic(n_bits, a_val, b_val):
    """Ancilla cleanup verification in dynamic/tracing mode."""

    @boolean_simulation
    def main(bits, av, bv):
        b_qf = QuantumFloat(bits)
        b_qf[:] = bv
        gidney_adder(av, b_qf)
        user_ids = {id(b_qf)}
        leaked = [v for v in b_qf.qs.qv_list if id(v) not in user_ids]
        assert leaked == []
        return measure(b_qf)

    expected = (a_val + b_val) % (1 << n_bits)
    assert main(n_bits, a_val, b_val) == expected


def test_gidney_adder_jaspr_mode():
    """Verify gidney_adder works as expected in dynamic mode."""

    @boolean_simulation
    def main(N, L, j, k):

        A = QuantumFloat(N)
        B = QuantumFloat(L)
        A[:] = j
        B[:] = k

        gidney_adder(j, B)
        return measure(A), measure(B)

    for N in range(1, 5):
        for L in range(1, 5):
            for j in range(2**N):
                for k in range(2**L):
                    A, B = main(N, L, j, k)
                    assert A == j
                    assert B == (k + j) % (2**L)


def test_gidney_adder_dynamic_mode_with_control():
    """Verify gidney_adder is triggered when the control qubit is in the |1> state
    in dynamic mode."""

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

    for N in range(1, 4):
        for L in range(1, 4):
            for j in range(2**N):
                for k in range(2**L):
                    A, B = main(N, L, j, k)
                    assert A == j
                    assert B == (k + j) % (2**L)
